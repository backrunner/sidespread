#!/usr/bin/env python3
"""P4: Export UniverSR ConvNeXt-V2 backbone to ONNX and save reference I/O for Rust validation.

Produces:
  - models/universr_backbone.onnx     (single ODE-step network)
  - models/universr_config.json       (STFT/ODE/CFG params for Rust)
  - tests/fixtures/universr_ref.npz   (reference inputs + output for numerical validation)
"""
import os, sys, json, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "external", "universr"))

import numpy as np
import torch
import torch.nn as nn
import yaml
from huggingface_hub import hf_hub_download

from universr.models.unet import ConvNeXtUNetCond
from universr.utils.spectral_ops import AmplitudeCompressedComplexSTFT
from universr.flow.path import OriginalCFMPath

MODELS = os.path.join(ROOT, "models")
FIX = os.path.join(ROOT, "tests", "fixtures")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(FIX, exist_ok=True)

device = "cpu"
torch.manual_seed(42); np.random.seed(42)

# --- load config + weights ---
cfg_path = hf_hub_download(repo_id="woongzip1/universr-audio", filename="config.yaml")
wpath    = hf_hub_download(repo_id="woongzip1/universr-audio", filename="pytorch_model.bin")
with open(cfg_path) as f:
    config = yaml.safe_load(f)

model = ConvNeXtUNetCond(**config["model"])
sd = torch.load(wpath, map_location="cpu", weights_only=True)
model.load_state_dict(sd)
model.to(device).eval()

transform = AmplitudeCompressedComplexSTFT(**config["transform"])
transform.to(device)
path = OriginalCFMPath(**config["path"]["init_args"])

print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# --- build a minimum-context LR input (16kHz effective bandwidth at 48k) ---
effective_sr = 16000; sr_khz = 16; target_sr = 48000
MIN_SAMPLES = 32768
T = MIN_SAMPLES
# Make a tonal + noise signal, then bandwidth-limit it (down→up resample).
t_axis = np.arange(T)
wav_clean = 0.3*np.sin(2*math.pi*440*t_axis/target_sr) \
          + 0.1*np.sin(2*math.pi*8000*t_axis/target_sr)
wav_clean = torch.tensor(wav_clean, dtype=torch.float32)
import torchaudio
lr = torchaudio.functional.resample(wav_clean, target_sr, effective_sr)
lr = torchaudio.functional.resample(lr, effective_sr, target_sr)  # bandwidth-limited 48k
orig_len = lr.shape[-1]
lr = torch.nn.functional.pad(lr, (0, max(0, MIN_SAMPLES - lr.shape[-1])))
lr = lr.unsqueeze(0).unsqueeze(0)  # [1,1,T]

# --- preprocess: STFT → compressed complex → [B,2,F-1,T] ---
def preprocess(waveform):
    spec = transform(waveform)                 # [B,C,F,T] complex
    real = torch.view_as_real(spec.squeeze(1)) # [B,F,T,2]
    real = real.permute(0,3,1,2)               # [B,2,F,T]
    return real[:, :, :-1, :]                  # drop Nyquist → [B,2,F-1=511,T]

def postprocess(spec):
    spec = torch.nn.functional.pad(spec, [0,0,0,1], value=0)  # restore Nyquist
    spec = spec.permute(0,2,3,1).contiguous()
    spec = torch.view_as_complex(spec)
    return transform.invert(spec)              # [B,T]

Y = preprocess(lr)                              # [1,2,511,T]
lr_bin_count = model.sr_to_lr_bins[sr_khz]      # 170 for 16k
hf_start_bin = model.total_freq_bins - model.hr_freq_bins  # 512-432=80
Y_lr = Y[:, :, :lr_bin_count, :]               # [1,2,170,T]
Y_hr_shape = Y[:, :, hf_start_bin:, :]          # [1,2,432,T] for shape ref
x0 = path.sample_source(Y_hr_shape)             # randn [1,2,432,T]
print(f"Y: {Y.shape}  Y_lr: {Y_lr.shape}  x0: {x0.shape}  T_frames: {Y.shape[-1]}")

# --- run reference ODE (midpoint, 4 steps, CFG scale 1.5) ---
guidance_scale = 1.5
ts = torch.linspace(0, 1, 5)                    # 4 steps
sr_tensor = torch.tensor([sr_khz])

def net_call(xt, t, y):
    with torch.no_grad():
        return model(xt, t, y, sr_values=[sr_khz])

def drift(xt, t, y):
    g = net_call(xt, t, y)
    u = net_call(xt, t, None)
    return (1 - guidance_scale) * u + guidance_scale * g

# midpoint: x_{n+1} = x_n + h * f(x_n + h/2 * f(x_n))
x = x0.clone()
traj = [x0.clone()]
for i in range(len(ts) - 1):
    t = ts[i:i+1]
    h = ts[i+1] - ts[i]
    k1 = drift(x, t, Y_lr)
    x_mid = x + 0.5 * h * k1
    k2 = drift(x_mid, t + 0.5*h, Y_lr)
    x = x + h * k2
    traj.append(x.clone())

x1_spec = x
# concat LR + generated HF (handle overlap)
slice_start = max(0, lr_bin_count - hf_start_bin)  # max(0, 170-80)=90
x1_full = x1_spec[:, :, slice_start:, :]
full_spec = torch.cat([Y_lr, x1_full], dim=2)      # [1,2,511,T]
out_wav = postprocess(full_spec)
print(f"output waveform: {out_wav.shape}, rms={out_wav.pow(2).mean().sqrt().item():.4f}")

# --- save reference fixtures for Rust validation ---
# We save: Y_lr, x0, sr_khz, one-step (t=0.5) inputs/outputs, full output
t_probe = torch.tensor([0.5])
y_probe = Y_lr
x_probe = x0
with torch.no_grad():
    out_guided   = model(x_probe, t_probe, y_probe, sr_values=[sr_khz])
    out_unguided = model(x_probe, t_probe, None,   sr_values=[sr_khz])

np.savez(
    os.path.join(FIX, "universr_ref.npz"),
    # config
    n_fft=np.array(config["transform"]["n_fft"]),
    hop_length=np.array(config["transform"]["hop_length"]),
    alpha=np.array(config["transform"]["alpha"]),
    beta=np.array(config["transform"]["beta"]),
    comp_eps=np.array(config["transform"]["comp_eps"]),
    total_freq_bins=np.array(config["model"]["total_freq_bins"]),
    hr_freq_bins=np.array(config["model"]["hr_freq_bins"]),
    sr_khz=np.array(sr_khz),
    lr_bin_count=np.array(lr_bin_count),
    hf_start_bin=np.array(hf_start_bin),
    guidance_scale=np.array(guidance_scale),
    sigma_min=np.array(config["path"]["init_args"]["sigma_min"]),
    # the STFT window (hann, n_fft=1024)
    window=np.array(torch.signal.windows.hann(config["transform"]["n_fft"]).numpy()),
    # inputs for one-step probe
    x_probe=x_probe.numpy(),
    t_probe=np.array([0.5], dtype=np.float32),
    y_probe=y_probe.numpy(),
    sr_values=np.array([sr_khz], dtype=np.float32),
    out_guided=out_guided.numpy(),
    out_unguided=out_unguided.numpy(),
    # full pipeline
    Y_lr=Y_lr.numpy(),
    x0=x0.numpy(),
    full_spec_out=full_spec.numpy(),
    out_wav=out_wav.numpy(),
    lr_audio=lr.numpy(),
)
print(f"saved fixtures → {os.path.join(FIX,'universr_ref.npz')}")

# --- export one ONNX graph with guided + unconditioned outputs ---
class OnnxWrapper(nn.Module):
    def __init__(self, m, sr_khz_fixed: int):
        super().__init__()
        self.m = m
        self.sr_khz = sr_khz_fixed

    def forward(self, x, t, y):
        guided = self.m(x, t, y, sr_values=[self.sr_khz])
        unconditioned = self.m(x, t, None, sr_values=[self.sr_khz])
        return guided, unconditioned

wrapped = OnnxWrapper(model, sr_khz).eval()
onnx_path = os.path.join(MODELS, "universr_backbone.onnx")
torch.onnx.export(
    wrapped,
    (x_probe, t_probe, y_probe),
    onnx_path,
    input_names=["x", "t", "y"],
    output_names=["guided", "unconditioned"],
    opset_version=18,
    dynamo=True,
    external_data=False,
)
print(f"exported ONNX → {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

# --- validate ONNX vs PyTorch ---
import onnxruntime as ort
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
out_onnx_g, out_onnx_u = session.run(
    None,
    {"x": x_probe.numpy(), "t": t_probe.numpy(), "y": y_probe.numpy()},
)
err_g = np.abs(out_onnx_g - out_guided.numpy()).max()
err_u = np.abs(out_onnx_u - out_unguided.numpy()).max()
print(f"ONNX vs PyTorch  guided max|err| = {err_g:.2e}")
print(f"ONNX vs PyTorch  uncond max|err| = {err_u:.2e}")
assert err_g < 5e-4 and err_u < 5e-4, f"ONNX export mismatch too large: g={err_g} u={err_u}"

# --- save config.json for Rust ---
rust_cfg = {
    "n_fft": config["transform"]["n_fft"],
    "hop_length": config["transform"]["hop_length"],
    "window_fn": config["transform"]["window_fn"],
    "sampling_rate": config["transform"]["sampling_rate"],
    "alpha": config["transform"]["alpha"],
    "beta": config["transform"]["beta"],
    "comp_eps": config["transform"]["comp_eps"],
    "total_freq_bins": config["model"]["total_freq_bins"],
    "hr_freq_bins": config["model"]["hr_freq_bins"],
    "sr_to_lr_bins": {str(k): v for k, v in config["model"]["sr_to_lr_bins"].items()},
    "sigma_min": config["path"]["init_args"]["sigma_min"],
    "guidance_scale": guidance_scale,
    "ode_steps": 4,
    "ode_method": "midpoint",
    "min_samples": MIN_SAMPLES,
    "target_sr": 48000,
    "model_onnx": "universr_backbone.onnx",
}
with open(os.path.join(MODELS, "universr_config.json"), "w") as f:
    json.dump(rust_cfg, f, indent=2)
print(f"saved config → {os.path.join(MODELS,'universr_config.json')}")
print("\n✓ P4 export complete.")
