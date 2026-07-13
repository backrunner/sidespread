# UniverSR Pipeline Specification (for Rust P5 integration)

This document captures the exact UniverSR inference pipeline as derived from the Python source
(`external/universr/universr/`), so the Rust `ort` integration can reproduce it numerically.

## 1. Config (from `models/universr_config.json`, also `configs/config.yaml`)

| Param | Value | Source |
|---|---|---|
| n_fft | 1024 | transform.n_fft |
| hop_length | 512 | transform.hop_length |
| window | hann (torch.signal.windows.hann) | transform.window_fn |
| sampling_rate | 48000 | transform.sampling_rate |
| alpha (compression exponent) | 0.2 | transform.alpha |
| beta (scale factor) | 1 | transform.beta |
| comp_eps | 1e-4 | transform.comp_eps |
| center | True | ComplexSTFT hardcoded |
| onesided | True | ComplexSTFT hardcoded |
| total_freq_bins | 512 | model.total_freq_bins |
| hr_freq_bins | 432 | model.hr_freq_bins |
| sr_to_lr_bins | {8:80, 12:128, 16:170, 24:256} | model.sr_to_lr_bins |
| sigma_min | 1e-4 | path.init_args.sigma_min |
| guidance_scale | 1.5 (default) | inference arg |
| ode_steps | 4 (default) | inference arg |
| ode_method | midpoint | inference arg |
| min_samples | 32768 | inference hardcoded |
| target_sr | 48000 | TARGET_SR |

## 2. STFT Frontend (`AmplitudeCompressedComplexSTFT`)

Forward (waveform → compressed complex spectrum):
1. `X = torch.stft(x, n_fft=1024, hop=512, window=hann(1024), center=True, onesided=True, return_complex=True)`
   - Input `x`: shape `[B, C, T]` (or `[B, T]`). Squeeze channel, treat as `[B*T,]` → output `[B*T, F=513, T_frames]`.
   - `T_frames = 1 + T // hop` with center padding (reflect-equivalent).
2. Amplitude compression: `Xc = (|X + comp_eps|)^alpha * exp(1j * angle(X + comp_eps)) * beta`
   - alpha=0.2, beta=1, comp_eps=1e-4. Note: `X + comp_eps` is **complex** addition (eps added to complex).
3. View as real/imag: `real = view_as_real(Xc)` → `[B, F, T_frames, 2]`.
4. Permute to `[B, 2, F, T_frames]`.
5. Drop Nyquist bin: `[..., :-1, :]` → `[B, 2, F-1=512, T_frames]`.

So the model's input `Y` has **512 frequency bins** (n_fft/2 = 512, dropping the Nyquist bin at index 512).

Inverse (compressed complex spectrum → waveform):
1. Pad Nyquist bin back: `F.pad(spec, [0,0,0,1], value=0)` → `[B, 2, 513, T_frames]`.
2. Permute to `[B, F, T, 2]`, `view_as_complex` → `[B, F, T]` complex.
3. Invert compression: `X = |Xc / beta|^(1/alpha) * exp(1j * angle(Xc / beta))` (note: the invert does **not** re-add comp_eps).
4. `x = torch.istft(X, n_fft=1024, hop=512, window=hann, center=True, onesided=True, length=orig_len)`.

## 3. Frequency Bin Layout

- Total bins in Y: 512 (indices 0..511).
- For input SR `sr_khz` (8/12/16/24): `lr_bin_count = sr_to_lr_bins[sr_khz]` (80/128/170/256).
  - `Y_lr = Y[:, :, :lr_bin_count, :]` — the low-resolution condition (preserved from input).
- `hr_freq_bins = 432`; `hf_start_bin = total_freq_bins - hr_freq_bins = 512 - 432 = 80`.
  - `Y_hr = Y[:, :, hf_start_bin:, :]` = `Y[:, :, 80:, :]` — the high-res region (432 bins). The flow-matching **generates** this region.
- Note the **overlap**: bins 80..(lr_bin_count-1) appear in BOTH Y_lr and Y_hr. The final concat handles this:
  - `slice_start = max(0, lr_bin_count - hf_start_bin) = max(0, lr_bin_count - 80)`.
  - For 16k: `slice_start = max(0, 170-80) = 90`. The generated spec `x1_spec[:, :, 90:, :]` (342 bins) is concatenated after `Y_lr` (170 bins) → full 512 bins.

## 4. Initial Noise (`OriginalCFMPath.sample_source`)

`x0 = torch.randn_like(Y_hr_shape)` — Gaussian noise shaped like `Y_hr` (`[B, 2, 432, T_frames]`).
`sigma_min = 1e-4` (used only in `sample_xt` during training, not at inference).

## 5. ODE Solver (`TorchDiffeqSolver`, `method="midpoint"`)

The flow-matching ODE: `dx/dt = v(x, t, y)` where `v` is the model's predicted vector field.
Time discretization: `ts = linspace(0, 1, ode_steps+1)` → `[0, 0.25, 0.5, 0.75, 1.0]` for 4 steps.

The solver uses `torchdiffeq.odeint` with `method="midpoint"`. The midpoint method:
```
For each step i (t_i → t_{i+1}), h = t_{i+1} - t_i:
  k1 = f(x, t_i)
  x_mid = x + 0.5 * h * k1
  k2 = f(x_mid, t_i + 0.5*h)
  x = x + h * k2
```

In Rust, we unroll this loop ourselves (don't bake it into ONNX). Each `f(...)` call = one ONNX forward (guided + uncond for CFG).

## 6. CFG (Classifier-Free Guidance)

`drift(xt, t, y) = (1 - guidance_scale) * model_unguided(xt, t) + guidance_scale * model_guided(xt, t, y)`

- A single ONNX graph, `universr_backbone.onnx`, takes `(x, t, y)` and returns both
  `guided` and `unconditioned` outputs. The backbone weights are stored once.
- ONNX Runtime memory-pattern caching is disabled and CPU intra-op threads are bounded to keep
  peak memory within the project budget.
- `guidance_scale` default 1.5.

## 7. Model Forward Signature (PyTorch)

`ConvNeXtUNetCond.forward(x, t, y, sr_values)`:
- `x`: `[B, 2, 432, T]` (noisy HF spec, x_t).
- `t`: `[B]` or `[B,1]` scalar time in [0,1] (flow-matching time, NOT a diffusion timestep).
- `y`: `[B, 2, lr_bin_count, T]` (LR condition) or `None` (uncond).
- `sr_values`: list of int, e.g. `[16]`. Used for: (a) `lr_bin_count` lookup, (b) sr embedding.

The ONNX graphs have `sr_values` baked (one graph per sr_khz). We export **per sr_khz**: currently only `16` (16kHz effective bandwidth). For other rates, re-export with `sr_khz_fixed=8/12/24`.

**Time embedding**: `SinusoidalTimeEmbedding(dim=256, mode='learnable', time_scale=100)`.
`freqs = t * weights * 2π`, `embed = [sin, cos] * sqrt(2)`. The `t` input is a float tensor [B,1] in [0,1].

## 8. Full Inference Pipeline (Rust P5 to reproduce)

```
1. Load audio (mono side channel), resample to 48k if needed.
2. Pad to min 32768 samples.
3. STFT: n_fft=1024, hop=512, hann, center=True, onesided → complex [F=513, T_frames].
4. Amplitude compress: |X+1e-4|^0.2 * exp(1j*angle(X+1e-4)) * 1.
5. To real/imag channels: [2, 512, T_frames] (drop Nyquist).
6. Slice: Y_lr = [:lr_bin_count], Y_hr_shape = [80:] (for noise shape).
7. x0 = randn(Y_hr_shape)  [B,2,432,T].
8. ts = linspace(0,1,5).
9. For i in 0..4:
     h = ts[i+1]-ts[i]
     (k1_g, k1_u) = ort(x, ts[i], Y_lr)
     k1 = (1-gs)*k1_u + gs*k1_g
     x_mid = x + 0.5*h*k1
     (k2_g, k2_u) = ort(x_mid, ts[i]+0.5*h, Y_lr)
     k2 = (1-gs)*k2_u + gs*k2_g
     x = x + h*k2
10. slice_start = max(0, lr_bin_count - 80)
    full_spec = cat([Y_lr, x[:, :, slice_start:, :]], dim=freq) → [2, 512, T].
11. Pad Nyquist: [2, 513, T].
12. Invert compress: |Xc|^5 * exp(1j*angle(Xc))  (no eps).
13. iSTFT: n_fft=1024, hop=512, hann, center=True, onesided → waveform [T'].
14. Trim to original length. Resample back if input wasn't 48k.
```

## 9. Band-Merge (sidespread-specific, NOT in UniverSR)

UniverSR generates the full side channel. For sidespread's **targeted HF repair**, we:
- Keep the original side's bins below `fc` (our detection cutoff, e.g. 8kHz).
- Take only UniverSR's HF bins above `fc`.
- Crossfade in a transition band.

This is a post-step on `full_spec` before iSTFT, using `repair::common::band_mask`.

## 10. Numerical Validation

Fixtures in `tests/fixtures/universr_ref.npz`:
- `x_probe`, `t_probe`, `y_probe`, `sr_values` — one-step inputs.
- `out_guided`, `out_unguided` — PyTorch reference outputs for that one step.
- `x0`, `Y_lr`, `full_spec_out`, `out_wav`, `lr_audio` — full pipeline reference.
- `window` — the hann window (1024).

Rust P5 milestones:
1. STFT frontend: Rust `realfft` vs Python `torch.stft` — RMSE < 3.5e-4 and max abs
   diff < 5e-3. The max bound covers near-cancelled FFT bins whose phase is amplified by
   the 0.2 compression exponent; end-to-end waveform error remains the acceptance gate.
2. Amplitude compress: Rust vs Python — max abs diff < 1e-5.
3. Single ONNX step: Rust `ort` vs `out_guided`/`out_unguided` fixtures — max abs diff < 5e-4.
4. Full ODE: Rust 4-step midpoint vs `full_spec_out` — SNR > 40 dB.
5. End-to-end: Rust output vs `out_wav` — SNR > 35 dB and max abs diff < 1e-3.
