# Sidespread Benchmark Baseline

Recorded 2026-07-14 on macOS arm64 using the release build and CPUExecutionProvider.

## DSP Route

- Input: 10 seconds, 48 kHz stereo WAV.
- Command: `sidespread process ... --mode dsp`.
- Original warm wall time: 0.26-0.27 seconds.
- Parallel routing + cached FFT warm wall time: 0.19-0.20 seconds.
- Historical replacement-route RTF: 0.019-0.020.
- The optimized WAV and JSON report are byte-identical to the serial baseline.

### Additive v0.2.0 route

- Input: 29.989 seconds, 44.1 kHz, 24-bit stereo FMA music with synthetic Side-HF loss.
- Command: `sidespread process degraded.wav` with the public defaults.
- Warm wall time: 0.81 seconds; user CPU time: 1.02 seconds.
- Current RTF: 0.027 (target: < 0.1).
- Peak resident memory: 201,424,896 bytes.
- All 749 detected segments were processed; a second pass detected zero deficient segments and did
  not write another WAV.

## UniverSR Route

- Validation input: one fixed 32768-sample (0.683-second) model context at 48 kHz.
- ODE: four midpoint steps, CFG 1.5.
- Current full 32768-sample ODE/iSTFT validation: 26.37 seconds on an otherwise idle host.
- Current RTF for the fixed 0.683-second context: 38.6.
- Maximum resident set size: 1,836,433,408 bytes.
- Historical peak memory footprint: 1,962,264,592 bytes (target: < 2 GB).
- ONNX intra-op inference remains at one thread because the measured memory headroom is too small
  to increase parallelism without risking the 2 GB limit.

## Numerical Validation

- Single ONNX step max absolute error: guided <= 1.62e-4, unconditioned <= 7.43e-5.
- Full four-step ODE spectrum SNR: > 40 dB.
- End-to-end iSTFT waveform SNR: > 35 dB.
- Rust/PyTorch frontend comparison: max absolute error < 5e-3 and RMSE < 3.5e-4;
  the largest differences occur in near-cancelled FFT bins after amplitude compression.
