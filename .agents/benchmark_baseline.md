# Sidespread Benchmark Baseline

Recorded 2026-07-14 on macOS arm64 using the release build and CPUExecutionProvider.

## DSP Route

- Input: 10 seconds, 48 kHz stereo WAV.
- Command: `sidespread process ... --mode dsp`.
- Wall time: 0.23 seconds.
- CPU time: 0.20 seconds.
- RTF: 0.023 (target: < 0.1).

## UniverSR Route

- Input: 1 second, 48 kHz stereo WAV; fixed 32768-sample model chunks with 50% overlap.
- ODE: four midpoint steps, CFG 1.5.
- Wall time: 47.66 seconds on CPU while the host was under concurrent build load.
- Maximum resident set size: 1,843,462,144 bytes.
- Peak memory footprint: 1,962,264,592 bytes (target: < 2 GB).

## Numerical Validation

- Single ONNX step max absolute error: guided <= 1.62e-4, unconditioned <= 7.43e-5.
- Full four-step ODE spectrum SNR: > 40 dB.
- End-to-end iSTFT waveform SNR: > 35 dB.
- Rust/PyTorch frontend comparison: max absolute error < 5e-3 and RMSE < 3.5e-4;
  the largest differences occur in near-cancelled FFT bins after amplitude compression.
