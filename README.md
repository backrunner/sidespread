# Sidespread

Sidespread fills in missing upper-mid and high-frequency detail in stereo music. It is aimed at
AI-generated tracks whose left/right difference has one or more local frequency-band defects from
roughly 5 kHz upward, as well as tracks with a shared brick-wall cutoff across the complete stereo
signal.

The missing signal cannot be recovered exactly. Sidespread instead creates a diffuse high band from
the intact audio around it, while leaving healthy sections and frequencies below the transition
untouched.

## Get Started

Download a binary from [GitHub Releases](https://github.com/backrunner/sidespread/releases), or build
it with Rust:

```bash
cargo build --release
```

Process a stereo WAV:

```bash
./target/release/sidespread process song.wav
# The command name is optional:
./target/release/sidespread song.wav
```

This writes `song.repaired.wav`. A JSON report is optional:

```bash
./target/release/sidespread process song.wav --output-report report.json
```

If the track already has enough high-frequency content, Sidespread leaves it alone and writes no
WAV. It still prints the analysis in the terminal and only writes JSON when `--output-report` is
specified. The former `--report` spelling remains available as a compatibility alias.

The terminal UI uses color, stage markers, a compact segment table, and one live row for each active
stage. A row is updated in place at most twice per second, so long processing runs do not print one
line per frame or percentage point. The default workflow reports analysis, DSP repair, optional
UniverSR inference and guarding, shared-bandwidth extension, harmonic debleed, high-frequency
smearing repair, phase stabilization, each fidelity gate, headroom, and output verification.
Progress includes elapsed time and an estimate of the remaining time. Its layout adapts to the
terminal width and reserves the final column, preventing wrapped refreshes from leaving stale rows.
Pipes and CI receive one plain completion line per stage without cursor control; set
`FORCE_COLOR=1` to opt into color explicitly, or `NO_COLOR=1` to disable color.

Sidespread does not overwrite an existing WAV or report silently. Interactive runs ask for
confirmation with a default of `N`; non-interactive runs stop with an error. Pass `--force` when an
automated workflow is intentionally replacing its outputs.

If a neural repair route is selected and the default UniverSR model is missing, Sidespread asks
whether it should download the model (about 232 MB). Use `sidespread model download` to install it
ahead of time, or answer `n` to keep the current run offline.

## Repair Modes

`process` defaults to `--mode dsp`, so UniverSR is not loaded unless it is explicitly selected:

| Mode | Missing-band processing | Relative cost |
| --- | --- | --- |
| `dsp` | Deterministic M/S spectral compensation on detected defects | Fastest; default |
| `hybrid` | Continuous DSP repair plus guarded UniverSR texture only on deep local dropouts | Selective neural cost |
| `nn` | Guarded UniverSR on every detected repair span | Slowest |

```bash
# Default: DSP only
sidespread process song.wav --mode dsp

# Recommended neural-assisted mode
sidespread process song.wav --mode hybrid

# UniverSR for every detected missing-band interval
sidespread process song.wav --mode nn
```

The neural defaults are calibrated conservatively:

- `--ode-steps 2` is the default quality/speed balance. Use `1` for a faster coarse pass or `4` for
  the slower reference-quality integration.
- `--hybrid-neural-mix 0.30` adds 30% of the guarded neural delta on top of the complete DSP result;
  it does not turn the DSP result down.
- `--hybrid-neural-depth 0.35` invokes UniverSR only when the observed local high-frequency level is
  at or below 35% of the inferred target. Raising it repairs more intervals and costs more time.
- `--neural-max-hf-boost-db 0` prevents neural STFT frames from exceeding the local high-frequency
  target inferred from Mid and the intact Side/Mid band below the cutoff. Raise it explicitly only
  when a brighter neural fill is wanted.
- `--model-path /path/to/universr_backbone.onnx` selects a non-default model location.

The neural loudness guard preserves the original Side waveform and adds only a band-limited delta.
It removes any delta component that would cancel existing high-frequency content, smooths frame
gains to avoid pumping, and reduces or bypasses generated detail that exceeds the local energy
ceiling. Hybrid mode first repairs the complete defect with DSP and then overlays this guarded
delta only inside independently selected, content-aware neural boundaries. UniverSR uses 250 ms
overlap between one-second inference chunks; this avoids redundant inference while retaining a
long crossfade around model chunk boundaries.

The three broader artifact processors described below remain DSP-based and enabled in all three
`process` modes. Shared full-signal brick-wall extension is also independent of the selected mode.

The default workflow is intentionally automatic. It detects deficient sections, estimates their
natural side level from nearby intact frequencies, and adds only the missing high-frequency energy.
The target follows each STFT frame, so a deeper local dropout receives more compensation than a
partly intact one. Existing side-channel complex spectrum is kept in place; phase-diffused content
is layered on top without cancellation.

By default, `--scan-start-hz 5000` scans from 5 kHz to Nyquist in independent 500 Hz analysis
bands. Several non-adjacent defects can be selected in the same time span, and DSP changes only
the selected time-frequency regions. `--fc 8000` remains the historical high-frequency metric and
UniverSR conditioning cutoff; it does not force DSP repair to start at 8 kHz. UniverSR generates
only its supported region from about 8 kHz upward, so in `hybrid` mode DSP supplies any selected
5-8 kHz compensation. Change `--scan-start-hz` when material needs a different lower scan edge.

The terminal and optional JSON reports separate defect detection from execution. The
`missing_band_processing` summary includes detected and routed segment counts, the scan/band
settings, the number of multi-band defects, and the lowest defect frequency. A segment `route` of
`skip` applies only to dynamic Side missing-band fill; full-track smearing, harmonic debleed, phase,
and shared-bandwidth stages are reported independently. Long terminal reports sample actual routed
segments from across the track instead of showing only the first and last windows.

Overlapping analysis windows are used only to locate defects; they are not used as fixed repair
cuts. Consecutive detections are merged into content-aware repair spans whose boundaries move
outward to locally quiet, low-transient, phase-stable points. The repair delta then fades in and out
over a boundary-specific 15–60 ms envelope, keeping the original waveform exactly at each boundary.

Sidespread also checks for a dense, shared brick-wall cutoff in both Mid and Side. When one is
detected, it extends both channels from the intact spectrum below the edge. This is perceptual
bandwidth extension, not lossless recovery: information removed by generation or encoding cannot
be reconstructed exactly. The synthesized M/S deltas share the same clipping protection, and there
is no backend mode to choose for missing-band repair.

Three broader AI-audio artifact processors run by default in the `process` command. They use a
multi-resolution real-time DSP chain rather than the much slower research neural backend:

- Smearing repair enhances only high-frequency transients isolated by harmonic/percussive
  source-separation soft masks. Its calibrated default strength is `0.35`.
- Harmonic debleed uses a peak-protected Wiener mask to reduce low-level residue between stable
  harmonics. Its calibrated default strength is `0.65`.
- Phase stabilization smooths unstable Mid/Side relative phase and gently reduces only incoherent
  high-frequency Side energy. Its calibrated default strength is `0.50`.

Override each strength independently in `[0, 1]`; `0` is an exact bypass:

```bash
sidespread process song.wav \
  --smearing-strength 0.45 \
  --bleeding-strength 0.55 \
  --phase-strength 0.60
```

Disable a stage for intentionally diffuse or noisy material with `--no-repair-smearing`,
`--no-repair-bleeding`, or `--no-stabilize-phase`. `detect` and the hidden evaluation command do
not enable these processors. They are established low-latency restoration techniques, not a claim
that missing source information is recovered.

Every artifact stage is followed by a fidelity gate. It requires at least 50 dB protected-band
SNR, verifies that the stage-specific defect metric actually improves, limits total high-frequency
energy movement, protects existing harmonic peaks within 0.5 dB, and prevents phase repair from
removing more than 3 dB of Side energy. A failing stage is retried at 75%, 50%, and 25% mix; if none
passes, that stage is bypassed. The terminal and optional JSON report show requested versus applied
strength. When every requested artifact stage is rejected and no missing-band repair is needed, no
WAV is written.

## Commands

```bash
# Repair a track
sidespread process song.wav

# Analyze without writing audio or JSON
sidespread detect song.wav

# Persist the detailed analysis when needed
sidespread detect song.wav --output-report report.json

# Show WAV metadata and a quick M/S frequency summary
sidespread info song.wav

# Download or verify the optional neural model
sidespread model download
sidespread model verify
```

Use `sidespread <command> --help` for output paths and advanced cutoff settings. Input must be a
stereo 44.1 or 48 kHz WAV in 16/24/32-bit PCM or 32-bit float format.

## What To Expect

On a 40-track stratified FMA-small evaluation at 8 kHz, Sidespread detected 88.1% of synthetically
degraded sections and repaired every detected section. The clean control set flagged 3.6% of
sections. The repaired side-energy ratio was within a median 3.06 dB of the clean reference, and the
median L/R high-frequency correlation error was 0.076. Frequencies below the 500 Hz transition
margin remained at or above 34.3 dB level-matched SNR. Projection of the repaired high band onto
the Side information already present in the degraded input never fell below +0.93 dB, so the added
layer did not attenuate existing HF content in this set.

Ground-truth high-frequency SNR fell by 2.57 dB on average. This is expected when synthesizing
information that is absent from the input: the output can recover plausible width and spectral
balance, but it should not be described as the original lost waveform. The benchmark is an
engineering validation, not a substitute for level-matched listening tests.

## Development

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --locked
```

The FMA-small benchmark is resumable and keeps all dataset audio outside the repository:

```bash
python3 scripts/benchmark_fma.py --dataset-root /path/to/fma
```

Pushing a tag such as `v0.2.0` builds Linux, macOS, and Windows archives, generates checksums, and
publishes a GitHub Release.

## License

Sidespread is licensed under [Apache-2.0](LICENSE). The optional UniverSR research backend remains
under its upstream MIT license; its notice is included in
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/UniverSR-MIT.txt).
