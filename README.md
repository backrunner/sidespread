# Sidespread

Sidespread repairs missing high-frequency detail in the side channel of stereo audio. It first
checks which parts of a track are affected, then uses either a fast DSP repair or an optional
UniverSR neural model. Healthy audio is left alone.

## Get Started

Build the CLI with Rust:

```bash
cargo build --release
```

Run the DSP route, which does not need a model:

```bash
./target/release/sidespread process song.wav --mode dsp
```

This writes `song.repaired.wav` and `report.json`. If the track does not need repair, Sidespread
writes only the report.

Release archives contain the CLI and model configuration. You can also download an archive from
GitHub Releases and place the `sidespread` binary on your `PATH`.

## Commands

```bash
# Inspect a file without changing it
sidespread detect song.wav

# Detect and choose DSP, neural, or hybrid repair per segment
sidespread process song.wav

# Force the fast, model-free DSP route
sidespread process song.wav --mode dsp

# Create a synthetic defect and compare the repair with clean ground truth
sidespread eval clean.wav --mode dsp

# Print WAV metadata and a quick M/S frequency summary
sidespread info song.wav
```

Use `sidespread <command> --help` for all options. Sidespread accepts stereo WAV files at 44.1 or
48 kHz in 16/24/32-bit PCM or 32-bit float format.

## Neural Repair

The neural route uses the MIT-licensed UniverSR model. The 221 MB ONNX file is not stored in Git or
bundled with release archives. Build it locally with:

```bash
python3 scripts/build_universr_model.py
```

The script creates an isolated Python environment, downloads the upstream weights, and writes
`models/universr_backbone.onnx`. You can then use automatic routing or force neural repair:

```bash
sidespread process song.wav --mode nn
```

The bundled neural condition is designed for the default cutoff near 8 kHz. Custom cutoff values
remain available for detection and DSP repair.

## Development

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --locked
```

## License

Sidespread is licensed under [Apache-2.0](LICENSE). UniverSR remains under its upstream MIT license;
its notice is included in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/UniverSR-MIT.txt).
