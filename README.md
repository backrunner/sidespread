# Sidespread

Sidespread fills in missing high-frequency detail in the side channel of stereo music. It is aimed
at AI-generated tracks whose top end sounds narrow because the left/right difference loses energy
above roughly 8 kHz.

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
```

This writes `song.repaired.wav` and `report.json`. If the track already has enough side-channel high
frequency content, Sidespread leaves it alone and writes only the report.

The default workflow is intentionally automatic. It detects deficient sections, estimates their
natural side level from nearby intact frequencies, and adds only the missing high-frequency energy.
Existing side-channel complex spectrum is kept in place; phase-diffused content is layered on top
without cancellation. The result is kept below clipping, and there are no repair modes to choose.

## Commands

```bash
# Repair a track
sidespread process song.wav

# Analyze without writing audio
sidespread detect song.wav

# Show WAV metadata and a quick M/S frequency summary
sidespread info song.wav
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
