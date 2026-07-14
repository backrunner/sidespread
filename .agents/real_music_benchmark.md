# Real-Music Side-HF Recovery Benchmark

Recorded 2026-07-15 on macOS arm64 with Sidespread CPU inference.

## Sources

All sources are real stereo music from the librosa example-data repository. The HQ Ogg files were
decoded to 44.1 kHz, 24-bit PCM WAV for the benchmark. Decoding does not add information that was
not present in the source.

| Track | Style | License | Full duration | Neural excerpt |
|---|---|---|---:|---:|
| Kevin MacLeod - Vibe Ace | Jazz combo | CC BY 3.0 | 61.456 s | 20-22 s |
| Admiral Bob - Choice | Drum and bass | CC BY-NC 4.0 | 25.026 s | 10-12 s |
| Karissa Hobbs - Let's Go Fishin' | Folk/pop with vocals | CC BY 3.0 | 132.988 s | 60-62 s |

Source repository: <https://github.com/librosa/data/tree/main/audio>

## Method

`sidespread eval` treats each original side channel as clean ground truth, removes side content
above the configured cutoff, repairs the degraded signal, and compares the written WAV with the
original side.

The benchmark adds `snr_hf_db`, calculated directly over STFT bins from the cutoff to Nyquist:

```text
10 * log10(sum(|S_reference|^2) / sum(|S_reference - S_candidate|^2))
```

A completely removed high band is expected to start near 0 dB: its error energy is approximately
equal to the original signal energy. Positive movement indicates recovery of original information;
negative movement means the repair added more error than leaving the band missing.

## 16 kHz Cutoff

The full tracks were processed with `eval --fc 16000 --mode dsp`. Neural repair cannot be used at
this cutoff because the bundled UniverSR graph only supports the condition near the default 8 kHz
cutoff.

| Track | Clean HF corr | LSD-HF before -> after | HF-SNR before -> after | HF-SNR delta |
|---|---:|---:|---:|---:|
| Vibe Ace | -0.189 | 28.949 -> 8.174 dB | 0.268 -> -2.794 dB | -3.062 dB |
| Choice | 0.815 | 41.835 -> 12.483 dB | 0.489 -> 4.682 dB | +4.194 dB |
| Let's Go Fishin' | -0.284 | 18.365 -> 14.720 dB | 0.733 -> -4.670 dB | -5.403 dB |

Mean LSD-HF reduction was 17.924 dB, but mean HF-SNR change was -1.424 dB. DSP recovered useful
information only for the track whose original mid and side high bands were strongly correlated.

## 8 kHz Cutoff

Two-second excerpts were processed with forced DSP and forced neural routes. Four ODE midpoint
steps were used for neural inference.

| Track | Clean HF corr | Route | LSD-HF before -> after | HF-SNR before -> after | HF-SNR delta |
|---|---:|---|---:|---:|---:|
| Vibe Ace | -0.091 | DSP | 41.428 -> 12.186 dB | 0.062 -> -3.996 dB | -4.058 dB |
| Vibe Ace | -0.091 | Neural | 41.428 -> 20.568 dB | 0.062 -> -0.659 dB | -0.721 dB |
| Choice | 0.929 | DSP | 51.523 -> 11.473 dB | 0.064 -> 6.705 dB | +6.641 dB |
| Choice | 0.929 | Neural | 51.523 -> 15.447 dB | 0.064 -> -5.871 dB | -5.935 dB |
| Let's Go Fishin' | -0.562 | DSP | 58.285 -> 17.355 dB | 0.075 -> -7.957 dB | -8.032 dB |
| Let's Go Fishin' | -0.562 | Neural | 58.285 -> 33.708 dB | 0.075 -> -4.362 dB | -4.437 dB |

DSP mean HF-SNR change was -1.816 dB. Neural mean HF-SNR change was -3.698 dB. Every route reduced
LSD-HF, but only DSP on the strongly correlated drum-and-bass excerpt recovered original high-band
information.

## Routing Finding

After synthetic degradation, `detect` routed all 49 segments of every excerpt to neural repair.
This included Choice, where DSP improved HF-SNR by 6.641 dB and neural reduced it by 5.935 dB.

The current router calculates correlation in the deficient high band. Once side HF has been removed,
that correlation collapses toward zero and no longer represents the clean mid/side relationship.
Routing should instead use an intact band below the cutoff, or another feature that survives the
defect.

## Conclusion

The current implementation can generate plausible high-frequency spectral shape, as shown by the
consistent LSD-HF reductions, but it does not reliably reconstruct the original high-frequency
information. HF-SNR must remain a release-gating metric. Before claiming accurate restoration:

1. Base route selection on intact-band evidence rather than correlation in the missing band.
2. Validate neural repair on a larger real-music set and reject repairs that cannot beat the degraded
   HF-SNR baseline.
3. Treat 16 kHz cutoff repair as DSP-only unless a matching neural condition is exported.

This is a diagnostic benchmark, not a statistically complete listening test. It covers three tracks,
and neural measurements use two-second excerpts because CPU inference is roughly 40x slower than
real time.

## Conservative Auto Follow-up

The router was changed to use correlation from the intact band below the cutoff, smoothed across
nine overlapping segments (about 400 ms). The calibrated automatic threshold is 0.35. Automatic
mode now applies DSP only above that confidence and leaves all other deficient segments unchanged;
neural repair remains available through explicit `--mode nn`.

| Cutoff | Track | HF-SNR before -> after | Result |
|---|---|---:|---|
| 8 kHz | Vibe Ace excerpt | 0.062 -> 0.062 dB | unchanged |
| 8 kHz | Choice excerpt | 0.064 -> 2.590 dB | improved |
| 8 kHz | Let's Go Fishin' excerpt | 0.075 -> 0.075 dB | unchanged |
| 16 kHz | Vibe Ace full track | 0.268 -> 0.268 dB | unchanged |
| 16 kHz | Choice full track | 0.489 -> 4.720 dB | improved |
| 16 kHz | Let's Go Fishin' full track | 0.733 -> 0.733 dB | unchanged |

On this diagnostic set, conservative auto eliminated every observed HF-SNR regression while
retaining useful DSP recovery on the correlated track. This is a safety improvement, not proof that
the three-track set covers all music; a larger corpus remains necessary before raising coverage.

## FMA-small Stratified Follow-up

The larger follow-up uses FMA-small, a common music-information-retrieval dataset with 8000
30-second tracks across eight balanced top-level genres. Each audio file retains its artist-selected
Creative Commons license; no dataset audio is stored in this repository.

The benchmark selected 25 valid stereo tracks per genre with seed `20260715`. Mono or invalid files
were replaced from the same genre. A result was eligible only when the original side channel had at
least `1e-4` of its STFT energy above the cutoff. This left 171 eligible tracks at 8 kHz and 94 at
16 kHz; the lower 16 kHz count reflects the high-frequency limits of some source MP3 files.

Automatic routing now requires all of the following, smoothed over nine overlapping segments:

1. Complex M/S correlation of at least `0.35` in `[0.75 * fc, fc - 500 Hz]`.
2. Complex M/S correlation of at least `0.40` in `[fc + 250 Hz, fc + 500 Hz]`.
3. An S/M energy ratio of at least `1e-3` in that outer transition band.

The transition energy condition prevents normalized correlation from looking confident when the
remaining side signal is effectively numerical residue. A hard cutoff with no measurable evidence
is intentionally skipped in automatic mode; forced DSP remains available explicitly.

| Cutoff | Eligible | Repaired tracks | HF-SNR improved | HF-SNR regressed | Mean HF-SNR delta | Minimum delta | Mean LSD-HF delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 kHz | 171 | 60 | 21 | 0 | +0.102 dB | -0.034 dB | -0.908 dB |
| 16 kHz | 94 | 38 | 15 | 0 | +0.058 dB | -0.042 dB | -0.544 dB |

An improvement or regression means a change beyond `+/-0.05 dB` HF-SNR. Repair covered about 2.1%
of deficient 8 kHz segments and 2.0% of deficient 16 kHz segments. The policy is deliberately
conservative: it keeps only segments with observable evidence that mid high frequencies remain a
useful predictor of the missing side information.

The reproducible macOS benchmark command is:

```bash
python3 scripts/benchmark_fma.py \
  --dataset-root /path/to/fma \
  --per-genre 25 \
  --thresholds 0.35 \
  --transition-threshold 0.40
```
