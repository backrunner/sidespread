//! Signal-domain quality gate for artifact processing.

use crate::analysis::spectrum::bin_of;
use crate::analysis::stft::{stft, SpectrumFrame, StftConfig};
use crate::config::Config;
use crate::terminal::TaskProgress;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub enum ArtifactKind {
    Smearing,
    HarmonicBleeding,
    PhaseIncoherence,
}

pub struct GuardedAudio {
    pub mid: Vec<f32>,
    pub side: Vec<f32>,
    pub retained_mix: f32,
}

pub struct AudioPair<'a> {
    pub mid: &'a [f32],
    pub side: &'a [f32],
}

#[cfg(test)]
pub fn guard(
    original: AudioPair<'_>,
    candidate: AudioPair<'_>,
    kind: ArtifactKind,
    requested_strength: f32,
    config: &Config,
    sample_rate: u32,
) -> GuardedAudio {
    guard_with_progress(
        original,
        candidate,
        kind,
        requested_strength,
        config,
        sample_rate,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn guard_with_progress(
    original: AudioPair<'_>,
    candidate: AudioPair<'_>,
    kind: ArtifactKind,
    requested_strength: f32,
    config: &Config,
    sample_rate: u32,
    show_progress: bool,
) -> GuardedAudio {
    if candidate.mid.len() != original.mid.len()
        || candidate.side.len() != original.side.len()
        || !candidate
            .mid
            .iter()
            .chain(candidate.side)
            .all(|sample| sample.is_finite())
    {
        return bypass(original.mid, original.side);
    }

    let progress = TaskProgress::new_if(kind.gate_label(), 5, "checks", show_progress);
    let baseline = Diagnostics::new(original.mid, original.side, config, sample_rate, kind);
    progress.advance(1);
    for mix in [1.0f32, 0.75, 0.5, 0.25] {
        let mid = blend(original.mid, candidate.mid, mix);
        let side = blend(original.side, candidate.side, mix);
        let candidate = Diagnostics::new(&mid, &side, config, sample_rate, kind);
        let accepted = passes(
            &baseline,
            &candidate,
            original.mid,
            original.side,
            &mid,
            &side,
            kind,
            requested_strength * mix,
            config,
            sample_rate,
        );
        progress.advance(1);
        if accepted {
            progress.finish();
            return GuardedAudio {
                mid,
                side,
                retained_mix: mix,
            };
        }
    }
    progress.finish();
    bypass(original.mid, original.side)
}

impl ArtifactKind {
    fn gate_label(self) -> &'static str {
        match self {
            Self::Smearing => "SMEARING SAFETY",
            Self::HarmonicBleeding => "DEBLEED SAFETY",
            Self::PhaseIncoherence => "PHASE SAFETY",
        }
    }
}

struct Diagnostics {
    high_energy: f64,
    side_high_energy: f64,
    spectral_flux: f64,
    inter_peak_ratio: f64,
    phase_jitter: f64,
    interchannel_correlation: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct CalibrationDiagnostics {
    pub high_energy: f64,
    pub side_high_energy: f64,
    pub spectral_flux: f64,
    pub inter_peak_ratio: f64,
    pub phase_jitter: f64,
    pub interchannel_correlation: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct CalibrationFidelity {
    pub protected_snr_db: f32,
    pub peak_retention_db: f32,
}

#[cfg(test)]
pub(crate) fn calibration_diagnostics(
    mid: &[f32],
    side: &[f32],
    config: &Config,
    sample_rate: u32,
    kind: ArtifactKind,
) -> CalibrationDiagnostics {
    let diagnostics = Diagnostics::new(mid, side, config, sample_rate, kind);
    CalibrationDiagnostics {
        high_energy: diagnostics.high_energy,
        side_high_energy: diagnostics.side_high_energy,
        spectral_flux: diagnostics.spectral_flux,
        inter_peak_ratio: diagnostics.inter_peak_ratio,
        phase_jitter: diagnostics.phase_jitter,
        interchannel_correlation: diagnostics.interchannel_correlation,
    }
}

#[cfg(test)]
pub(crate) fn calibration_fidelity(
    original: AudioPair<'_>,
    candidate: AudioPair<'_>,
    config: &Config,
    sample_rate: u32,
    kind: ArtifactKind,
) -> CalibrationFidelity {
    CalibrationFidelity {
        protected_snr_db: protected_band_snr(
            original.mid,
            original.side,
            candidate.mid,
            candidate.side,
            protected_end_hz(config, sample_rate, kind),
            config,
            sample_rate,
        ),
        peak_retention_db: peak_retention_db(
            original.mid,
            original.side,
            candidate.mid,
            candidate.side,
            config,
            sample_rate,
        ),
    }
}

impl Diagnostics {
    fn new(
        mid: &[f32],
        side: &[f32],
        config: &Config,
        sample_rate: u32,
        kind: ArtifactKind,
    ) -> Self {
        let settings = diagnostic_stft(config, kind);
        let mid = stft(mid, &settings);
        let side = stft(side, &settings);
        let start = start_bin(config, sample_rate, settings.n_fft, kind);
        Self {
            high_energy: band_energy_pair(&mid, &side, start),
            side_high_energy: band_energy(&side, start),
            spectral_flux: spectral_flux(&mid, &side, start),
            inter_peak_ratio: inter_peak_ratio(&mid, &side, start, settings.n_fft, sample_rate),
            phase_jitter: relative_phase_jitter(&mid, &side, start),
            interchannel_correlation: interchannel_correlation(&mid, &side, start),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn passes(
    baseline: &Diagnostics,
    candidate: &Diagnostics,
    original_mid: &[f32],
    original_side: &[f32],
    candidate_mid: &[f32],
    candidate_side: &[f32],
    kind: ArtifactKind,
    effective_strength: f32,
    config: &Config,
    sample_rate: u32,
) -> bool {
    if effective_strength <= 0.0 || baseline.high_energy <= 1e-12 {
        return false;
    }
    let protected_end = protected_end_hz(config, sample_rate, kind);
    if protected_band_snr(
        original_mid,
        original_side,
        candidate_mid,
        candidate_side,
        protected_end,
        config,
        sample_rate,
    ) < 50.0
    {
        return false;
    }

    let energy_ratio = candidate.high_energy / baseline.high_energy;
    let required_change = 0.002 * effective_strength as f64;
    match kind {
        ArtifactKind::Smearing => {
            (0.99..=2.0).contains(&energy_ratio)
                && baseline.spectral_flux > 1e-8
                && candidate.spectral_flux >= baseline.spectral_flux * (1.0 + required_change)
        }
        ArtifactKind::HarmonicBleeding => {
            let peak_retention = peak_retention_db(
                original_mid,
                original_side,
                candidate_mid,
                candidate_side,
                config,
                sample_rate,
            );
            (0.5..=1.05).contains(&energy_ratio)
                && baseline.inter_peak_ratio > 1e-10
                && candidate.inter_peak_ratio <= baseline.inter_peak_ratio * (1.0 - required_change)
                && peak_retention >= -0.5
        }
        ArtifactKind::PhaseIncoherence => {
            let side_ratio = if baseline.side_high_energy > 1e-12 {
                candidate.side_high_energy / baseline.side_high_energy
            } else {
                1.0
            };
            (0.5..=1.05).contains(&energy_ratio)
                && side_ratio >= 10.0f64.powf(-3.0 / 10.0)
                && baseline.phase_jitter > 1e-5
                && candidate.phase_jitter <= baseline.phase_jitter * (1.0 - required_change)
                && candidate.interchannel_correlation >= baseline.interchannel_correlation - 1e-4
        }
    }
}

fn diagnostic_stft(config: &Config, kind: ArtifactKind) -> StftConfig {
    match kind {
        ArtifactKind::Smearing => {
            let n_fft = config.n_fft.clamp(4, 2048);
            StftConfig::new(n_fft, (n_fft / 4).max(1))
        }
        ArtifactKind::HarmonicBleeding | ArtifactKind::PhaseIncoherence => {
            StftConfig::new(config.n_fft, config.hop)
        }
    }
}

fn start_bin(config: &Config, sample_rate: u32, n_fft: usize, kind: ArtifactKind) -> usize {
    let hz = match kind {
        ArtifactKind::Smearing => (config.fc as f32).max(10_000.0),
        ArtifactKind::HarmonicBleeding | ArtifactKind::PhaseIncoherence => config.fc as f32,
    };
    bin_of(hz, n_fft, sample_rate).min(n_fft / 2 + 1)
}

fn protected_end_hz(config: &Config, sample_rate: u32, kind: ArtifactKind) -> f32 {
    let start = match kind {
        ArtifactKind::Smearing => (config.fc as f32).max(10_000.0),
        ArtifactKind::HarmonicBleeding | ArtifactKind::PhaseIncoherence => config.fc as f32,
    };
    (start - 500.0).clamp(0.0, sample_rate as f32 / 2.0)
}

fn protected_band_snr(
    original_mid: &[f32],
    original_side: &[f32],
    candidate_mid: &[f32],
    candidate_side: &[f32],
    end_hz: f32,
    config: &Config,
    sample_rate: u32,
) -> f32 {
    let settings = StftConfig::new(config.n_fft, config.hop);
    let original_mid = stft(original_mid, &settings);
    let original_side = stft(original_side, &settings);
    let candidate_mid = stft(candidate_mid, &settings);
    let candidate_side = stft(candidate_side, &settings);
    let end = bin_of(end_hz, config.n_fft, sample_rate).min(config.n_fft / 2 + 1);
    let frames = original_mid
        .len()
        .min(original_side.len())
        .min(candidate_mid.len())
        .min(candidate_side.len());
    let mut signal = 0.0f64;
    let mut error = 0.0f64;
    for frame in 0..frames {
        for bin in 0..end {
            for (original, candidate) in [
                (&original_mid[frame], &candidate_mid[frame]),
                (&original_side[frame], &candidate_side[frame]),
            ] {
                let original_re = original.re(bin) as f64;
                let original_im = original.im(bin) as f64;
                let delta_re = original_re - candidate.re(bin) as f64;
                let delta_im = original_im - candidate.im(bin) as f64;
                signal += original_re * original_re + original_im * original_im;
                error += delta_re * delta_re + delta_im * delta_im;
            }
        }
    }
    if signal <= 1e-12 || error <= 1e-20 {
        f32::INFINITY
    } else {
        (10.0 * (signal / error).log10()) as f32
    }
}

fn spectral_flux(mid: &[SpectrumFrame], side: &[SpectrumFrame], start: usize) -> f64 {
    let frames = mid.len().min(side.len());
    if frames < 2 {
        return 0.0;
    }
    let n_bins = mid[0].n_bins.min(side[0].n_bins);
    let mut flux = 0.0f64;
    let mut energy = 0.0f64;
    for frame in 1..frames {
        for bin in start.min(n_bins)..n_bins {
            let current = mid[frame].mag(bin).hypot(side[frame].mag(bin)) as f64;
            let previous = mid[frame - 1].mag(bin).hypot(side[frame - 1].mag(bin)) as f64;
            flux += (current - previous).max(0.0).powi(2);
            energy += current * current;
        }
    }
    flux / energy.max(1e-12)
}

fn inter_peak_ratio(
    mid: &[SpectrumFrame],
    side: &[SpectrumFrame],
    start: usize,
    n_fft: usize,
    sample_rate: u32,
) -> f64 {
    let frames = mid.len().min(side.len());
    let n_bins = n_fft / 2 + 1;
    let radius = bin_of(600.0, n_fft, sample_rate).max(2);
    let mut valleys = 0.0f64;
    let mut peaks = 0.0f64;
    for frame in 0..frames {
        let magnitudes = (0..n_bins)
            .map(|bin| mid[frame].mag(bin).hypot(side[frame].mag(bin)))
            .collect::<Vec<_>>();
        let envelope = sliding_max(&magnitudes, radius);
        for bin in start.min(n_bins)..n_bins {
            let power = magnitudes[bin].powi(2) as f64;
            let ratio = magnitudes[bin] / envelope[bin].max(1e-8);
            if ratio >= 0.7 {
                peaks += power;
            } else if ratio <= 0.35 {
                valleys += power;
            }
        }
    }
    valleys / peaks.max(1e-12)
}

fn relative_phase_jitter(mid: &[SpectrumFrame], side: &[SpectrumFrame], start: usize) -> f64 {
    let frames = mid.len().min(side.len());
    if frames < 2 {
        return 0.0;
    }
    let n_bins = mid[0].n_bins.min(side[0].n_bins);
    let mut jitter = 0.0f64;
    let mut weights = 0.0f64;
    for frame in 1..frames {
        for bin in start.min(n_bins)..n_bins {
            let current_weight = mid[frame].mag(bin) as f64 * side[frame].mag(bin) as f64;
            let previous_weight = mid[frame - 1].mag(bin) as f64 * side[frame - 1].mag(bin) as f64;
            let weight = current_weight.min(previous_weight);
            if weight <= 1e-10 {
                continue;
            }
            let current = side[frame].phase(bin) - mid[frame].phase(bin);
            let previous = side[frame - 1].phase(bin) - mid[frame - 1].phase(bin);
            jitter += wrap_phase(current - previous).abs() as f64 * weight;
            weights += weight;
        }
    }
    jitter / weights.max(1e-12)
}

fn interchannel_correlation(mid: &[SpectrumFrame], side: &[SpectrumFrame], start: usize) -> f64 {
    let frames = mid.len().min(side.len());
    if frames == 0 {
        return 0.0;
    }
    let n_bins = mid[0].n_bins.min(side[0].n_bins);
    let mut dot = 0.0f64;
    let mut left_energy = 0.0f64;
    let mut right_energy = 0.0f64;
    for frame in 0..frames {
        for bin in start.min(n_bins)..n_bins {
            let mr = mid[frame].re(bin) as f64;
            let mi = mid[frame].im(bin) as f64;
            let sr = side[frame].re(bin) as f64;
            let si = side[frame].im(bin) as f64;
            let left_re = mr + sr;
            let left_im = mi + si;
            let right_re = mr - sr;
            let right_im = mi - si;
            dot += left_re * right_re + left_im * right_im;
            left_energy += left_re * left_re + left_im * left_im;
            right_energy += right_re * right_re + right_im * right_im;
        }
    }
    let denominator = (left_energy * right_energy).sqrt();
    if denominator <= 1e-12 {
        0.0
    } else {
        dot / denominator
    }
}

fn peak_retention_db(
    original_mid: &[f32],
    original_side: &[f32],
    candidate_mid: &[f32],
    candidate_side: &[f32],
    config: &Config,
    sample_rate: u32,
) -> f32 {
    let settings = StftConfig::new(config.n_fft, config.hop);
    let original_mid = stft(original_mid, &settings);
    let original_side = stft(original_side, &settings);
    let candidate_mid = stft(candidate_mid, &settings);
    let candidate_side = stft(candidate_side, &settings);
    let frames = original_mid
        .len()
        .min(original_side.len())
        .min(candidate_mid.len())
        .min(candidate_side.len());
    let n_bins = config.n_fft / 2 + 1;
    let start = bin_of(config.fc as f32, config.n_fft, sample_rate).min(n_bins);
    let radius = bin_of(600.0, config.n_fft, sample_rate).max(2);
    let mut original_energy = 0.0f64;
    let mut candidate_energy = 0.0f64;
    for frame in 0..frames {
        let magnitudes = (0..n_bins)
            .map(|bin| {
                original_mid[frame]
                    .mag(bin)
                    .hypot(original_side[frame].mag(bin))
            })
            .collect::<Vec<_>>();
        let envelope = sliding_max(&magnitudes, radius);
        for bin in start..n_bins {
            if magnitudes[bin] < envelope[bin] * 0.7 {
                continue;
            }
            original_energy += magnitudes[bin].powi(2) as f64;
            candidate_energy += candidate_mid[frame]
                .mag(bin)
                .hypot(candidate_side[frame].mag(bin))
                .powi(2) as f64;
        }
    }
    if original_energy <= 1e-12 {
        0.0
    } else {
        (10.0 * (candidate_energy / original_energy).max(1e-12).log10()) as f32
    }
}

fn band_energy_pair(mid: &[SpectrumFrame], side: &[SpectrumFrame], start: usize) -> f64 {
    band_energy(mid, start) + band_energy(side, start)
}

fn band_energy(spectrum: &[SpectrumFrame], start: usize) -> f64 {
    spectrum
        .iter()
        .map(|frame| {
            (start.min(frame.n_bins)..frame.n_bins)
                .map(|bin| frame.mag(bin).powi(2) as f64)
                .sum::<f64>()
        })
        .sum()
}

fn sliding_max(values: &[f32], radius: usize) -> Vec<f32> {
    let mut result = vec![0.0; values.len()];
    let mut deque = VecDeque::new();
    let mut next = 0usize;
    for (index, output) in result.iter_mut().enumerate() {
        let end = (index + radius + 1).min(values.len());
        while next < end {
            while deque
                .back()
                .is_some_and(|candidate| values[*candidate] <= values[next])
            {
                deque.pop_back();
            }
            deque.push_back(next);
            next += 1;
        }
        let start = index.saturating_sub(radius);
        while deque.front().is_some_and(|candidate| *candidate < start) {
            deque.pop_front();
        }
        *output = deque
            .front()
            .map(|candidate| values[*candidate])
            .unwrap_or(0.0);
    }
    result
}

fn blend(original: &[f32], candidate: &[f32], mix: f32) -> Vec<f32> {
    original
        .iter()
        .zip(candidate)
        .map(|(original, candidate)| original + mix * (candidate - original))
        .collect()
}

fn wrap_phase(value: f32) -> f32 {
    (value + std::f32::consts::PI).rem_euclid(2.0 * std::f32::consts::PI) - std::f32::consts::PI
}

fn bypass(mid: &[f32], side: &[f32]) -> GuardedAudio {
    GuardedAudio {
        mid: mid.to_vec(),
        side: side.to_vec(),
        retained_mix: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(frequency: f32, index: usize, sample_rate: u32) -> f32 {
        (2.0 * std::f32::consts::PI * frequency * index as f32 / sample_rate as f32).sin()
    }

    #[test]
    fn rejects_a_candidate_that_destroys_protected_content() {
        let sample_rate = 48_000;
        let length = sample_rate as usize;
        let mid = (0..length)
            .map(|index| {
                sine(1_000.0, index, sample_rate) * 0.2 + sine(12_000.0, index, sample_rate) * 0.05
            })
            .collect::<Vec<_>>();
        let side = mid.iter().map(|sample| sample * 0.3).collect::<Vec<_>>();
        let destroyed = vec![0.0f32; length];
        let guarded = guard(
            AudioPair {
                mid: &mid,
                side: &side,
            },
            AudioPair {
                mid: &destroyed,
                side: &destroyed,
            },
            ArtifactKind::Smearing,
            1.0,
            &Config::default(),
            sample_rate,
        );
        assert_eq!(guarded.retained_mix, 0.0);
        assert_eq!(guarded.mid, mid);
        assert_eq!(guarded.side, side);
    }

    #[test]
    fn rejects_non_finite_candidates() {
        let mid = vec![0.1f32; 4096];
        let side = vec![0.05f32; 4096];
        let mut candidate = mid.clone();
        candidate[100] = f32::NAN;
        let guarded = guard(
            AudioPair {
                mid: &mid,
                side: &side,
            },
            AudioPair {
                mid: &candidate,
                side: &side,
            },
            ArtifactKind::PhaseIncoherence,
            1.0,
            &Config::default(),
            48_000,
        );
        assert_eq!(guarded.retained_mix, 0.0);
        assert_eq!(guarded.mid, mid);
    }
}
