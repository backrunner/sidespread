//! Perceptual bandwidth extension for a shared M/S brick-wall cutoff.
//!
//! Missing waveform information cannot be recovered. This module maps intact lower-octave content
//! upward, follows each STFT frame's boundary envelope, and only adds an energy deficit.

use crate::analysis::spectrum::bin_of;
use crate::analysis::stft::{istft, stft, SpectrumFrame, StftConfig};
use crate::config::Config;
use crate::repair::common::{band_mask, phase_jitter};
use crate::repair::dsp::fill_energy_deficit;
use crate::terminal::TaskProgress;

pub fn repair_pair(
    mid: &[f32],
    side: &[f32],
    cutoff_hz: f32,
    config: &Config,
    sample_rate: u32,
) -> (Vec<f32>, Vec<f32>) {
    repair_pair_impl(mid, side, cutoff_hz, config, sample_rate, false)
}

pub fn repair_pair_with_progress(
    mid: &[f32],
    side: &[f32],
    cutoff_hz: f32,
    config: &Config,
    sample_rate: u32,
) -> (Vec<f32>, Vec<f32>) {
    repair_pair_impl(mid, side, cutoff_hz, config, sample_rate, true)
}

fn repair_pair_impl(
    mid: &[f32],
    side: &[f32],
    cutoff_hz: f32,
    config: &Config,
    sample_rate: u32,
    show_progress: bool,
) -> (Vec<f32>, Vec<f32>) {
    let settings = StftConfig::new(config.n_fft, config.hop);
    let frames = settings.num_frames(mid.len().min(side.len()));
    let progress = TaskProgress::new_if(
        "BANDWIDTH EXTEND",
        frames.saturating_add(2).saturating_mul(2),
        "steps",
        show_progress,
    );
    let mid = repair_channel(mid, cutoff_hz, config, sample_rate, &progress);
    let side = repair_channel(side, cutoff_hz, config, sample_rate, &progress);
    progress.finish();
    (mid, side)
}

fn repair_channel(
    signal: &[f32],
    cutoff_hz: f32,
    config: &Config,
    sample_rate: u32,
    progress: &TaskProgress,
) -> Vec<f32> {
    if signal.is_empty() || config.bandwidth_strength <= 0.0 {
        return signal.to_vec();
    }

    let stft_config = StftConfig::new(config.n_fft, config.hop);
    let original = stft(signal, &stft_config);
    progress.advance(1);
    let n_bins = config.n_fft / 2 + 1;
    let cutoff = bin_of(cutoff_hz, config.n_fft, sample_rate).clamp(2, n_bins - 1);
    let transition = bin_of(500.0, config.n_fft, sample_rate).max(2);
    let full_bin = (cutoff + transition).min(n_bins - 1);
    let frame_gains = boundary_gains(&original, cutoff, transition);
    let phase_spread = config.dsp_phase_degrees.to_radians();

    let mut repaired = Vec::with_capacity(original.len());
    for (frame_index, frame) in original.iter().enumerate() {
        let mut output = SpectrumFrame::new(n_bins);
        for bin in 0..n_bins {
            if bin < cutoff {
                output.cplx[2 * bin] = frame.re(bin);
                output.cplx[2 * bin + 1] = frame.im(bin);
                continue;
            }

            let source = extension_source_bin(bin, cutoff, n_bins);
            let harmonic = bin as f32 / source.max(1) as f32;
            let octave = (bin as f32 / cutoff as f32).log2().max(0.0);
            let rolloff = 10.0f32.powf(-config.bandwidth_rolloff_db_per_octave * octave / 20.0);
            let target =
                frame.mag(source) * frame_gains[frame_index] * rolloff * config.bandwidth_strength;
            let phase =
                frame.phase(source) * harmonic + phase_jitter(bin + frame_index * 17, phase_spread);
            let mask = band_mask(bin, cutoff, full_bin);
            let (re, im) = fill_energy_deficit(frame.re(bin), frame.im(bin), target, phase, mask);
            output.cplx[2 * bin] = re;
            output.cplx[2 * bin + 1] = im;
        }
        repaired.push(output);
        progress.advance(1);
    }

    let repaired = istft(&repaired, &stft_config, signal.len());
    progress.advance(1);
    repaired
}

fn boundary_gains(frames: &[SpectrumFrame], cutoff: usize, transition: usize) -> Vec<f32> {
    let lower_lo = cutoff.saturating_sub(transition * 2).max(1);
    let lower_hi = cutoff.saturating_sub(transition / 2).max(lower_lo + 1);
    let raw = frames
        .iter()
        .map(|frame| {
            let mut edge_energy = 0.0f64;
            let mut source_energy = 0.0f64;
            for bin in lower_lo..lower_hi {
                let source = (bin / 2).max(1);
                edge_energy += frame.mag(bin).powi(2) as f64;
                source_energy += frame.mag(source).powi(2) as f64;
            }
            if source_energy > 1e-12 {
                (edge_energy / source_energy).sqrt().clamp(0.02, 1.5) as f32
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();

    (0..raw.len())
        .map(|index| {
            let start = index.saturating_sub(2);
            let end = (index + 3).min(raw.len());
            let active = raw[start..end]
                .iter()
                .copied()
                .filter(|gain| *gain > 0.0)
                .collect::<Vec<_>>();
            if active.is_empty() {
                0.0
            } else {
                active.iter().sum::<f32>() / active.len() as f32
            }
        })
        .collect()
}

fn extension_source_bin(target: usize, cutoff: usize, n_bins: usize) -> usize {
    let source_lo = (cutoff / 2).max(1);
    let source_width = cutoff.saturating_sub(source_lo).max(1);
    let target_width = n_bins.saturating_sub(cutoff).max(1);
    let offset = target.saturating_sub(cutoff).min(target_width);
    (source_lo + offset * source_width / target_width).min(cutoff - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(length: usize) -> Vec<f32> {
        let mut state = 7u32;
        (0..length)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 16) as f32 / 32_768.0 - 1.0) * 0.15
            })
            .collect()
    }

    fn lowpass(signal: &[f32], cutoff_hz: f32, config: &Config, sample_rate: u32) -> Vec<f32> {
        let stft_config = StftConfig::new(config.n_fft, config.hop);
        let mut spectrum = stft(signal, &stft_config);
        let cutoff = bin_of(cutoff_hz, config.n_fft, sample_rate);
        for frame in &mut spectrum {
            for bin in cutoff..frame.n_bins {
                frame.cplx[2 * bin] = 0.0;
                frame.cplx[2 * bin + 1] = 0.0;
            }
        }
        istft(&spectrum, &stft_config, signal.len())
    }

    fn band_energy(signal: &[f32], start_hz: f32, config: &Config, sample_rate: u32) -> f64 {
        let stft_config = StftConfig::new(config.n_fft, config.hop);
        let spectrum = stft(signal, &stft_config);
        let start = bin_of(start_hz, config.n_fft, sample_rate);
        spectrum
            .iter()
            .map(|frame| {
                (start..frame.n_bins)
                    .map(|bin| frame.mag(bin).powi(2) as f64)
                    .sum::<f64>()
            })
            .sum()
    }

    #[test]
    fn extends_a_hard_cutoff_without_changing_length() {
        let config = Config::default();
        let input = lowpass(&noise(48_000), 12_000.0, &config, 48_000);
        let progress = TaskProgress::new_if("TEST", 0, "steps", false);
        let output = repair_channel(&input, 12_000.0, &config, 48_000, &progress);
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert!(
            band_energy(&output, 13_000.0, &config, 48_000)
                > band_energy(&input, 13_000.0, &config, 48_000) * 10.0
        );
    }

    #[test]
    fn source_bins_always_come_from_the_intact_band() {
        for target in 683..=2048 {
            assert!(extension_source_bin(target, 683, 2049) < 683);
        }
    }
}
