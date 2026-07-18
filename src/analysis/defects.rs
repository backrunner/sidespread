//! Dynamic multi-band Side deficiency analysis from the upper midrange to Nyquist.

use crate::analysis::spectrum::bin_of;
use crate::analysis::stft::SpectrumFrame;
use crate::repair::common::smoothstep;

const BAND_WIDTH_HZ: f32 = 500.0;
const DEFICIENT_RATIO_HIGH: f32 = 0.78;
const DEFICIENT_RATIO_LOW: f32 = 0.30;

#[derive(Debug, Clone, Copy)]
pub struct DefectProfile {
    pub relative_ratio_low: f32,
    pub deficient_band_fraction: f32,
    pub deficient_band_count: usize,
    pub active_band_count: usize,
    pub first_defect_hz: Option<f32>,
}

pub struct DefectMap {
    pub masks: Vec<Vec<f32>>,
    pub profile: DefectProfile,
}

pub fn analyze(
    mid: &[SpectrumFrame],
    side: &[SpectrumFrame],
    scan_start_hz: usize,
    n_fft: usize,
    sample_rate: u32,
) -> DefectMap {
    let frames = mid.len().min(side.len());
    let n_bins = n_fft / 2 + 1;
    if frames == 0 || n_bins == 0 {
        return DefectMap {
            masks: Vec::new(),
            profile: empty_profile(),
        };
    }

    let scan_start = bin_of(scan_start_hz as f32, n_fft, sample_rate).min(n_bins);
    let band_width = bin_of(BAND_WIDTH_HZ, n_fft, sample_rate).max(2);
    let band_count = n_bins.saturating_sub(scan_start).div_ceil(band_width);
    if band_count == 0 {
        return DefectMap {
            masks: vec![vec![0.0; n_bins]; frames],
            profile: empty_profile(),
        };
    }

    let anchor_start =
        bin_of((scan_start_hz as f32 * 0.60).max(500.0), n_fft, sample_rate).min(scan_start);
    let anchor_end = bin_of(
        (scan_start_hz as f32 - BAND_WIDTH_HZ).max(1_000.0),
        n_fft,
        sample_rate,
    )
    .clamp(
        anchor_start.saturating_add(1),
        scan_start.max(anchor_start + 1),
    );

    let mut relative = vec![vec![f32::INFINITY; band_count]; frames];
    let mut raw_strength = vec![vec![0.0f32; band_count]; frames];
    let mut active = vec![vec![false; band_count]; frames];
    for frame in 0..frames {
        let anchor_mid = band_energy(&mid[frame], anchor_start, anchor_end);
        let anchor_side = band_energy(&side[frame], anchor_start, anchor_end);
        let anchor_bins = anchor_end.saturating_sub(anchor_start).max(1) as f64;
        let anchor_mid_mean = anchor_mid / anchor_bins;
        let mut mid_energies = vec![0.0f64; band_count];
        let mut side_energies = vec![0.0f64; band_count];
        let mut maximum_mid_mean = 0.0f64;
        for band in 0..band_count {
            let start = scan_start + band * band_width;
            let end = (start + band_width).min(n_bins);
            let bins = end.saturating_sub(start).max(1) as f64;
            mid_energies[band] = band_energy(&mid[frame], start, end);
            side_energies[band] = band_energy(&side[frame], start, end);
            maximum_mid_mean = maximum_mid_mean.max(mid_energies[band] / bins);
        }
        let activity_floor = anchor_mid_mean.max(maximum_mid_mean) * 0.02;
        let mut active_band_ratios = Vec::new();
        for band in 0..band_count {
            let start = scan_start + band * band_width;
            let end = (start + band_width).min(n_bins);
            let bins = end.saturating_sub(start).max(1) as f64;
            let is_active =
                mid_energies[band] / bins >= activity_floor && mid_energies[band] > 1e-12;
            active[frame][band] = is_active;
            if is_active {
                active_band_ratios
                    .push((side_energies[band] / mid_energies[band].max(1e-12)) as f32);
            }
        }
        active_band_ratios.sort_by(f32::total_cmp);
        let spectral_anchor = percentile(&active_band_ratios, 0.75).unwrap_or(0.05) as f64;
        let anchor_ratio = if anchor_mid_mean >= maximum_mid_mean * 0.02 {
            anchor_side / anchor_mid.max(1e-12)
        } else {
            spectral_anchor
        }
        .clamp(0.0025, 4.0);

        for band in 0..band_count {
            if !active[frame][band] {
                continue;
            }
            let ratio = (side_energies[band] / mid_energies[band].max(1e-12) / anchor_ratio) as f32;
            relative[frame][band] = ratio;
            raw_strength[frame][band] = smoothstep(
                (DEFICIENT_RATIO_HIGH - ratio) / (DEFICIENT_RATIO_HIGH - DEFICIENT_RATIO_LOW),
            );
        }
    }

    let strengths = temporal_smooth(&raw_strength, &active, 2);
    let mut masks = vec![vec![0.0f32; n_bins]; frames];
    for frame in 0..frames {
        for (bin, mask) in masks[frame].iter_mut().enumerate().skip(scan_start) {
            let position = (bin - scan_start) as f32 / band_width as f32 - 0.5;
            let lower = position.floor() as isize;
            let fraction = position - lower as f32;
            let left = band_strength(&strengths[frame], lower);
            let right = band_strength(&strengths[frame], lower + 1);
            *mask = left + (right - left) * smoothstep(fraction);
        }
    }

    let mut active_ratios = relative
        .iter()
        .zip(&active)
        .flat_map(|(ratios, active)| {
            ratios
                .iter()
                .zip(active)
                .filter_map(|(ratio, active)| (*active && ratio.is_finite()).then_some(*ratio))
        })
        .collect::<Vec<_>>();
    active_ratios.sort_by(f32::total_cmp);
    let relative_ratio_low = percentile(&active_ratios, 0.20).unwrap_or(1.0);
    let band_means = (0..band_count)
        .map(|band| strengths.iter().map(|frame| frame[band]).sum::<f32>() / frames as f32)
        .collect::<Vec<_>>();
    let active_band_count = (0..band_count)
        .filter(|band| {
            active.iter().filter(|frame| frame[*band]).count() as f32 / frames as f32 >= 0.35
        })
        .count();
    let deficient_band_count = band_means
        .iter()
        .filter(|strength| **strength >= 0.35)
        .count();
    let deficient_band_fraction = deficient_band_count as f32 / band_count as f32;
    let first_defect_hz = band_means
        .iter()
        .position(|strength| *strength >= 0.35)
        .map(|band| (scan_start + band * band_width) as f32 * sample_rate as f32 / n_fft as f32);

    DefectMap {
        masks,
        profile: DefectProfile {
            relative_ratio_low,
            deficient_band_fraction,
            deficient_band_count,
            active_band_count,
            first_defect_hz,
        },
    }
}

fn temporal_smooth(raw: &[Vec<f32>], active: &[Vec<bool>], radius: usize) -> Vec<Vec<f32>> {
    let frames = raw.len();
    let bands = raw.first().map_or(0, Vec::len);
    let mut output = vec![vec![0.0f32; bands]; frames];
    for (frame, output_frame) in output.iter_mut().enumerate() {
        let start = frame.saturating_sub(radius);
        let end = (frame + radius + 1).min(frames);
        for band in 0..bands {
            let mut sum = 0.0f32;
            let mut weight = 0.0f32;
            for neighbor in start..end {
                if active[neighbor][band] {
                    let distance = frame.abs_diff(neighbor) as f32;
                    let local_weight = 1.0 / (1.0 + distance);
                    sum += raw[neighbor][band] * local_weight;
                    weight += local_weight;
                }
            }
            if weight > 0.0 {
                output_frame[band] = smoothstep((sum / weight - 0.15) / 0.70);
            }
        }
    }
    output
}

fn band_strength(strengths: &[f32], index: isize) -> f32 {
    if index < 0 {
        0.0
    } else {
        strengths.get(index as usize).copied().unwrap_or(0.0)
    }
}

fn band_energy(frame: &SpectrumFrame, start: usize, end: usize) -> f64 {
    (start.min(frame.n_bins)..end.min(frame.n_bins))
        .map(|bin| frame.mag(bin).powi(2) as f64)
        .sum()
}

fn percentile(values: &[f32], quantile: f32) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let index = ((values.len() - 1) as f32 * quantile.clamp(0.0, 1.0)).round() as usize;
    values.get(index).copied()
}

fn empty_profile() -> DefectProfile {
    DefectProfile {
        relative_ratio_low: 1.0,
        deficient_band_fraction: 0.0,
        deficient_band_count: 0,
        active_band_count: 0,
        first_defect_hz: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::stft::{stft, StftConfig};

    fn sine(frequency: f32, index: usize, sample_rate: u32) -> f32 {
        (2.0 * std::f32::consts::PI * frequency * index as f32 / sample_rate as f32).sin()
    }

    #[test]
    fn finds_multiple_deficient_bands_above_five_khz() {
        let sample_rate = 48_000;
        let length = sample_rate as usize;
        let frequencies = [4_000.0, 5_500.0, 7_000.0, 9_000.0, 12_000.0];
        let mid = (0..length)
            .map(|index| {
                frequencies
                    .iter()
                    .map(|frequency| 0.1 * sine(*frequency, index, sample_rate))
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let side = (0..length)
            .map(|index| {
                0.05 * sine(4_000.0, index, sample_rate)
                    + 0.05 * sine(7_000.0, index, sample_rate)
                    + 0.05 * sine(12_000.0, index, sample_rate)
            })
            .collect::<Vec<_>>();
        let settings = StftConfig::new(4096, 1024);
        let map = analyze(
            &stft(&mid, &settings),
            &stft(&side, &settings),
            5_000,
            4096,
            sample_rate,
        );

        assert!(map.profile.deficient_band_count >= 2);
        assert!(map.profile.relative_ratio_low < 0.4);
        let center = map.masks.len() / 2;
        let bin = |hz| bin_of(hz, 4096, sample_rate);
        assert!(map.masks[center][bin(5_500.0)] > 0.5);
        assert!(map.masks[center][bin(9_000.0)] > 0.5);
        assert!(map.masks[center][bin(7_000.0)] < 0.2);
    }
}
