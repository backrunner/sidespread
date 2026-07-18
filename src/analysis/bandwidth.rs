//! Track-level detection of a shared M/S brick-wall high-frequency cutoff.

use crate::analysis::spectrum::bin_of;
use crate::analysis::stft::{stft, StftConfig};
use crate::config::Config;

const MINIMUM_CONFIDENCE: f32 = 0.15;

/// Evidence for full-band extension, kept separate from Side deficiency metrics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BandwidthReport {
    /// Estimated common cutoff. `None` means no sufficiently sharp shared edge was found.
    pub detected_cutoff_hz: Option<f32>,
    /// Energy above the edge relative to the intact band below it.
    pub tail_to_edge_db: Option<f32>,
    /// Normalized margin beyond the configured hard-edge threshold.
    pub confidence: f32,
    pub needs_extension: bool,
}

impl BandwidthReport {
    pub fn healthy() -> Self {
        Self {
            detected_cutoff_hz: None,
            tail_to_edge_db: None,
            confidence: 0.0,
            needs_extension: false,
        }
    }
}

/// Analyze the combined M/S spectrum without retaining a whole-track STFT in memory.
pub fn analyze(m: &[f32], s: &[f32], config: &Config, sample_rate: u32) -> BandwidthReport {
    if !config.bandwidth_extension || m.len().min(s.len()) < sample_rate as usize / 2 {
        return BandwidthReport::healthy();
    }

    let stft_config = StftConfig::new(config.n_fft, config.hop);
    let n_bins = config.n_fft / 2 + 1;
    let mut average_power = vec![0.0f64; n_bins];
    let mut frame_count = 0usize;
    let chunk_length = sample_rate as usize;
    let length = m.len().min(s.len());

    for start in (0..length).step_by(chunk_length) {
        let end = (start + chunk_length).min(length);
        let m_spec = stft(&m[start..end], &stft_config);
        let s_spec = stft(&s[start..end], &stft_config);
        for (m_frame, s_frame) in m_spec.iter().zip(&s_spec) {
            for (bin, total) in average_power.iter_mut().enumerate() {
                let mr = m_frame.re(bin) as f64;
                let mi = m_frame.im(bin) as f64;
                let sr = s_frame.re(bin) as f64;
                let si = s_frame.im(bin) as f64;
                *total += mr * mr + mi * mi + sr * sr + si * si;
            }
            frame_count += 1;
        }
    }

    if frame_count == 0 {
        return BandwidthReport::healthy();
    }
    for value in &mut average_power {
        *value /= frame_count as f64;
    }
    detect_from_average_power(&average_power, config, sample_rate)
}

fn detect_from_average_power(power: &[f64], config: &Config, sample_rate: u32) -> BandwidthReport {
    let n_bins = power.len().min(config.n_fft / 2 + 1);
    if n_bins < 4 {
        return BandwidthReport::healthy();
    }

    let hz_to_bins = |hz: f32| bin_of(hz, config.n_fft, sample_rate).max(1);
    let search_start = bin_of(config.fc as f32, config.n_fft, sample_rate).min(n_bins);
    let tail_margin = hz_to_bins(1_500.0);
    let search_end = n_bins.saturating_sub(tail_margin);
    let lower_width = hz_to_bins(1_000.0);
    let edge_guard = hz_to_bins(200.0);
    let upper_width = hz_to_bins(1_000.0);

    if search_start >= search_end || search_start < lower_width {
        return BandwidthReport::healthy();
    }

    let reference_lo = hz_to_bins(1_000.0).min(search_start);
    let reference_hi = search_start.max(reference_lo + 1).min(n_bins);
    let reference_mean = mean(&power[reference_lo..reference_hi]);
    if reference_mean <= 1e-18 {
        return BandwidthReport::healthy();
    }
    let numerical_floor = reference_mean * 1e-12;

    for cutoff in search_start..search_end {
        let lower_lo = cutoff.saturating_sub(lower_width);
        let lower_hi = cutoff.saturating_sub(edge_guard).max(lower_lo + 1);
        let upper_lo = (cutoff + edge_guard).min(n_bins);
        let upper_hi = (upper_lo + upper_width).min(n_bins);
        if lower_hi > n_bins || upper_lo >= upper_hi {
            continue;
        }

        let lower = &power[lower_lo..lower_hi];
        let lower_mean = mean(lower);
        // Require a dense, materially active edge. This rejects isolated tones and quiet endings.
        let occupancy = lower
            .iter()
            .filter(|value| **value >= lower_mean * 0.01)
            .count() as f32
            / lower.len().max(1) as f32;
        if lower_mean < reference_mean * 1e-4 || occupancy < 0.30 {
            continue;
        }

        let immediate_tail = mean(&power[upper_lo..upper_hi]);
        let far_tail_lo = upper_hi;
        let far_tail = if far_tail_lo < n_bins {
            mean(&power[far_tail_lo..n_bins])
        } else {
            immediate_tail
        };
        let tail = immediate_tail.max(far_tail).max(numerical_floor);
        let drop_db = 10.0 * (tail / lower_mean).log10() as f32;
        if drop_db <= config.bandwidth_drop_db {
            let cutoff_hz = cutoff as f32 * sample_rate as f32 / config.n_fft as f32;
            let confidence = ((config.bandwidth_drop_db - drop_db) / 20.0).clamp(0.0, 1.0);
            if confidence < MINIMUM_CONFIDENCE {
                continue;
            }
            return BandwidthReport {
                detected_cutoff_hz: Some(cutoff_hz),
                tail_to_edge_db: Some(drop_db),
                confidence,
                needs_extension: true,
            };
        }
    }

    BandwidthReport::healthy()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_spectrum(sample_rate: u32, config: &Config) -> Vec<f64> {
        (0..=config.n_fft / 2)
            .map(|bin| {
                let hz = bin as f64 * sample_rate as f64 / config.n_fft as f64;
                1.0 / (1.0 + hz / 2_000.0)
            })
            .collect()
    }

    #[test]
    fn detects_a_dense_brick_wall_cutoff() {
        let config = Config::default();
        let sample_rate = 48_000;
        let mut power = synthetic_spectrum(sample_rate, &config);
        let cutoff = bin_of(16_000.0, config.n_fft, sample_rate);
        for value in &mut power[cutoff..] {
            *value *= 1e-8;
        }

        let report = detect_from_average_power(&power, &config, sample_rate);
        assert!(report.needs_extension);
        assert!((report.detected_cutoff_hz.unwrap() - 16_000.0).abs() < 350.0);
        assert!(report.tail_to_edge_db.unwrap() < -30.0);
    }

    #[test]
    fn ignores_a_healthy_gradual_rolloff() {
        let config = Config::default();
        let report =
            detect_from_average_power(&synthetic_spectrum(48_000, &config), &config, 48_000);
        assert!(!report.needs_extension);
    }

    #[test]
    fn ignores_an_isolated_tonal_edge() {
        let config = Config::default();
        let mut power = vec![1e-12; config.n_fft / 2 + 1];
        let tone = bin_of(10_000.0, config.n_fft, 48_000);
        power[tone] = 1.0;
        let report = detect_from_average_power(&power, &config, 48_000);
        assert!(!report.needs_extension);
    }

    #[test]
    fn ignores_a_cutoff_without_enough_threshold_margin() {
        let config = Config::default();
        let sample_rate = 48_000;
        let mut power = vec![1.0; config.n_fft / 2 + 1];
        let cutoff = bin_of(16_000.0, config.n_fft, sample_rate);
        for value in &mut power[cutoff..] {
            *value = 10.0f64.powf(-3.1);
        }

        let report = detect_from_average_power(&power, &config, sample_rate);

        assert!(!report.needs_extension);
    }
}
