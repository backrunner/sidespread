//! Deficiency detector: per-segment metrics (R_hf, LSD, corr) and route decision.

use crate::analysis::spectrum::{bin_of, log_power, power};
use crate::analysis::stft::{stft, SpectrumFrame, StftConfig};
use crate::config::{Config, Route};

/// Metrics for one analyzed segment.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SegmentMetrics {
    /// High-frequency energy ratio: E_S(>fc) / E_M(>fc).
    pub r_hf: f32,
    /// Log-spectral distance over [fc, Nyquist] between M and S (lower = more similar).
    pub lsd_hf: f32,
    /// High-frequency normalized cross-correlation between M and S magnitudes.
    pub corr_hf: f32,
    /// S/M energy ratio in the intact band immediately below the cutoff.
    pub r_intact: f32,
    /// Complex M/S correlation in the intact band, used for route selection.
    pub corr_intact: f32,
}

/// Report for one segment, including route decision.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SegmentReport {
    pub start: usize,
    pub end: usize,
    pub needs_processing: bool,
    pub route: Route,
    pub metrics: SegmentMetrics,
}

/// Analyze a single segment of M and S, returning metrics + route.
pub fn analyze(
    m_seg: &[f32],
    s_seg: &[f32],
    start: usize,
    end: usize,
    cfg: &Config,
    sample_rate: u32,
) -> SegmentReport {
    let stft_cfg = StftConfig::new(cfg.n_fft, cfg.hop);
    let m_spec = stft(m_seg, &stft_cfg);
    let s_spec = stft(s_seg, &stft_cfg);
    let metrics = compute_metrics(&m_spec, &s_spec, cfg.fc, sample_rate, cfg.n_fft);
    let needs = metrics.r_hf < cfg.rhf_threshold;
    let route = cfg.decide(needs, metrics.corr_intact);
    SegmentReport {
        start,
        end,
        needs_processing: needs,
        route,
        metrics,
    }
}

/// Compute the three metrics across aligned frames.
pub fn compute_metrics(
    m_spec: &[SpectrumFrame],
    s_spec: &[SpectrumFrame],
    fc_hz: usize,
    sample_rate: u32,
    n_fft: usize,
) -> SegmentMetrics {
    let fc_bin = bin_of(fc_hz as f32, n_fft, sample_rate);
    let n_bins = n_fft / 2 + 1;

    let mut e_m_hf = 0.0f64;
    let mut e_s_hf = 0.0f64;
    let mut lsd_sum = 0.0f64;
    let mut lsd_count = 0u32;
    let mut corr_num = 0.0f64;
    let mut corr_m_den = 0.0f64;
    let mut corr_s_den = 0.0f64;
    let transition_hz = 500.0f32;
    let intact_lo = bin_of((fc_hz as f32 * 0.5).max(500.0), n_fft, sample_rate).min(fc_bin);
    let intact_hi = bin_of(
        (fc_hz as f32 - transition_hz).max(500.0),
        n_fft,
        sample_rate,
    )
    .clamp(intact_lo.saturating_add(1), fc_bin.max(intact_lo + 1));
    let mut intact_m_energy = 0.0f64;
    let mut intact_s_energy = 0.0f64;
    let mut intact_corr_num = 0.0f64;

    let frames = m_spec.len().min(s_spec.len());
    for f in 0..frames {
        let m_pow = power(&m_spec[f]);
        let s_pow = power(&s_spec[f]);
        let m_log = log_power(&m_spec[f], 1e-10);
        let s_log = log_power(&s_spec[f], 1e-10);
        for b in intact_lo..intact_hi {
            let mr = m_spec[f].re(b) as f64;
            let mi = m_spec[f].im(b) as f64;
            let sr = s_spec[f].re(b) as f64;
            let si = s_spec[f].im(b) as f64;
            intact_m_energy += mr * mr + mi * mi;
            intact_s_energy += sr * sr + si * si;
            intact_corr_num += mr * sr + mi * si;
        }
        let hf_bins = n_bins.saturating_sub(fc_bin).max(1);
        let m_mean = m_log[fc_bin..n_bins].iter().sum::<f32>() / hf_bins as f32;
        let s_mean = s_log[fc_bin..n_bins].iter().sum::<f32>() / hf_bins as f32;
        for b in fc_bin..n_bins {
            e_m_hf += m_pow[b] as f64;
            e_s_hf += s_pow[b] as f64;
            let d = ((m_log[b] - m_mean) - (s_log[b] - s_mean)) as f64;
            lsd_sum += d * d;
            lsd_count += 1;
            let mr = m_spec[f].re(b) as f64;
            let mi = m_spec[f].im(b) as f64;
            let sr = s_spec[f].re(b) as f64;
            let si = s_spec[f].im(b) as f64;
            corr_num += mr * sr + mi * si;
            corr_m_den += mr * mr + mi * mi;
            corr_s_den += sr * sr + si * si;
        }
    }

    let r_hf = if e_m_hf > 1e-12 {
        (e_s_hf / e_m_hf) as f32
    } else if e_s_hf > 1e-12 {
        1.0e6
    } else {
        1.0
    };
    let lsd_hf = if lsd_count > 0 {
        (lsd_sum / lsd_count as f64).sqrt() as f32
    } else {
        0.0
    };
    let corr_hf = if e_m_hf <= 1e-12 && e_s_hf <= 1e-12 {
        0.0
    } else {
        let denom = (corr_m_den * corr_s_den).sqrt();
        if denom > 1e-12 {
            (corr_num / denom) as f32
        } else {
            0.0
        }
    };
    let r_intact = if intact_m_energy > 1e-12 {
        (intact_s_energy / intact_m_energy) as f32
    } else if intact_s_energy > 1e-12 {
        1.0e6
    } else {
        1.0
    };
    let intact_denom = (intact_m_energy * intact_s_energy).sqrt();
    let corr_intact = if intact_denom > 1e-12 {
        (intact_corr_num / intact_denom) as f32
    } else {
        0.0
    };

    SegmentMetrics {
        r_hf,
        lsd_hf,
        corr_hf,
        r_intact,
        corr_intact,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white_noise(n: usize, seed: u32) -> Vec<f32> {
        let mut x = seed;
        let mut out = vec![0.0f32; n];
        for sample in out.iter_mut().take(n) {
            // simple LCG
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *sample = ((x >> 16) as f32 / 32768.0) - 1.0;
        }
        out
    }

    #[test]
    fn r_hf_zero_for_silent_side() {
        let m = white_noise(8192, 42);
        let s = vec![0.0f32; 8192];
        let stft_cfg = StftConfig::new(4096, 1024);
        let m_spec = stft(&m, &stft_cfg);
        let s_spec = stft(&s, &stft_cfg);
        let mtr = compute_metrics(&m_spec, &s_spec, 8000, 48000, 4096);
        assert!(
            mtr.r_hf < 1e-6,
            "r_hf should be ~0 for silent side, got {}",
            mtr.r_hf
        );
    }

    #[test]
    fn r_hf_near_one_for_identical_channels() {
        let m = white_noise(8192, 7);
        let s = m.clone();
        let stft_cfg = StftConfig::new(4096, 1024);
        let m_spec = stft(&m, &stft_cfg);
        let s_spec = stft(&s, &stft_cfg);
        let mtr = compute_metrics(&m_spec, &s_spec, 8000, 48000, 4096);
        assert!(
            (mtr.r_hf - 1.0).abs() < 1e-3,
            "r_hf should be ~1 for identical, got {}",
            mtr.r_hf
        );
        assert!(
            (mtr.corr_hf - 1.0).abs() < 1e-3,
            "corr_hf should be ~1 for identical, got {}",
            mtr.corr_hf
        );
    }

    #[test]
    fn independent_signals_have_low_complex_correlation() {
        let m = white_noise(8192, 7);
        let s = white_noise(8192, 99)
            .into_iter()
            .map(|sample| sample * 0.2)
            .collect::<Vec<_>>();
        let stft_cfg = StftConfig::new(4096, 1024);
        let metrics = compute_metrics(
            &stft(&m, &stft_cfg),
            &stft(&s, &stft_cfg),
            8000,
            48000,
            4096,
        );
        assert!(metrics.corr_hf.abs() < 0.2, "corr={}", metrics.corr_hf);
    }

    #[test]
    fn side_only_high_frequencies_are_not_deficient() {
        let sample_rate = 48_000;
        let length = 8192;
        let m = vec![0.0f32; length];
        let s = (0..length)
            .map(|index| {
                let phase =
                    2.0 * std::f32::consts::PI * 10_000.0 * index as f32 / sample_rate as f32;
                phase.sin() * 0.25
            })
            .collect::<Vec<_>>();
        let report = analyze(&m, &s, 0, length, &Config::default(), sample_rate);
        assert!(report.metrics.r_hf >= 1.0e6);
        assert!(!report.needs_processing);
        assert_eq!(report.route, Route::Skip);
    }

    #[test]
    fn intact_correlation_routes_missing_high_band_to_dsp() {
        let sample_rate = 48_000;
        let length = 8192;
        let signal = |frequency: f32, index: usize| {
            (2.0 * std::f32::consts::PI * frequency * index as f32 / sample_rate as f32).sin()
        };
        let mid = (0..length)
            .map(|index| signal(6_000.0, index) * 0.25 + signal(12_000.0, index) * 0.2)
            .collect::<Vec<_>>();
        let side = (0..length)
            .map(|index| signal(6_000.0, index) * 0.1)
            .collect::<Vec<_>>();

        let report = analyze(&mid, &side, 0, length, &Config::default(), sample_rate);

        assert!(report.needs_processing);
        assert!(report.metrics.corr_hf.abs() < 0.2);
        assert!(report.metrics.corr_intact > 0.9);
        assert_eq!(report.route, Route::Dsp);
    }
}
