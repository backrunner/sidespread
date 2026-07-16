//! Evaluation metrics for M/S alignment and ground-truth comparisons.

use crate::analysis::spectrum::{bin_of, log_power, power};
use crate::analysis::stft::{istft, stft, SpectrumFrame, StftConfig};

#[derive(Debug, Clone, serde::Serialize)]
pub struct QualityMetrics {
    pub r_hf: f32,
    pub lsd_hf: f32,
    pub mcd: f32,
    pub iccc_hf: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProcessingMetrics {
    pub before: QualityMetrics,
    pub after: QualityMetrics,
    /// Fixed gain applied to the complete repaired output to preserve peak headroom.
    pub output_gain_db: f32,
    /// Fraction of the synthesized side delta retained after the loudness safety cap.
    pub synthesis_mix: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ReferenceMetrics {
    pub lsd_hf: f32,
    pub snr_db: Option<f32>,
    /// Spectral SNR below the 500 Hz repair transition, which should remain untouched.
    pub snr_preserved_db: Option<f32>,
    pub snr_hf_db: Option<f32>,
    /// Fraction of the reference STFT energy at or above the configured cutoff.
    pub reference_hf_ratio: Option<f32>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EvaluationMetrics {
    pub reference: QualityMetrics,
    pub degraded: ReferenceMetrics,
    pub repaired: ReferenceMetrics,
    /// Repaired HF projected onto the degraded input HF; negative values indicate attenuation.
    pub existing_hf_projection_db: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct MetricConfig {
    pub fc_hz: usize,
    pub n_fft: usize,
    pub hop: usize,
    pub sample_rate: u32,
}

pub fn quality_metrics(
    m: &[f32],
    s: &[f32],
    l: &[f32],
    r: &[f32],
    settings: MetricConfig,
) -> QualityMetrics {
    let cfg = StftConfig::new(settings.n_fft, settings.hop);
    let m_spec = stft(m, &cfg);
    let s_spec = stft(s, &cfg);
    let (r_hf, lsd_hf) = spectral_alignment(
        &m_spec,
        &s_spec,
        settings.fc_hz,
        settings.n_fft,
        settings.sample_rate,
    );
    let mcd = mel_cepstral_distance(
        &m_spec,
        &s_spec,
        settings.n_fft,
        settings.sample_rate,
        40,
        13,
    );
    let iccc_hf = high_frequency_correlation(l, r, settings.fc_hz, &cfg, settings.sample_rate);

    QualityMetrics {
        r_hf,
        lsd_hf,
        mcd,
        iccc_hf,
    }
}

pub fn compare_reference(
    reference: &[f32],
    candidate: &[f32],
    settings: MetricConfig,
) -> ReferenceMetrics {
    let cfg = StftConfig::new(settings.n_fft, settings.hop);
    let ref_spec = stft(reference, &cfg);
    let candidate_spec = stft(candidate, &cfg);
    let (_, lsd_hf) = spectral_alignment(
        &ref_spec,
        &candidate_spec,
        settings.fc_hz,
        settings.n_fft,
        settings.sample_rate,
    );

    ReferenceMetrics {
        lsd_hf,
        snr_db: snr(reference, candidate),
        snr_preserved_db: spectral_snr_band_aligned(
            &ref_spec,
            &candidate_spec,
            0,
            bin_of(
                settings.fc_hz.saturating_sub(500) as f32,
                settings.n_fft,
                settings.sample_rate,
            ),
        ),
        snr_hf_db: spectral_snr(
            &ref_spec,
            &candidate_spec,
            settings.fc_hz,
            settings.n_fft,
            settings.sample_rate,
        ),
        reference_hf_ratio: spectral_energy_ratio(
            &ref_spec,
            settings.fc_hz,
            settings.n_fft,
            settings.sample_rate,
        ),
    }
}

pub fn high_band_projection_db(
    reference: &[f32],
    candidate: &[f32],
    settings: MetricConfig,
) -> Option<f32> {
    let cfg = StftConfig::new(settings.n_fft, settings.hop);
    let reference = stft(reference, &cfg);
    let candidate = stft(candidate, &cfg);
    let n_bins = settings.n_fft / 2 + 1;
    let start = bin_of(settings.fc_hz as f32, settings.n_fft, settings.sample_rate).min(n_bins);
    let frames = reference.len().min(candidate.len());
    let mut reference_energy = 0.0f64;
    let mut projection = 0.0f64;
    for frame in 0..frames {
        let bins = reference[frame].n_bins.min(candidate[frame].n_bins);
        for bin in start.min(bins)..bins {
            let rr = reference[frame].re(bin) as f64;
            let ri = reference[frame].im(bin) as f64;
            let cr = candidate[frame].re(bin) as f64;
            let ci = candidate[frame].im(bin) as f64;
            reference_energy += rr * rr + ri * ri;
            projection += rr * cr + ri * ci;
        }
    }
    if reference_energy <= 1e-12 {
        None
    } else {
        Some((20.0 * (projection / reference_energy).max(1e-12).log10()) as f32)
    }
}

fn spectral_energy_ratio(
    spectrum: &[SpectrumFrame],
    fc_hz: usize,
    n_fft: usize,
    sample_rate: u32,
) -> Option<f32> {
    let n_bins = n_fft / 2 + 1;
    let cutoff = bin_of(fc_hz as f32, n_fft, sample_rate).min(n_bins);
    let mut total = 0.0f64;
    let mut high = 0.0f64;
    for frame in spectrum {
        for bin in 0..n_bins.min(frame.n_bins) {
            let re = frame.re(bin) as f64;
            let im = frame.im(bin) as f64;
            let energy = re * re + im * im;
            total += energy;
            if bin >= cutoff {
                high += energy;
            }
        }
    }
    if total <= 1e-12 {
        None
    } else {
        Some((high / total) as f32)
    }
}

fn spectral_snr(
    reference: &[SpectrumFrame],
    candidate: &[SpectrumFrame],
    fc_hz: usize,
    n_fft: usize,
    sample_rate: u32,
) -> Option<f32> {
    let n_bins = n_fft / 2 + 1;
    let cutoff = bin_of(fc_hz as f32, n_fft, sample_rate).min(n_bins);
    spectral_snr_band(reference, candidate, cutoff, n_bins)
}

fn spectral_snr_band(
    reference: &[SpectrumFrame],
    candidate: &[SpectrumFrame],
    start_bin: usize,
    end_bin: usize,
) -> Option<f32> {
    let n_bins = reference
        .first()
        .map(|frame| frame.n_bins)
        .unwrap_or(0)
        .min(candidate.first().map(|frame| frame.n_bins).unwrap_or(0));
    let start = start_bin.min(n_bins);
    let end = end_bin.min(n_bins).max(start);
    let frames = reference.len().min(candidate.len());
    let mut signal = 0.0f64;
    let mut noise = 0.0f64;
    for frame in 0..frames {
        for bin in start..end {
            let reference_re = reference[frame].re(bin) as f64;
            let reference_im = reference[frame].im(bin) as f64;
            let error_re = reference_re - candidate[frame].re(bin) as f64;
            let error_im = reference_im - candidate[frame].im(bin) as f64;
            signal += reference_re * reference_re + reference_im * reference_im;
            noise += error_re * error_re + error_im * error_im;
        }
    }
    if signal <= 1e-12 {
        None
    } else if noise <= 1e-12 {
        Some(f32::INFINITY)
    } else {
        Some((10.0 * (signal / noise).log10()) as f32)
    }
}

fn spectral_snr_band_aligned(
    reference: &[SpectrumFrame],
    candidate: &[SpectrumFrame],
    start_bin: usize,
    end_bin: usize,
) -> Option<f32> {
    let n_bins = reference
        .first()
        .map(|frame| frame.n_bins)
        .unwrap_or(0)
        .min(candidate.first().map(|frame| frame.n_bins).unwrap_or(0));
    let start = start_bin.min(n_bins);
    let end = end_bin.min(n_bins).max(start);
    let frames = reference.len().min(candidate.len());
    let mut reference_energy = 0.0f64;
    let mut candidate_energy = 0.0f64;
    let mut dot = 0.0f64;
    for frame in 0..frames {
        for bin in start..end {
            let rr = reference[frame].re(bin) as f64;
            let ri = reference[frame].im(bin) as f64;
            let cr = candidate[frame].re(bin) as f64;
            let ci = candidate[frame].im(bin) as f64;
            reference_energy += rr * rr + ri * ri;
            candidate_energy += cr * cr + ci * ci;
            dot += rr * cr + ri * ci;
        }
    }
    if reference_energy <= 1e-12 {
        return None;
    }
    let gain = if candidate_energy > 1e-12 {
        dot / candidate_energy
    } else {
        1.0
    };
    let mut noise = 0.0f64;
    for frame in 0..frames {
        for bin in start..end {
            let error_re = reference[frame].re(bin) as f64 - gain * candidate[frame].re(bin) as f64;
            let error_im = reference[frame].im(bin) as f64 - gain * candidate[frame].im(bin) as f64;
            noise += error_re * error_re + error_im * error_im;
        }
    }
    if noise <= 1e-12 {
        Some(f32::INFINITY)
    } else {
        Some((10.0 * (reference_energy / noise).log10()) as f32)
    }
}

fn spectral_alignment(
    a_spec: &[SpectrumFrame],
    b_spec: &[SpectrumFrame],
    fc_hz: usize,
    n_fft: usize,
    sample_rate: u32,
) -> (f32, f32) {
    let n_bins = n_fft / 2 + 1;
    let fc_bin = bin_of(fc_hz as f32, n_fft, sample_rate).min(n_bins);
    let frames = a_spec.len().min(b_spec.len());
    let mut a_energy = 0.0f64;
    let mut b_energy = 0.0f64;
    let mut lsd_sum = 0.0f64;
    let mut lsd_count = 0usize;

    for frame in 0..frames {
        let a_power = power(&a_spec[frame]);
        let b_power = power(&b_spec[frame]);
        let a_log = log_power(&a_spec[frame], 1e-10);
        let b_log = log_power(&b_spec[frame], 1e-10);
        for bin in fc_bin..n_bins {
            a_energy += a_power[bin] as f64;
            b_energy += b_power[bin] as f64;
            let delta = (a_log[bin] - b_log[bin]) as f64;
            lsd_sum += delta * delta;
            lsd_count += 1;
        }
    }

    let ratio = if a_energy > 1e-12 {
        (b_energy / a_energy) as f32
    } else if b_energy <= 1e-12 {
        1.0
    } else {
        1.0e6
    };
    let lsd = if lsd_count > 0 {
        (lsd_sum / lsd_count as f64).sqrt() as f32
    } else {
        0.0
    };
    (ratio, lsd)
}

fn mel_cepstral_distance(
    a_spec: &[SpectrumFrame],
    b_spec: &[SpectrumFrame],
    n_fft: usize,
    sample_rate: u32,
    mel_bands: usize,
    coefficients: usize,
) -> f32 {
    let filters = mel_filter_bank(n_fft, sample_rate, mel_bands);
    let frames = a_spec.len().min(b_spec.len());
    if frames == 0 {
        return 0.0;
    }

    let mut total = 0.0f64;
    for frame in 0..frames {
        let a_cepstrum = mel_cepstrum(&power(&a_spec[frame]), &filters, coefficients);
        let b_cepstrum = mel_cepstrum(&power(&b_spec[frame]), &filters, coefficients);
        let squared = a_cepstrum
            .iter()
            .zip(&b_cepstrum)
            .map(|(a, b)| {
                let delta = (*a - *b) as f64;
                delta * delta
            })
            .sum::<f64>();
        total += squared.sqrt();
    }

    let scale = 10.0 / std::f64::consts::LN_10 * 2.0f64.sqrt();
    (scale * total / frames as f64) as f32
}

fn mel_filter_bank(n_fft: usize, sample_rate: u32, mel_bands: usize) -> Vec<Vec<f32>> {
    let n_bins = n_fft / 2 + 1;
    let nyquist = sample_rate as f32 / 2.0;
    let mel_max = hz_to_mel(nyquist);
    let mel_points = (0..mel_bands + 2)
        .map(|index| mel_max * index as f32 / (mel_bands + 1) as f32)
        .map(mel_to_hz)
        .collect::<Vec<_>>();
    let bins = mel_points
        .iter()
        .map(|hz| bin_of(*hz, n_fft, sample_rate).min(n_bins - 1))
        .collect::<Vec<_>>();

    (0..mel_bands)
        .map(|band| {
            let mut filter = vec![0.0f32; n_bins];
            let left = bins[band];
            let center = bins[band + 1].max(left + 1).min(n_bins - 1);
            let right = bins[band + 2].max(center + 1).min(n_bins - 1);
            for (bin, value) in filter.iter_mut().enumerate().take(center).skip(left) {
                *value = (bin - left) as f32 / (center - left) as f32;
            }
            for (bin, value) in filter.iter_mut().enumerate().take(right + 1).skip(center) {
                *value = (right - bin) as f32 / (right - center) as f32;
            }
            filter
        })
        .collect()
}

fn mel_cepstrum(power_spectrum: &[f32], filters: &[Vec<f32>], coefficients: usize) -> Vec<f32> {
    let log_mel = filters
        .iter()
        .map(|filter| {
            power_spectrum
                .iter()
                .zip(filter)
                .map(|(power, weight)| power * weight)
                .sum::<f32>()
                .max(1e-10)
                .ln()
        })
        .collect::<Vec<_>>();
    let bands = log_mel.len() as f32;

    (1..=coefficients)
        .map(|coefficient| {
            log_mel
                .iter()
                .enumerate()
                .map(|(band, value)| {
                    value
                        * (std::f32::consts::PI * coefficient as f32 * (band as f32 + 0.5) / bands)
                            .cos()
                })
                .sum::<f32>()
        })
        .collect()
}

fn high_frequency_correlation(
    l: &[f32],
    r: &[f32],
    fc_hz: usize,
    cfg: &StftConfig,
    sample_rate: u32,
) -> f32 {
    let l_hf = highpass(l, fc_hz, cfg, sample_rate);
    let r_hf = highpass(r, fc_hz, cfg, sample_rate);
    pearson_correlation(&l_hf, &r_hf)
}

fn highpass(signal: &[f32], fc_hz: usize, cfg: &StftConfig, sample_rate: u32) -> Vec<f32> {
    let mut spectrum = stft(signal, cfg);
    let cutoff = bin_of(fc_hz as f32, cfg.n_fft, sample_rate).min(cfg.n_fft / 2 + 1);
    for frame in &mut spectrum {
        for bin in 0..cutoff {
            frame.cplx[2 * bin] = 0.0;
            frame.cplx[2 * bin + 1] = 0.0;
        }
    }
    istft(&spectrum, cfg, signal.len())
}

fn pearson_correlation(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let a_mean = a[..len].iter().sum::<f32>() / len as f32;
    let b_mean = b[..len].iter().sum::<f32>() / len as f32;
    let mut numerator = 0.0f64;
    let mut a_energy = 0.0f64;
    let mut b_energy = 0.0f64;
    for index in 0..len {
        let av = (a[index] - a_mean) as f64;
        let bv = (b[index] - b_mean) as f64;
        numerator += av * bv;
        a_energy += av * av;
        b_energy += bv * bv;
    }
    let denominator = (a_energy * b_energy).sqrt();
    if denominator > 1e-12 {
        (numerator / denominator) as f32
    } else {
        0.0
    }
}

fn snr(reference: &[f32], candidate: &[f32]) -> Option<f32> {
    if reference.len() != candidate.len() || reference.is_empty() {
        return None;
    }
    let (signal, noise) = reference.iter().zip(candidate).fold(
        (0.0f64, 0.0f64),
        |(signal, noise), (reference, candidate)| {
            (
                signal + *reference as f64 * *reference as f64,
                noise + (*candidate as f64 - *reference as f64).powi(2),
            )
        },
    );
    if signal <= 1e-12 {
        None
    } else if noise <= 1e-12 {
        Some(f32::INFINITY)
    } else {
        Some((10.0 * (signal / noise).log10()) as f32)
    }
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(frequency: f32, sample_rate: u32, length: usize) -> Vec<f32> {
        (0..length)
            .map(|index| {
                (2.0 * std::f32::consts::PI * frequency * index as f32 / sample_rate as f32).sin()
                    * 0.25
            })
            .collect()
    }

    #[test]
    fn identical_channels_have_zero_distances() {
        let signal = sine(10_000.0, 48_000, 8192);
        let settings = MetricConfig {
            fc_hz: 8000,
            n_fft: 4096,
            hop: 1024,
            sample_rate: 48_000,
        };
        let quality = quality_metrics(&signal, &signal, &signal, &signal, settings);
        assert!((quality.r_hf - 1.0).abs() < 1e-4);
        assert!(quality.lsd_hf < 1e-4);
        assert!(quality.mcd < 1e-4);
        assert!(quality.iccc_hf > 0.999);
    }

    #[test]
    fn side_only_high_band_reports_a_large_energy_ratio() {
        let side = sine(10_000.0, 48_000, 8192);
        let mid = vec![0.0; side.len()];
        let left = side.clone();
        let right = side.iter().map(|sample| -sample).collect::<Vec<_>>();
        let settings = MetricConfig {
            fc_hz: 8000,
            n_fft: 4096,
            hop: 1024,
            sample_rate: 48_000,
        };

        let quality = quality_metrics(&mid, &side, &left, &right, settings);

        assert!(quality.r_hf >= 1.0e6);
    }

    #[test]
    fn high_band_projection_tracks_retained_reference_gain() {
        let reference = sine(10_000.0, 48_000, 8192);
        let candidate = reference
            .iter()
            .map(|sample| sample * 0.5)
            .collect::<Vec<_>>();
        let settings = MetricConfig {
            fc_hz: 8000,
            n_fft: 4096,
            hop: 1024,
            sample_rate: 48_000,
        };

        let projection = high_band_projection_db(&reference, &candidate, settings).unwrap();

        assert!((projection - 20.0 * 0.5f32.log10()).abs() < 1e-4);
    }

    #[test]
    fn ground_truth_metrics_reward_a_closer_candidate() {
        let reference = sine(10_000.0, 48_000, 8192);
        let degraded = reference
            .iter()
            .map(|sample| sample * 0.1)
            .collect::<Vec<_>>();
        let repaired = reference
            .iter()
            .map(|sample| sample * 0.9)
            .collect::<Vec<_>>();
        let settings = MetricConfig {
            fc_hz: 8000,
            n_fft: 4096,
            hop: 1024,
            sample_rate: 48_000,
        };
        let degraded_metrics = compare_reference(&reference, &degraded, settings);
        let repaired_metrics = compare_reference(&reference, &repaired, settings);
        assert!(repaired_metrics.lsd_hf < degraded_metrics.lsd_hf);
        assert!(repaired_metrics.snr_db > degraded_metrics.snr_db);
        assert!(repaired_metrics.snr_hf_db > degraded_metrics.snr_hf_db);
        assert_eq!(
            repaired_metrics.reference_hf_ratio,
            degraded_metrics.reference_hf_ratio
        );
        assert!(degraded_metrics.reference_hf_ratio.unwrap() > 0.99);
    }
}
