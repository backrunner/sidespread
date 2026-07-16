//! A route: DSP compensation — fold mid's high-frequency energy into the side channel.
//!
//! Strategy (per segment):
//! 1. STFT both M and S.
//! 2. For bins above fc, preserve the original S complex value and add only the energy missing
//!    from an M-derived target magnitude.
//! 3. Use M's phase plus smooth diffusion, flipped when needed to avoid cancelling original S.
//! 4. Crossfade with the original S in a transition band around fc.
//! 5. iSTFT back to time domain.

use crate::analysis::spectrum::{bin_of, power};
use crate::analysis::stft::{istft, stft, SpectrumFrame, StftConfig};
use crate::config::Config;
use crate::repair::common::{band_mask, phase_jitter};

/// Repair one segment of S given the corresponding M segment.
/// Returns the repaired S time-domain samples (same length as `s_seg`).
pub fn repair(m_seg: &[f32], s_seg: &[f32], cfg: &Config, sample_rate: u32) -> Vec<f32> {
    let stft_cfg = StftConfig::new(cfg.n_fft, cfg.hop);
    let m_spec = stft(m_seg, &stft_cfg);
    let s_spec = stft(s_seg, &stft_cfg);
    let n_bins = cfg.n_fft / 2 + 1;
    let fc_bin = bin_of(cfg.fc as f32, cfg.n_fft, sample_rate);
    let transition_bins = bin_of(500.0, cfg.n_fft, sample_rate).max(2);
    let lo_bin = fc_bin.saturating_sub(transition_bins);
    let hi_bin = (fc_bin + transition_bins).min(n_bins - 1);

    // Estimate per-bin gain from S's midband energy profile relative to M's highband.
    // Use the average S/M magnitude ratio in the midband just below fc as the scale.
    let midband_lo = bin_of((cfg.fc as f32 * 0.5).max(500.0), cfg.n_fft, sample_rate);
    let midband_hi = fc_bin;
    let mut mid_energy = 0.0f64;
    let mut side_energy = 0.0f64;
    for f in 0..m_spec.len().min(s_spec.len()) {
        let m_pow = power(&m_spec[f]);
        let s_pow = power(&s_spec[f]);
        for b in midband_lo..midband_hi {
            mid_energy += m_pow[b] as f64;
            side_energy += s_pow[b] as f64;
        }
    }
    let gain = if mid_energy > 1e-12 {
        (side_energy / mid_energy).sqrt().clamp(0.05, 2.0) as f32
    } else {
        0.5
    };

    // Reconstruct modified S spectra.
    let frames = m_spec.len().min(s_spec.len());
    let mut new_spec: Vec<SpectrumFrame> = Vec::with_capacity(frames);
    let jitter_rad = cfg.dsp_phase_degrees.to_radians();

    for f in 0..frames {
        let mut sf = SpectrumFrame::new(n_bins);
        for b in 0..n_bins {
            let mask = band_mask(b, lo_bin, hi_bin) * cfg.dsp_strength;
            let m_mag = m_spec[f].mag(b);
            let m_phase = m_spec[f].phase(b);
            let s_re_orig = s_spec[f].re(b);
            let s_im_orig = s_spec[f].im(b);

            let target_mag = m_mag * gain;
            let synthesis_phase = m_phase + phase_jitter(b, jitter_rad);
            let (re, im) =
                fill_energy_deficit(s_re_orig, s_im_orig, target_mag, synthesis_phase, mask);
            sf.cplx[2 * b] = re;
            sf.cplx[2 * b + 1] = im;
        }
        new_spec.push(sf);
    }

    istft(&new_spec, &stft_cfg, s_seg.len())
}

fn fill_energy_deficit(
    original_re: f32,
    original_im: f32,
    target_magnitude: f32,
    mut synthesis_phase: f32,
    mix: f32,
) -> (f32, f32) {
    let original_power = original_re * original_re + original_im * original_im;
    let target_power = target_magnitude * target_magnitude;
    if target_power <= original_power || mix <= 0.0 {
        return (original_re, original_im);
    }

    let mut unit_re = synthesis_phase.cos();
    let mut unit_im = synthesis_phase.sin();
    let mut projection = original_re * unit_re + original_im * unit_im;
    if projection < 0.0 {
        synthesis_phase += std::f32::consts::PI;
        unit_re = synthesis_phase.cos();
        unit_im = synthesis_phase.sin();
        projection = -projection;
    }
    let discriminant = (projection * projection + target_power - original_power).max(0.0);
    let added_magnitude = (-projection + discriminant.sqrt()).max(0.0) * mix;
    (
        original_re + added_magnitude * unit_re,
        original_im + added_magnitude * unit_im,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn repair_preserves_length() {
        let cfg = Config::default();
        let m: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.01).sin()).collect();
        let s: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.003).sin() * 0.1).collect();
        let out = repair(&m, &s, &cfg, 48000);
        assert_eq!(out.len(), s.len(), "repair must preserve segment length");
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn energy_fill_preserves_existing_complex_content() {
        let original = (0.3f32, -0.4f32);
        assert_eq!(
            fill_energy_deficit(original.0, original.1, 0.4, 0.7, 1.0),
            original
        );

        let repaired = fill_energy_deficit(original.0, original.1, 0.8, 2.9, 1.0);
        let repaired_magnitude = (repaired.0 * repaired.0 + repaired.1 * repaired.1).sqrt();
        let delta = (repaired.0 - original.0, repaired.1 - original.1);
        assert!((repaired_magnitude - 0.8).abs() < 1e-5);
        assert!(delta.0 * original.0 + delta.1 * original.1 >= -1e-6);
    }
}
