//! A route: DSP compensation — fold mid's high-frequency energy into the side channel.
//!
//! Strategy (per segment):
//! 1. STFT both M and S.
//! 2. For bins above fc, preserve the original S complex value and add only the energy missing
//!    from an M-derived target magnitude.
//! 3. Use M's phase plus smooth diffusion, flipped when needed to avoid cancelling original S.
//! 4. Crossfade with the original S in a transition band around fc.
//! 5. iSTFT back to time domain.

use crate::analysis::defects;
use crate::analysis::spectrum::{bin_of, power};
use crate::analysis::stft::{istft, stft, SpectrumFrame, StftConfig};
use crate::config::Config;
use crate::repair::common::phase_jitter;

/// Repair one segment of S given the corresponding M segment.
/// Returns the repaired S time-domain samples (same length as `s_seg`).
pub fn repair(m_seg: &[f32], s_seg: &[f32], cfg: &Config, sample_rate: u32) -> Vec<f32> {
    repair_with_progress(m_seg, s_seg, cfg, sample_rate, || {})
}

pub(crate) fn repair_with_progress<F>(
    m_seg: &[f32],
    s_seg: &[f32],
    cfg: &Config,
    sample_rate: u32,
    mut on_frame: F,
) -> Vec<f32>
where
    F: FnMut(),
{
    let stft_cfg = StftConfig::new(cfg.n_fft, cfg.hop);
    let m_spec = stft(m_seg, &stft_cfg);
    let s_spec = stft(s_seg, &stft_cfg);
    let n_bins = cfg.n_fft / 2 + 1;
    let defect_map = defects::analyze(&m_spec, &s_spec, cfg.scan_start_hz, cfg.n_fft, sample_rate);

    // Estimate the segment anchor plus a smoothed per-frame S/M ratio below the cutoff. The
    // frame-local target lets a short HF hole receive more fill without imposing one static gain
    // on the complete segment.
    let midband_lo = bin_of(
        (cfg.scan_start_hz as f32 * 0.60).max(500.0),
        cfg.n_fft,
        sample_rate,
    );
    let midband_hi = bin_of(
        (cfg.scan_start_hz as f32 - 500.0).max(1_000.0),
        cfg.n_fft,
        sample_rate,
    )
    .max(midband_lo + 1)
    .min(n_bins);
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
    let frame_gains = estimate_frame_gains(&m_spec, &s_spec, midband_lo, midband_hi, gain);

    // Reconstruct modified S spectra.
    let frames = m_spec.len().min(s_spec.len());
    let mut new_spec: Vec<SpectrumFrame> = Vec::with_capacity(frames);
    let jitter_rad = cfg.dsp_phase_degrees.to_radians();

    for f in 0..frames {
        let mut sf = SpectrumFrame::new(n_bins);
        let synthesis_strength = (cfg.dsp_strength * 0.5).clamp(0.0, 1.0);
        for b in 0..n_bins {
            let mask = defect_map.masks[f][b] * synthesis_strength;
            let m_mag = m_spec[f].mag(b);
            let m_phase = m_spec[f].phase(b);
            let s_re_orig = s_spec[f].re(b);
            let s_im_orig = s_spec[f].im(b);

            let frequency_hz = b as f32 * sample_rate as f32 / cfg.n_fft as f32;
            let octave = (frequency_hz / cfg.scan_start_hz as f32).max(1.0).log2();
            let rolloff = 10.0f32.powf(-0.5 * cfg.bandwidth_rolloff_db_per_octave * octave / 20.0);
            let target_mag = m_mag * frame_gains[f] * rolloff;
            let synthesis_phase = m_phase + phase_jitter(b, jitter_rad);
            let (re, im) =
                fill_energy_deficit(s_re_orig, s_im_orig, target_mag, synthesis_phase, mask);
            sf.cplx[2 * b] = re;
            sf.cplx[2 * b + 1] = im;
        }
        new_spec.push(sf);
        on_frame();
    }

    istft(&new_spec, &stft_cfg, s_seg.len())
}

pub(crate) fn fill_energy_deficit(
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

fn estimate_frame_gains(
    m_spec: &[SpectrumFrame],
    s_spec: &[SpectrumFrame],
    lo: usize,
    hi: usize,
    anchor: f32,
) -> Vec<f32> {
    let frames = m_spec.len().min(s_spec.len());
    let local = (0..frames)
        .map(|frame| {
            let mut mid_energy = 0.0f64;
            let mut side_energy = 0.0f64;
            for bin in lo..hi {
                mid_energy += m_spec[frame].mag(bin).powi(2) as f64;
                side_energy += s_spec[frame].mag(bin).powi(2) as f64;
            }
            if mid_energy > 1e-12 && side_energy > 1e-14 {
                (side_energy / mid_energy).sqrt().clamp(0.05, 2.0) as f32
            } else {
                anchor
            }
        })
        .collect::<Vec<_>>();
    let local_mean = if local.is_empty() {
        anchor
    } else {
        local.iter().sum::<f32>() / local.len() as f32
    };
    let normalization = if local_mean > 1e-8 {
        anchor / local_mean
    } else {
        1.0
    };
    let normalized = local
        .into_iter()
        .map(|gain| (gain * normalization).clamp(anchor * 0.5, anchor * 2.0))
        .collect::<Vec<_>>();

    let mut smoothed = (0..frames)
        .map(|frame| {
            let start = frame.saturating_sub(4);
            let end = (frame + 5).min(frames);
            let mut neighborhood = normalized[start..end].to_vec();
            neighborhood.sort_by(f32::total_cmp);
            let median = neighborhood[neighborhood.len() / 2];
            (0.25 * median + 0.75 * anchor).clamp(0.05, 2.0)
        })
        .collect::<Vec<_>>();
    let maximum_frame_ratio = 10.0f32.powf(1.0 / 20.0);
    for frame in 1..frames {
        smoothed[frame] = smoothed[frame].clamp(
            smoothed[frame - 1] / maximum_frame_ratio,
            smoothed[frame - 1] * maximum_frame_ratio,
        );
    }
    for frame in (0..frames.saturating_sub(1)).rev() {
        smoothed[frame] = smoothed[frame].clamp(
            smoothed[frame + 1] / maximum_frame_ratio,
            smoothed[frame + 1] * maximum_frame_ratio,
        );
    }
    smoothed
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

    #[test]
    fn a_deeper_local_hole_receives_more_compensation() {
        let sample_rate = 48_000;
        let length = sample_rate as usize;
        let config = Config::default();
        let mid = (0..length)
            .map(|index| {
                let time = index as f32 / sample_rate as f32;
                0.2 * (2.0 * std::f32::consts::PI * 6_000.0 * time).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 10_000.0 * time).sin()
            })
            .collect::<Vec<_>>();
        let side = (0..length)
            .map(|index| {
                let time = index as f32 / sample_rate as f32;
                let intact = 0.1 * (2.0 * std::f32::consts::PI * 6_000.0 * time).sin();
                let high = if index < length / 2 {
                    0.1 * (2.0 * std::f32::consts::PI * 10_000.0 * time).sin()
                } else {
                    0.0
                };
                intact + high
            })
            .collect::<Vec<_>>();

        let repaired = repair(&mid, &side, &config, sample_rate);
        let delta_rms = |range: std::ops::Range<usize>| {
            let energy = range
                .clone()
                .map(|index| (repaired[index] - side[index]).powi(2))
                .sum::<f32>();
            (energy / range.len() as f32).sqrt()
        };
        let present = delta_rms(length / 8..3 * length / 8);
        let missing = delta_rms(5 * length / 8..7 * length / 8);
        assert!(
            missing > present * 4.0,
            "deeper hole should receive more fill: present={present}, missing={missing}"
        );
    }
}
