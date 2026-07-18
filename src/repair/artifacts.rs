//! Optional repair for high-frequency smearing, harmonic bleeding, and Side phase drift.

use crate::analysis::spectrum::bin_of;
use crate::analysis::stft::{istft, stft, SpectrumFrame, StftConfig};
use crate::config::Config;
use crate::repair::common::{band_mask, smoothstep};
use crate::repair::safety::{self, ArtifactKind, AudioPair};
use crate::terminal::{self, TaskProgress, Tone};
use std::collections::VecDeque;

/// Apply the artifact processors in a conservative fixed order.
pub struct ArtifactOutcome {
    pub mid: Vec<f32>,
    pub side: Vec<f32>,
    pub smearing_mix: f32,
    pub bleeding_mix: f32,
    pub phase_mix: f32,
}

pub fn process(mid: &[f32], side: &[f32], config: &Config, sample_rate: u32) -> ArtifactOutcome {
    process_impl(mid, side, config, sample_rate, false)
}

pub fn process_with_progress(
    mid: &[f32],
    side: &[f32],
    config: &Config,
    sample_rate: u32,
) -> ArtifactOutcome {
    process_impl(mid, side, config, sample_rate, true)
}

fn process_impl(
    mid: &[f32],
    side: &[f32],
    config: &Config,
    sample_rate: u32,
    show_progress: bool,
) -> ArtifactOutcome {
    let mut current_mid = mid.to_vec();
    let mut current_side = side.to_vec();
    let mut smearing_mix = 0.0;
    let mut bleeding_mix = 0.0;
    let mut phase_mix = 0.0;

    if config.repair_bleeding && config.bleeding_strength > 0.0 {
        let (candidate_mid, candidate_side) = harmonic_debleed_pair_impl(
            &current_mid,
            &current_side,
            config,
            sample_rate,
            config.bleeding_strength,
            show_progress,
        );
        let guarded = safety::guard_with_progress(
            AudioPair {
                mid: &current_mid,
                side: &current_side,
            },
            AudioPair {
                mid: &candidate_mid,
                side: &candidate_side,
            },
            ArtifactKind::HarmonicBleeding,
            config.bleeding_strength,
            config,
            sample_rate,
            show_progress,
        );
        current_mid = guarded.mid;
        current_side = guarded.side;
        bleeding_mix = guarded.retained_mix;
    } else if show_progress {
        terminal::status("HARMONIC DEBLEED", "disabled", Tone::Muted);
    }
    if config.repair_smearing && config.smearing_strength > 0.0 {
        let (candidate_mid, candidate_side) = transient_sharpen_pair_impl(
            &current_mid,
            &current_side,
            config,
            sample_rate,
            config.smearing_strength,
            show_progress,
        );
        let guarded = safety::guard_with_progress(
            AudioPair {
                mid: &current_mid,
                side: &current_side,
            },
            AudioPair {
                mid: &candidate_mid,
                side: &candidate_side,
            },
            ArtifactKind::Smearing,
            config.smearing_strength,
            config,
            sample_rate,
            show_progress,
        );
        current_mid = guarded.mid;
        current_side = guarded.side;
        smearing_mix = guarded.retained_mix;
    } else if show_progress {
        terminal::status("HF SMEARING", "disabled", Tone::Muted);
    }
    if config.stabilize_phase && config.phase_strength > 0.0 {
        let candidate_side = stabilize_side_phase_impl(
            &current_mid,
            &current_side,
            config,
            sample_rate,
            config.phase_strength,
            show_progress,
        );
        let guarded = safety::guard_with_progress(
            AudioPair {
                mid: &current_mid,
                side: &current_side,
            },
            AudioPair {
                mid: &current_mid,
                side: &candidate_side,
            },
            ArtifactKind::PhaseIncoherence,
            config.phase_strength,
            config,
            sample_rate,
            show_progress,
        );
        current_mid = guarded.mid;
        current_side = guarded.side;
        phase_mix = guarded.retained_mix;
    } else if show_progress {
        terminal::status("PHASE STABILIZE", "disabled", Tone::Muted);
    }

    ArtifactOutcome {
        mid: current_mid,
        side: current_side,
        smearing_mix,
        bleeding_mix,
        phase_mix,
    }
}

#[cfg(test)]
fn transient_sharpen_pair(
    mid: &[f32],
    side: &[f32],
    config: &Config,
    sample_rate: u32,
    strength: f32,
) -> (Vec<f32>, Vec<f32>) {
    transient_sharpen_pair_impl(mid, side, config, sample_rate, strength, false)
}

fn transient_sharpen_pair_impl(
    mid: &[f32],
    side: &[f32],
    config: &Config,
    sample_rate: u32,
    strength: f32,
    show_progress: bool,
) -> (Vec<f32>, Vec<f32>) {
    if strength <= 0.0 || mid.is_empty() || side.is_empty() {
        return (mid.to_vec(), side.to_vec());
    }

    let stft_config = detail_stft_config(config);
    let expected_frames = stft_config.num_frames(mid.len().min(side.len()));
    let progress = TaskProgress::new_if(
        "HF SMEARING",
        expected_frames.saturating_add(4),
        "steps",
        show_progress,
    );
    let mid_spec = stft(mid, &stft_config);
    progress.advance(1);
    let side_spec = stft(side, &stft_config);
    progress.advance(1);
    let frames = mid_spec.len().min(side_spec.len());
    let n_bins = stft_config.n_fft / 2 + 1;
    let start = bin_of(
        (config.fc as f32).max(10_000.0),
        stft_config.n_fft,
        sample_rate,
    )
    .min(n_bins);
    let full = (start + bin_of(1_000.0, stft_config.n_fft, sample_rate)).min(n_bins - 1);
    let combined = combined_magnitudes(&mid_spec, &side_spec, frames, n_bins);
    let temporal_radius = 4usize;
    let spectral_radius = bin_of(220.0, stft_config.n_fft, sample_rate).max(3);
    let mut mid_out = Vec::with_capacity(frames);
    let mut side_out = Vec::with_capacity(frames);

    for frame in 0..frames {
        let previous = frame.saturating_sub(1);
        let next = (frame + 1).min(frames.saturating_sub(1));
        let mut detail_sum = 0.0f64;
        let mut magnitude_sum = 0.0f64;
        let mut percussive_sum = 0.0f64;
        for bin in start..n_bins {
            let reference = 0.7 * combined[previous][bin] + 0.3 * combined[next][bin];
            let detail = (combined[frame][bin] - reference).max(0.0);
            let harmonic = temporal_median(&combined, frame, bin, temporal_radius);
            let percussive = spectral_median(&combined[frame], bin, spectral_radius);
            let percussive_mask = soft_component_mask(percussive, harmonic);
            detail_sum += detail as f64;
            magnitude_sum += combined[frame][bin] as f64;
            percussive_sum += (percussive_mask * combined[frame][bin]) as f64;
        }
        let broadband_detail = if magnitude_sum > 1e-12 {
            (detail_sum / magnitude_sum) as f32
        } else {
            0.0
        };
        let percussive_share = if magnitude_sum > 1e-12 {
            (percussive_sum / magnitude_sum) as f32
        } else {
            0.0
        };
        let onset_gate = smoothstep((broadband_detail - 0.015) / 0.20)
            * smoothstep((percussive_share - 0.15) / 0.55);
        let mut m_frame = SpectrumFrame::new(n_bins);
        let mut s_frame = SpectrumFrame::new(n_bins);
        for bin in 0..n_bins {
            let gain = if bin >= start && onset_gate > 0.0 {
                let reference = 0.7 * combined[previous][bin] + 0.3 * combined[next][bin];
                let detail = (combined[frame][bin] - reference).max(0.0);
                let detail_ratio = detail / combined[frame][bin].max(1e-8);
                let harmonic = temporal_median(&combined, frame, bin, temporal_radius);
                let percussive = spectral_median(&combined[frame], bin, spectral_radius);
                let percussive_mask = soft_component_mask(percussive, harmonic);
                let frequency_mask = band_mask(bin, start, full);
                1.0 + 2.0
                    * strength
                    * onset_gate
                    * detail_ratio.min(1.0)
                    * percussive_mask.sqrt()
                    * frequency_mask
            } else {
                1.0
            };
            copy_scaled(&mid_spec[frame], &mut m_frame, bin, gain);
            copy_scaled(&side_spec[frame], &mut s_frame, bin, gain);
        }
        mid_out.push(m_frame);
        side_out.push(s_frame);
        progress.advance(1);
    }

    let mid = istft(&mid_out, &stft_config, mid.len());
    progress.advance(1);
    let side = istft(&side_out, &stft_config, side.len());
    progress.advance(1);
    progress.finish();
    (mid, side)
}

#[cfg(test)]
fn harmonic_debleed_pair(
    mid: &[f32],
    side: &[f32],
    config: &Config,
    sample_rate: u32,
    strength: f32,
) -> (Vec<f32>, Vec<f32>) {
    harmonic_debleed_pair_impl(mid, side, config, sample_rate, strength, false)
}

fn harmonic_debleed_pair_impl(
    mid: &[f32],
    side: &[f32],
    config: &Config,
    sample_rate: u32,
    strength: f32,
    show_progress: bool,
) -> (Vec<f32>, Vec<f32>) {
    if strength <= 0.0 || mid.is_empty() || side.is_empty() {
        return (mid.to_vec(), side.to_vec());
    }

    let stft_config = StftConfig::new(config.n_fft, config.hop);
    let expected_frames = stft_config.num_frames(mid.len().min(side.len()));
    let progress = TaskProgress::new_if(
        "HARMONIC DEBLEED",
        expected_frames.saturating_add(4),
        "steps",
        show_progress,
    );
    let mid_spec = stft(mid, &stft_config);
    progress.advance(1);
    let side_spec = stft(side, &stft_config);
    progress.advance(1);
    let frames = mid_spec.len().min(side_spec.len());
    let n_bins = config.n_fft / 2 + 1;
    let start = bin_of(config.fc as f32, config.n_fft, sample_rate).min(n_bins);
    let full = (start + bin_of(500.0, config.n_fft, sample_rate)).min(n_bins - 1);
    let peak_radius = bin_of(600.0, config.n_fft, sample_rate).max(2);
    let combined = combined_magnitudes(&mid_spec, &side_spec, frames, n_bins);
    let mut previous_mask = vec![1.0f32; n_bins];
    let mut mid_out = Vec::with_capacity(frames);
    let mut side_out = Vec::with_capacity(frames);

    for frame in 0..frames {
        let high_band = &combined[frame][start..n_bins];
        let arithmetic_power = high_band
            .iter()
            .map(|magnitude| magnitude * magnitude)
            .sum::<f32>()
            / high_band.len().max(1) as f32;
        let geometric_power = if arithmetic_power > 1e-14 {
            (high_band
                .iter()
                .map(|magnitude| (magnitude * magnitude + 1e-14).ln())
                .sum::<f32>()
                / high_band.len().max(1) as f32)
                .exp()
        } else {
            arithmetic_power
        };
        let flatness = if arithmetic_power > 1e-14 {
            geometric_power / arithmetic_power
        } else {
            1.0
        };
        let tonality = smoothstep((0.55 - flatness) / 0.45);
        let noise_floor = median(high_band.iter().copied());
        let local_peaks = sliding_max(&combined[frame], peak_radius);
        let mut raw_mask = vec![1.0f32; n_bins];
        let mut m_frame = SpectrumFrame::new(n_bins);
        let mut s_frame = SpectrumFrame::new(n_bins);

        for bin in start..n_bins {
            let peak_ratio = combined[frame][bin] / local_peaks[bin].max(1e-8);
            let valley = smoothstep((0.35 - peak_ratio) / 0.30);
            let frequency_mask = band_mask(bin, start, full);
            let power = combined[frame][bin].powi(2);
            let noise_power = (noise_floor * 1.5).powi(2);
            let wiener = if power + noise_power > 1e-14 {
                power / (power + noise_power)
            } else {
                1.0
            };
            let suppression = strength * tonality * valley * frequency_mask;
            let wiener_gain = wiener.powf(1.0 + 2.0 * strength);
            let maximum_attenuation = 10.0f32.powf(-18.0 * strength / 20.0);
            raw_mask[bin] = (1.0 - suppression * (1.0 - wiener_gain)).max(maximum_attenuation);
        }
        for bin in 0..n_bins {
            let gain = if frame == 0 {
                raw_mask[bin]
            } else {
                0.65 * previous_mask[bin] + 0.35 * raw_mask[bin]
            };
            copy_scaled(&mid_spec[frame], &mut m_frame, bin, gain);
            copy_scaled(&side_spec[frame], &mut s_frame, bin, gain);
        }
        previous_mask = raw_mask;
        mid_out.push(m_frame);
        side_out.push(s_frame);
        progress.advance(1);
    }

    let mid = istft(&mid_out, &stft_config, mid.len());
    progress.advance(1);
    let side = istft(&side_out, &stft_config, side.len());
    progress.advance(1);
    progress.finish();
    (mid, side)
}

#[cfg(test)]
fn stabilize_side_phase(
    mid: &[f32],
    side: &[f32],
    config: &Config,
    sample_rate: u32,
    strength: f32,
) -> Vec<f32> {
    stabilize_side_phase_impl(mid, side, config, sample_rate, strength, false)
}

fn stabilize_side_phase_impl(
    mid: &[f32],
    side: &[f32],
    config: &Config,
    sample_rate: u32,
    strength: f32,
    show_progress: bool,
) -> Vec<f32> {
    if strength <= 0.0 || mid.is_empty() || side.is_empty() {
        return side.to_vec();
    }

    let stft_config = StftConfig::new(config.n_fft, config.hop);
    let expected_frames = stft_config.num_frames(mid.len().min(side.len()));
    let progress = TaskProgress::new_if(
        "PHASE STABILIZE",
        expected_frames.saturating_add(3),
        "steps",
        show_progress,
    );
    let mid_spec = stft(mid, &stft_config);
    progress.advance(1);
    let side_spec = stft(side, &stft_config);
    progress.advance(1);
    let frames = mid_spec.len().min(side_spec.len());
    let n_bins = config.n_fft / 2 + 1;
    let start = bin_of(config.fc as f32, config.n_fft, sample_rate).min(n_bins);
    let transition_start = start.saturating_sub(bin_of(500.0, config.n_fft, sample_rate));
    let mut output = Vec::with_capacity(frames);

    for frame in 0..frames {
        let mut repaired = SpectrumFrame::new(n_bins);
        let window_start = frame.saturating_sub(2);
        let window_end = (frame + 3).min(frames);
        for bin in 0..n_bins {
            if bin < transition_start {
                copy_scaled(&side_spec[frame], &mut repaired, bin, 1.0);
                continue;
            }

            let mut relative_re = 0.0f64;
            let mut relative_im = 0.0f64;
            let mut weight_sum = 0.0f64;
            for neighbor in window_start..window_end {
                let mr = mid_spec[neighbor].re(bin) as f64;
                let mi = mid_spec[neighbor].im(bin) as f64;
                let sr = side_spec[neighbor].re(bin) as f64;
                let si = side_spec[neighbor].im(bin) as f64;
                let rel_re = mr * sr + mi * si;
                let rel_im = mr * si - mi * sr;
                relative_re += rel_re;
                relative_im += rel_im;
                weight_sum += (rel_re * rel_re + rel_im * rel_im).sqrt();
            }
            let mask = band_mask(bin, transition_start, start);
            let side_magnitude = side_spec[frame].mag(bin);
            let mid_magnitude = mid_spec[frame].mag(bin);
            if weight_sum <= 1e-12 || side_magnitude <= 1e-8 || mid_magnitude <= 1e-8 {
                copy_scaled(&side_spec[frame], &mut repaired, bin, 1.0);
                continue;
            }

            let coherence = ((relative_re * relative_re + relative_im * relative_im).sqrt()
                / weight_sum)
                .clamp(0.0, 1.0) as f32;
            let instability = 1.0 - coherence;
            let target_relative_phase = relative_im.atan2(relative_re) as f32;
            let target_phase = mid_spec[frame].phase(bin) + target_relative_phase;
            let current_phase = side_spec[frame].phase(bin);
            let difference = wrap_phase(target_phase - current_phase);
            let phase_pull = 0.75 * strength * instability * mask;
            let output_phase = current_phase + difference * phase_pull;
            let attenuation_db = 6.0 * strength * instability * instability * mask;
            let output_magnitude = side_magnitude * 10.0f32.powf(-attenuation_db / 20.0);
            repaired.cplx[2 * bin] = output_magnitude * output_phase.cos();
            repaired.cplx[2 * bin + 1] = output_magnitude * output_phase.sin();
        }
        output.push(repaired);
        progress.advance(1);
    }

    let output = istft(&output, &stft_config, side.len());
    progress.advance(1);
    progress.finish();
    output
}

fn detail_stft_config(config: &Config) -> StftConfig {
    let n_fft = config.n_fft.clamp(4, 2048);
    StftConfig::new(n_fft, (n_fft / 4).max(1))
}

fn combined_magnitudes(
    mid: &[SpectrumFrame],
    side: &[SpectrumFrame],
    frames: usize,
    n_bins: usize,
) -> Vec<Vec<f32>> {
    (0..frames)
        .map(|frame| {
            (0..n_bins)
                .map(|bin| mid[frame].mag(bin).hypot(side[frame].mag(bin)))
                .collect()
        })
        .collect()
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

fn temporal_median(magnitudes: &[Vec<f32>], frame: usize, bin: usize, radius: usize) -> f32 {
    let start = frame.saturating_sub(radius);
    let end = (frame + radius + 1).min(magnitudes.len());
    median(magnitudes[start..end].iter().map(|values| values[bin]))
}

fn spectral_median(magnitudes: &[f32], bin: usize, radius: usize) -> f32 {
    let start = bin.saturating_sub(radius);
    let end = (bin + radius + 1).min(magnitudes.len());
    median(magnitudes[start..end].iter().copied())
}

fn median<I>(values: I) -> f32
where
    I: IntoIterator<Item = f32>,
{
    let mut values = values.into_iter().collect::<Vec<_>>();
    if values.is_empty() {
        return 0.0;
    }
    let middle = values.len() / 2;
    let (_, value, _) = values.select_nth_unstable_by(middle, f32::total_cmp);
    *value
}

fn soft_component_mask(component: f32, competing: f32) -> f32 {
    let component = component * component;
    let competing = competing * competing;
    component / (component + competing + 1e-12)
}

fn copy_scaled(source: &SpectrumFrame, target: &mut SpectrumFrame, bin: usize, gain: f32) {
    target.cplx[2 * bin] = source.re(bin) * gain;
    target.cplx[2 * bin + 1] = source.im(bin) * gain;
}

fn wrap_phase(value: f32) -> f32 {
    (value + std::f32::consts::PI).rem_euclid(2.0 * std::f32::consts::PI) - std::f32::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{lr_to_ms, read_wav};
    use crate::repair::safety::{
        calibration_diagnostics, calibration_fidelity, ArtifactKind, AudioPair,
    };

    fn sine(frequency: f32, index: usize, sample_rate: u32) -> f32 {
        (2.0 * std::f32::consts::PI * frequency * index as f32 / sample_rate as f32).sin()
    }

    fn rms(signal: &[f32]) -> f32 {
        (signal.iter().map(|sample| sample * sample).sum::<f32>() / signal.len().max(1) as f32)
            .sqrt()
    }

    fn ratio_db(after: f64, before: f64) -> f64 {
        10.0 * (after / before.max(1e-20)).max(1e-20).log10()
    }

    fn improvement_percent(before: f64, after: f64, increase_is_better: bool) -> f64 {
        if before <= 1e-20 {
            return 0.0;
        }
        let change = (after / before - 1.0) * 100.0;
        if increase_is_better {
            change
        } else {
            -change
        }
    }

    #[test]
    #[ignore = "manual signal-domain calibration; set SIDESPREAD_CALIBRATION_WAV"]
    fn calibrate_artifact_strengths_on_real_audio() {
        let path = std::env::var("SIDESPREAD_CALIBRATION_WAV")
            .expect("SIDESPREAD_CALIBRATION_WAV must point to a stereo WAV");
        let buffer = read_wav(path).unwrap();
        let sample_rate = buffer.sample_rate;
        let (mid, side) = lr_to_ms(&buffer).unwrap();
        let window_len = (sample_rate as usize * 5).min(mid.len());
        let window_starts = [0.2f32, 0.5, 0.8].map(|position| {
            ((mid.len().saturating_sub(window_len)) as f32 * position).round() as usize
        });
        let strengths = [0.20f32, 0.35, 0.50, 0.65, 0.80];
        let kinds = [
            ("smearing", ArtifactKind::Smearing),
            ("bleeding", ArtifactKind::HarmonicBleeding),
            ("phase", ArtifactKind::PhaseIncoherence),
        ];

        println!(
            "CALIBRATION stage,strength,accepted,mean_mix,mean_effect_pct,min_protected_snr_db,min_peak_retention_db,mean_hf_energy_db,mean_side_hf_db,mean_iccc_delta"
        );
        for (label, kind) in kinds {
            for strength in strengths {
                let mut accepted = 0usize;
                let mut mix_sum = 0.0f64;
                let mut effect_sum = 0.0f64;
                let mut protected_snr = f32::INFINITY;
                let mut peak_retention = f32::INFINITY;
                let mut high_energy_db = 0.0f64;
                let mut side_energy_db = 0.0f64;
                let mut iccc_delta = 0.0f64;

                for start in window_starts {
                    let end = start + window_len;
                    let source_mid = &mid[start..end];
                    let source_side = &side[start..end];
                    let mut config = Config::default();
                    match kind {
                        ArtifactKind::Smearing => {
                            config.repair_smearing = true;
                            config.smearing_strength = strength;
                        }
                        ArtifactKind::HarmonicBleeding => {
                            config.repair_bleeding = true;
                            config.bleeding_strength = strength;
                        }
                        ArtifactKind::PhaseIncoherence => {
                            config.stabilize_phase = true;
                            config.phase_strength = strength;
                        }
                    }
                    let output = process(source_mid, source_side, &config, sample_rate);
                    let retained_mix = match kind {
                        ArtifactKind::Smearing => output.smearing_mix,
                        ArtifactKind::HarmonicBleeding => output.bleeding_mix,
                        ArtifactKind::PhaseIncoherence => output.phase_mix,
                    };
                    accepted += usize::from(retained_mix > 0.0);
                    mix_sum += retained_mix as f64;

                    let before = calibration_diagnostics(
                        source_mid,
                        source_side,
                        &config,
                        sample_rate,
                        kind,
                    );
                    let after = calibration_diagnostics(
                        &output.mid,
                        &output.side,
                        &config,
                        sample_rate,
                        kind,
                    );
                    let effect = match kind {
                        ArtifactKind::Smearing => {
                            improvement_percent(before.spectral_flux, after.spectral_flux, true)
                        }
                        ArtifactKind::HarmonicBleeding => improvement_percent(
                            before.inter_peak_ratio,
                            after.inter_peak_ratio,
                            false,
                        ),
                        ArtifactKind::PhaseIncoherence => {
                            improvement_percent(before.phase_jitter, after.phase_jitter, false)
                        }
                    };
                    effect_sum += effect;
                    high_energy_db += ratio_db(after.high_energy, before.high_energy);
                    side_energy_db += ratio_db(after.side_high_energy, before.side_high_energy);
                    iccc_delta += after.interchannel_correlation - before.interchannel_correlation;

                    let fidelity = calibration_fidelity(
                        AudioPair {
                            mid: source_mid,
                            side: source_side,
                        },
                        AudioPair {
                            mid: &output.mid,
                            side: &output.side,
                        },
                        &config,
                        sample_rate,
                        kind,
                    );
                    protected_snr = protected_snr.min(fidelity.protected_snr_db);
                    peak_retention = peak_retention.min(fidelity.peak_retention_db);
                }

                let count = window_starts.len() as f64;
                println!(
                    "CALIBRATION {label},{strength:.2},{accepted}/{},{:.3},{:.3},{protected_snr:.2},{peak_retention:.3},{:.3},{:.3},{:.6}",
                    window_starts.len(),
                    mix_sum / count,
                    effect_sum / count,
                    high_energy_db / count,
                    side_energy_db / count,
                    iccc_delta / count,
                );
            }
        }
    }

    fn harmonic_valley_ratio(signal: &[f32], sample_rate: u32, config: &Config) -> f64 {
        let stft_config = StftConfig::new(config.n_fft, config.hop);
        let spectrum = stft(signal, &stft_config);
        let start = bin_of(config.fc as f32, config.n_fft, sample_rate);
        let harmonic_bins = [9_000.0, 10_000.0, 11_000.0, 12_000.0, 13_000.0]
            .map(|frequency| bin_of(frequency, config.n_fft, sample_rate));
        let mut peaks = 0.0f64;
        let mut valleys = 0.0f64;
        for frame in spectrum {
            for bin in start..frame.n_bins {
                let power = frame.mag(bin).powi(2) as f64;
                if harmonic_bins
                    .iter()
                    .any(|harmonic| bin.abs_diff(*harmonic) <= 3)
                {
                    peaks += power;
                } else {
                    valleys += power;
                }
            }
        }
        valleys / peaks.max(1e-12)
    }

    fn relative_phase_jitter(mid: &[f32], side: &[f32], sample_rate: u32, config: &Config) -> f32 {
        let stft_config = StftConfig::new(config.n_fft, config.hop);
        let mid = stft(mid, &stft_config);
        let side = stft(side, &stft_config);
        let bin = bin_of(10_000.0, config.n_fft, sample_rate);
        let phases = mid
            .iter()
            .zip(&side)
            .filter(|(mid, side)| mid.mag(bin) > 1e-3 && side.mag(bin) > 1e-3)
            .map(|(mid, side)| wrap_phase(side.phase(bin) - mid.phase(bin)))
            .collect::<Vec<_>>();
        if phases.len() < 2 {
            return 0.0;
        }
        phases
            .windows(2)
            .map(|pair| wrap_phase(pair[1] - pair[0]).abs())
            .sum::<f32>()
            / (phases.len() - 1) as f32
    }

    #[test]
    fn default_processing_is_an_exact_bypass() {
        let mid = vec![0.1, -0.2, 0.3];
        let side = vec![-0.05, 0.08, -0.02];
        let output = process(&mid, &side, &Config::default(), 48_000);
        assert_eq!((output.mid, output.side), (mid, side));
    }

    #[test]
    fn zero_strength_processors_are_exact_bypasses() {
        let config = Config {
            repair_smearing: true,
            smearing_strength: 0.0,
            repair_bleeding: true,
            bleeding_strength: 0.0,
            stabilize_phase: true,
            phase_strength: 0.0,
            ..Config::default()
        };
        let mid = vec![0.1, -0.2, 0.3];
        let side = vec![-0.05, 0.08, -0.02];
        let output = process(&mid, &side, &config, 48_000);
        assert_eq!((output.mid, output.side), (mid, side));
    }

    #[test]
    fn sliding_max_tracks_both_sides_of_the_window() {
        assert_eq!(
            sliding_max(&[1.0, 4.0, 2.0, 3.0], 1),
            vec![4.0, 4.0, 4.0, 3.0]
        );
    }

    #[test]
    fn soft_component_masks_are_bounded() {
        assert!(soft_component_mask(2.0, 0.1) > 0.99);
        assert!(soft_component_mask(0.1, 2.0) < 0.01);
    }

    #[test]
    fn smearing_repair_enhances_broadband_high_frequency_onsets() {
        let sample_rate = 48_000;
        let length = sample_rate as usize;
        let mut mid = vec![0.0f32; length];
        let mut side = vec![0.0f32; length];
        let start = 20_000usize;
        let end = 22_000usize;
        for index in start..end {
            let envelope = ((index - start) as f32 / 200.0).min(1.0)
                * ((end - index) as f32 / 1_000.0).min(1.0);
            let transient = (sine(11_000.0, index, sample_rate)
                + sine(13_000.0, index, sample_rate)
                + sine(15_000.0, index, sample_rate))
                * 0.06
                * envelope;
            mid[index] = transient;
            side[index] = transient * 0.35;
        }
        let config = Config::default();
        let (half_mid, _) = transient_sharpen_pair(&mid, &side, &config, sample_rate, 0.5);
        let (full_mid, _) = transient_sharpen_pair(&mid, &side, &config, sample_rate, 1.0);
        let input_onset = rms(&mid[start..start + 1_000]);
        let half_onset = rms(&half_mid[start..start + 1_000]);
        let full_onset = rms(&full_mid[start..start + 1_000]);
        assert!(half_onset > input_onset * 1.01);
        assert!(full_onset > half_onset * 1.01);
        assert!(rms(&full_mid[..start - 2_000]) < 1e-4);
    }

    #[test]
    fn harmonic_debleed_reduces_valleys_and_preserves_peaks() {
        let sample_rate = 48_000;
        let length = sample_rate as usize;
        let mut state = 91u32;
        let mid = (0..length)
            .map(|index| {
                let harmonics = [9_000.0, 10_000.0, 11_000.0, 12_000.0, 13_000.0]
                    .iter()
                    .map(|frequency| sine(*frequency, index, sample_rate) * 0.04)
                    .sum::<f32>();
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let residue = ((state >> 16) as f32 / 32_768.0 - 1.0) * 0.004;
                harmonics + residue
            })
            .collect::<Vec<_>>();
        let side = mid.iter().map(|sample| sample * 0.3).collect::<Vec<_>>();
        let config = Config::default();
        let (half_mid, _) = harmonic_debleed_pair(&mid, &side, &config, sample_rate, 0.5);
        let (full_mid, _) = harmonic_debleed_pair(&mid, &side, &config, sample_rate, 1.0);
        let input_ratio = harmonic_valley_ratio(&mid, sample_rate, &config);
        let half_ratio = harmonic_valley_ratio(&half_mid, sample_rate, &config);
        let full_ratio = harmonic_valley_ratio(&full_mid, sample_rate, &config);
        assert!(
            half_ratio < input_ratio * 0.9,
            "input={input_ratio}, half={half_ratio}"
        );
        assert!(
            full_ratio < half_ratio * 0.9,
            "half={half_ratio}, full={full_ratio}"
        );
        assert!(
            rms(&full_mid) > rms(&mid) * 0.75,
            "harmonic peaks were over-suppressed"
        );
    }

    #[test]
    fn phase_stabilization_reduces_relative_jitter_without_changing_mid() {
        let sample_rate = 48_000;
        let length = sample_rate as usize;
        let phases = [0.0f32, 2.2, -1.7, 1.4, -2.5, 0.8];
        let mid = (0..length)
            .map(|index| sine(10_000.0, index, sample_rate) * 0.15)
            .collect::<Vec<_>>();
        let side = (0..length)
            .map(|index| {
                let block = index * phases.len() / length;
                let phase = phases[block.min(phases.len() - 1)];
                (2.0 * std::f32::consts::PI * 10_000.0 * index as f32 / sample_rate as f32 + phase)
                    .sin()
                    * 0.08
            })
            .collect::<Vec<_>>();
        let config = Config::default();
        let half = stabilize_side_phase(&mid, &side, &config, sample_rate, 0.5);
        let full = stabilize_side_phase(&mid, &side, &config, sample_rate, 1.0);
        let before = relative_phase_jitter(&mid, &side, sample_rate, &config);
        let half_jitter = relative_phase_jitter(&mid, &half, sample_rate, &config);
        let full_jitter = relative_phase_jitter(&mid, &full, sample_rate, &config);
        assert!(half_jitter < before, "before={before}, half={half_jitter}");
        assert!(
            full_jitter < half_jitter,
            "half={half_jitter}, full={full_jitter}"
        );
        assert!(
            rms(&full) > rms(&side) * 0.45,
            "phase repair removed too much Side"
        );
    }

    #[test]
    fn all_artifact_processors_support_44100_hz() {
        let sample_rate = 44_100;
        let length = sample_rate as usize;
        let mid = (0..length)
            .map(|index| {
                sine(9_500.0, index, sample_rate) * 0.12 + sine(12_500.0, index, sample_rate) * 0.04
            })
            .collect::<Vec<_>>();
        let side = (0..length)
            .map(|index| {
                let phase = if index < length / 2 { 0.6 } else { -1.0 };
                (2.0 * std::f32::consts::PI * 10_500.0 * index as f32 / sample_rate as f32 + phase)
                    .sin()
                    * 0.05
            })
            .collect::<Vec<_>>();
        let config = Config {
            repair_smearing: true,
            repair_bleeding: true,
            stabilize_phase: true,
            ..Config::default()
        };

        let output = process(&mid, &side, &config, sample_rate);
        assert_eq!(output.mid.len(), length);
        assert_eq!(output.side.len(), length);
        assert!(output.mid.iter().all(|sample| sample.is_finite()));
        assert!(output.side.iter().all(|sample| sample.is_finite()));
    }
}
