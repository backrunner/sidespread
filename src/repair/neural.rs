//! B route: targeted high-frequency repair with UniverSR ONNX.

use crate::analysis::defects;
use crate::analysis::spectrum::bin_of;
use crate::analysis::stft::{istft, stft, SpectrumFrame, StftConfig};
use crate::config::Config;
use crate::repair::universr::config::UniversrConfig;
use crate::repair::universr::frontend::Frontend;
use crate::repair::universr::istft::Backend;
use crate::repair::universr::merge::band_merge;
use crate::repair::universr::ode::OdeSolver;
use crate::repair::universr::onnx_session::Sessions;
use anyhow::{bail, Context, Result};
use rubato::{FftFixedInOut, Resampler};
use std::path::{Path, PathBuf};

pub struct NeuralState {
    sessions: Sessions,
    config: UniversrConfig,
    frontend: Frontend,
    backend: Backend,
    solver: OdeSolver,
    chunk_samples: usize,
    cutoff_hz: usize,
}

pub struct GuardedNeuralAudio {
    pub signal: Vec<f32>,
    pub minimum_retained_mix: f32,
    pub mean_retained_mix: f32,
}

impl NeuralState {
    pub fn from_config(sidespread_config: &Config) -> Result<Self> {
        let models_dir = models_directory(sidespread_config);
        Self::load(&models_dir, sidespread_config)
    }

    pub fn load(models_dir: &Path, sidespread_config: &Config) -> Result<Self> {
        let mut config = UniversrConfig::load(models_dir).context("loading UniverSR config")?;
        if let Some(model_path) = sidespread_config.model_path.as_ref().filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "onnx")
        }) {
            config.model_onnx = model_path.clone();
        }
        let expected_condition_bins = config.lr_bin_count(16);
        validate_model_cutoff(
            sidespread_config.fc,
            expected_condition_bins,
            config.n_fft,
            config.target_sr,
        )?;
        if sidespread_config.model_path.is_none() && !config.model_onnx.exists() {
            crate::model::ensure_default_available(&config.model_onnx)
                .context("installing the default UniverSR model")?;
        }
        let sessions = Sessions::load(&config.model_onnx).with_context(|| {
            format!(
                "loading UniverSR model {}; install the default model with \
                 `sidespread model download`",
                config.model_onnx.display()
            )
        })?;
        if sessions.hr_bins != config.hr_freq_bins {
            bail!(
                "ONNX model has {} generated bins, config expects {}",
                sessions.hr_bins,
                config.hr_freq_bins
            );
        }
        if sessions.condition_bins != expected_condition_bins {
            bail!(
                "ONNX model has {} condition bins, config expects {} for the 16 kHz condition",
                sessions.condition_bins,
                expected_condition_bins
            );
        }
        let frontend = Frontend::new(
            config.n_fft,
            config.hop_length,
            config.alpha,
            config.beta,
            config.comp_eps,
        );
        let backend = Backend::new(config.n_fft, config.hop_length, config.alpha, config.beta);
        let preferred_chunk = config.target_sr as usize;
        let chunk_samples = if frontend.num_frames(preferred_chunk) == sessions.time_frames {
            preferred_chunk
        } else {
            sessions.time_frames.saturating_sub(1) * config.hop_length
        };
        if frontend.num_frames(chunk_samples) != sessions.time_frames {
            bail!("cannot derive waveform length for the fixed ONNX time dimension");
        }
        let solver = OdeSolver::new(sidespread_config.ode_steps, config.guidance_scale);
        Ok(Self {
            sessions,
            config,
            frontend,
            backend,
            solver,
            chunk_samples,
            cutoff_hz: sidespread_config.fc,
        })
    }

    pub fn repair_signal(&mut self, side: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        self.repair_signal_with_progress(side, sample_rate, || {})
    }

    pub fn repair_signal_with_progress<F>(
        &mut self,
        side: &[f32],
        sample_rate: u32,
        on_chunk: F,
    ) -> Result<Vec<f32>>
    where
        F: FnMut(),
    {
        let at_target_rate = resample(side, sample_rate, self.config.target_sr)?;
        let repaired = self.repair_48k(&at_target_rate, on_chunk)?;
        let mut output = resample(&repaired, self.config.target_sr, sample_rate)?;
        output.resize(side.len(), 0.0);
        output.truncate(side.len());
        Ok(output)
    }

    pub fn chunk_count(&self, input_samples: usize, sample_rate: u32) -> usize {
        if input_samples == 0 {
            return 0;
        }
        let target_samples = (input_samples as f64 * self.config.target_sr as f64
            / sample_rate.max(1) as f64)
            .round() as usize;
        if target_samples <= self.chunk_samples {
            return 1;
        }
        let hop = (self.chunk_samples * 3 / 4).max(1);
        1 + (target_samples - self.chunk_samples).div_ceil(hop)
    }

    fn repair_48k<F>(&mut self, side: &[f32], mut on_chunk: F) -> Result<Vec<f32>>
    where
        F: FnMut(),
    {
        if side.is_empty() {
            return Ok(Vec::new());
        }
        // A 250 ms overlap is ample for the model's one-second receptive field and avoids the
        // previous 50% overlap's duplicate inference cost.
        let hop = (self.chunk_samples * 3 / 4).max(1);
        let fade = self.chunk_samples - hop;
        let mut accumulated = vec![0.0f32; side.len()];
        let mut weights = vec![0.0f32; side.len()];
        let mut start = 0usize;
        let mut chunk_index = 0u32;

        loop {
            let end = (start + self.chunk_samples).min(side.len());
            let mut chunk = vec![0.0f32; self.chunk_samples];
            chunk[..end - start].copy_from_slice(&side[start..end]);
            let repaired = self.repair_chunk(&chunk, 42u32.wrapping_add(chunk_index))?;
            on_chunk();
            for local in 0..end - start {
                let mut weight = 1.0f32;
                if start > 0 && local < fade {
                    weight *= smoothstep(local as f32 / fade as f32);
                }
                if end < side.len() && local + fade >= self.chunk_samples {
                    let remaining = self.chunk_samples - local - 1;
                    weight *= smoothstep(remaining as f32 / fade as f32);
                }
                accumulated[start + local] += repaired[local] * weight;
                weights[start + local] += weight;
            }
            if end == side.len() {
                break;
            }
            start += hop;
            chunk_index = chunk_index.wrapping_add(1);
        }

        Ok(accumulated
            .into_iter()
            .zip(weights)
            .zip(side)
            .map(|((sample, weight), original)| {
                if weight > 1e-8 {
                    sample / weight
                } else {
                    *original
                }
            })
            .collect())
    }

    fn repair_chunk(&mut self, side: &[f32], seed: u32) -> Result<Vec<f32>> {
        let (original_spec, n_bins, time_frames) = self.frontend.preprocess(side);
        if time_frames != self.sessions.time_frames || n_bins != self.config.total_freq_bins {
            bail!("frontend output does not match the fixed ONNX model shape");
        }
        let condition_bins = self.sessions.condition_bins;
        let condition = slice_frequency(&original_spec, n_bins, time_frames, 0, condition_bins);
        let x_shape = (1, 2, self.sessions.hr_bins, time_frames);
        let y_shape = (1, 2, condition_bins, time_frames);
        let initial = gaussian_noise(2 * self.sessions.hr_bins * time_frames, seed);
        let generated =
            self.solver
                .solve(&mut self.sessions, &initial, x_shape, &condition, y_shape)?;

        let hf_start = self.config.hf_start_bin();
        let mut generated_full = original_spec.clone();
        for channel in 0..2 {
            for generated_bin in 0..self.sessions.hr_bins {
                let target_bin = hf_start + generated_bin;
                if target_bin >= n_bins {
                    break;
                }
                for time in 0..time_frames {
                    let source =
                        (channel * self.sessions.hr_bins + generated_bin) * time_frames + time;
                    let target = (channel * n_bins + target_bin) * time_frames + time;
                    generated_full[target] = generated[source];
                }
            }
        }

        let cutoff_bin = frequency_bin(
            self.cutoff_hz as f32,
            self.config.n_fft,
            self.config.target_sr,
        );
        let transition = frequency_bin(500.0, self.config.n_fft, self.config.target_sr).max(2);
        let mut merged = vec![0.0f32; generated_full.len()];
        band_merge(
            &mut merged,
            &original_spec,
            &generated_full,
            n_bins,
            time_frames,
            cutoff_bin,
            transition,
        );

        let full_bins = self.config.n_fft / 2 + 1;
        let mut spectrum = vec![0.0f32; 2 * full_bins * time_frames];
        for channel in 0..2 {
            for bin in 0..n_bins {
                for time in 0..time_frames {
                    spectrum[(channel * full_bins + bin) * time_frames + time] =
                        merged[(channel * n_bins + bin) * time_frames + time];
                }
            }
        }
        Ok(self
            .backend
            .postprocess(&spectrum, full_bins, time_frames, side.len()))
    }
}

pub fn repair_signal(side: &[f32], config: &Config, sample_rate: u32) -> Result<Vec<f32>> {
    let mut state = NeuralState::from_config(config)?;
    state.repair_signal(side, sample_rate)
}

/// Keep UniverSR as a band-limited, non-cancelling delta and limit every frame against a local
/// Side/Mid loudness estimate from the intact band below the repair cutoff.
pub fn guard_candidate(
    mid: &[f32],
    original: &[f32],
    candidate: &[f32],
    config: &Config,
    sample_rate: u32,
) -> GuardedNeuralAudio {
    guard_candidate_with_detection(mid, original, original, candidate, config, sample_rate)
}

pub(crate) fn guard_candidate_with_detection(
    mid: &[f32],
    original: &[f32],
    detection_side: &[f32],
    candidate: &[f32],
    config: &Config,
    sample_rate: u32,
) -> GuardedNeuralAudio {
    if mid.len() != original.len()
        || detection_side.len() != original.len()
        || candidate.len() != original.len()
        || original.is_empty()
        || !candidate.iter().all(|sample| sample.is_finite())
    {
        return bypass_candidate(original);
    }

    let settings = StftConfig::new(config.n_fft, config.hop);
    let mid_spec = stft(mid, &settings);
    let original_spec = stft(original, &settings);
    let detection_spec = stft(detection_side, &settings);
    let candidate_spec = stft(candidate, &settings);
    let frames = mid_spec
        .len()
        .min(original_spec.len())
        .min(detection_spec.len())
        .min(candidate_spec.len());
    if frames == 0 {
        return bypass_candidate(original);
    }

    let n_bins = config.n_fft / 2 + 1;
    let model_start = bin_of(config.fc as f32, config.n_fft, sample_rate).min(n_bins);
    let defect_map = defects::analyze(
        &mid_spec[..frames],
        &detection_spec[..frames],
        config.scan_start_hz,
        config.n_fft,
        sample_rate,
    );
    let intact_start =
        bin_of(config.scan_start_hz as f32, config.n_fft, sample_rate).min(model_start);
    let intact_end = model_start.max(intact_start + 1).min(n_bins);
    let boost_power = 10.0f64.powf(config.neural_max_hf_boost_db as f64 / 10.0);

    let mut raw_limits = (0..frames)
        .map(|frame| {
            frame_delta_limit(
                &mid_spec[frame],
                &original_spec[frame],
                &candidate_spec[frame],
                intact_start,
                intact_end,
                model_start,
                &defect_map.masks[frame],
                boost_power,
            )
        })
        .collect::<Vec<_>>();
    stabilize_padded_edges(&mut raw_limits, config.n_fft / 2 / config.hop.max(1));
    let limits = smooth_gain_limits(&raw_limits, 0.10);
    let minimum_retained_mix = limits.iter().copied().fold(1.0f32, f32::min);
    let mean_retained_mix = limits.iter().sum::<f32>() / limits.len() as f32;
    if limits.iter().all(|gain| *gain <= 1e-8) {
        return GuardedNeuralAudio {
            signal: original.to_vec(),
            minimum_retained_mix: 0.0,
            mean_retained_mix: 0.0,
        };
    }

    let mut delta_spec = Vec::with_capacity(frames);
    for frame in 0..frames {
        let mut delta_frame = SpectrumFrame::new(n_bins);
        for bin in model_start..n_bins {
            let (delta_re, delta_im) = non_cancelling_delta(
                original_spec[frame].re(bin),
                original_spec[frame].im(bin),
                candidate_spec[frame].re(bin),
                candidate_spec[frame].im(bin),
            );
            let mix = limits[frame] * defect_map.masks[frame][bin];
            delta_frame.cplx[2 * bin] = delta_re * mix;
            delta_frame.cplx[2 * bin + 1] = delta_im * mix;
        }
        delta_spec.push(delta_frame);
    }
    let delta = istft(&delta_spec, &settings, original.len());
    if delta.len() != original.len() || !delta.iter().all(|sample| sample.is_finite()) {
        return bypass_candidate(original);
    }
    GuardedNeuralAudio {
        signal: original
            .iter()
            .zip(delta)
            .map(|(original, delta)| original + delta)
            .collect(),
        minimum_retained_mix,
        mean_retained_mix,
    }
}

#[allow(clippy::too_many_arguments)]
fn frame_delta_limit(
    mid: &SpectrumFrame,
    original: &SpectrumFrame,
    candidate: &SpectrumFrame,
    intact_start: usize,
    intact_end: usize,
    model_start: usize,
    defect_mask: &[f32],
    boost_power: f64,
) -> f32 {
    let side_mid_ratio = band_energy(original, intact_start, intact_end)
        / band_energy(mid, intact_start, intact_end).max(1e-12);
    let side_mid_ratio = side_mid_ratio.clamp(0.0, 4.0);
    let original_hf = weighted_band_energy(original, defect_mask, model_start);
    let expected_hf = weighted_band_energy(mid, defect_mask, model_start) * side_mid_ratio;
    let ceiling = original_hf.max(expected_hf * boost_power);

    let mut projection = 0.0f64;
    let mut delta_energy = 0.0f64;
    for bin in model_start..original.n_bins.min(candidate.n_bins) {
        let frequency_mix = defect_mask.get(bin).copied().unwrap_or(0.0) as f64;
        let original_re = original.re(bin) as f64;
        let original_im = original.im(bin) as f64;
        let (delta_re, delta_im) = non_cancelling_delta(
            original_re as f32,
            original_im as f32,
            candidate.re(bin),
            candidate.im(bin),
        );
        let delta_re = delta_re as f64 * frequency_mix;
        let delta_im = delta_im as f64 * frequency_mix;
        projection += original_re * delta_re + original_im * delta_im;
        delta_energy += delta_re * delta_re + delta_im * delta_im;
    }
    if delta_energy <= 1e-20 {
        return 1.0;
    }
    let available = (ceiling - original_hf).max(0.0);
    let root = (-projection
        + (projection * projection + delta_energy * available)
            .max(0.0)
            .sqrt())
        / delta_energy;
    root.clamp(0.0, 1.0) as f32
}

fn non_cancelling_delta(
    original_re: f32,
    original_im: f32,
    candidate_re: f32,
    candidate_im: f32,
) -> (f32, f32) {
    let mut delta_re = candidate_re - original_re;
    let mut delta_im = candidate_im - original_im;
    let original_power = original_re * original_re + original_im * original_im;
    let projection = original_re * delta_re + original_im * delta_im;
    if projection < 0.0 && original_power > 1e-20 {
        let anti_parallel = projection / original_power;
        delta_re -= anti_parallel * original_re;
        delta_im -= anti_parallel * original_im;
    }
    (delta_re, delta_im)
}

fn band_energy(frame: &SpectrumFrame, start: usize, end: usize) -> f64 {
    (start.min(frame.n_bins)..end.min(frame.n_bins))
        .map(|bin| frame.mag(bin).powi(2) as f64)
        .sum()
}

fn weighted_band_energy(frame: &SpectrumFrame, mask: &[f32], start: usize) -> f64 {
    (start.min(frame.n_bins)..frame.n_bins)
        .map(|bin| {
            let weight = mask.get(bin).copied().unwrap_or(0.0) as f64;
            frame.mag(bin).powi(2) as f64 * weight * weight
        })
        .sum()
}

fn smooth_gain_limits(raw: &[f32], maximum_rise_per_frame: f32) -> Vec<f32> {
    let mut smoothed = raw.to_vec();
    for index in 1..smoothed.len() {
        smoothed[index] = smoothed[index].min(smoothed[index - 1] + maximum_rise_per_frame);
    }
    for index in (0..smoothed.len().saturating_sub(1)).rev() {
        smoothed[index] = smoothed[index].min(smoothed[index + 1] + maximum_rise_per_frame);
    }
    smoothed
}

fn stabilize_padded_edges(values: &mut [f32], requested_edge: usize) {
    let edge = requested_edge.min(values.len().saturating_sub(1) / 2);
    if edge == 0 {
        return;
    }
    let left = values[edge];
    let right = values[values.len() - edge - 1];
    values[..edge].fill(left);
    let length = values.len();
    values[length - edge..].fill(right);
}

fn bypass_candidate(original: &[f32]) -> GuardedNeuralAudio {
    GuardedNeuralAudio {
        signal: original.to_vec(),
        minimum_retained_mix: 0.0,
        mean_retained_mix: 0.0,
    }
}

fn validate_model_cutoff(
    fc_hz: usize,
    condition_bins: usize,
    n_fft: usize,
    sample_rate: u32,
) -> Result<()> {
    let bin_width_hz = sample_rate as f32 / n_fft as f32;
    let model_cutoff_hz = condition_bins as f32 * bin_width_hz;
    if (fc_hz as f32 - model_cutoff_hz).abs() > bin_width_hz {
        bail!(
            "neural repair only supports --fc near {model_cutoff_hz:.2} Hz \
             (the model's {condition_bins}-bin condition, tolerance {bin_width_hz:.2} Hz); \
             got {fc_hz} Hz"
        );
    }
    Ok(())
}

fn models_directory(config: &Config) -> PathBuf {
    match &config.model_path {
        Some(path) if path.is_dir() => path.clone(),
        Some(path) => path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
        None => PathBuf::from("models"),
    }
}

fn slice_frequency(
    spectrum: &[f32],
    bins: usize,
    time_frames: usize,
    start: usize,
    length: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; 2 * length * time_frames];
    for channel in 0..2 {
        for bin in 0..length {
            for time in 0..time_frames {
                output[(channel * length + bin) * time_frames + time] =
                    spectrum[(channel * bins + start + bin) * time_frames + time];
            }
        }
    }
    output
}

fn gaussian_noise(length: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
    let mut output = vec![0.0f32; length];
    let mut index = 0;
    while index < length {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let u1 = (state as f32 / u32::MAX as f32).max(1e-10);
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let u2 = state as f32 / u32::MAX as f32;
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f32::consts::PI * u2;
        output[index] = radius * angle.cos();
        if index + 1 < length {
            output[index + 1] = radius * angle.sin();
        }
        index += 2;
    }
    output
}

fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate || input.is_empty() {
        return Ok(input.to_vec());
    }
    let mut resampler = FftFixedInOut::<f32>::new(from_rate as usize, to_rate as usize, 1024, 1)
        .context("creating rubato resampler")?;
    let input_frames = resampler.input_frames_next();
    let delay = resampler.output_delay();
    let expected_length = (input.len() as f64 * to_rate as f64 / from_rate as f64).round() as usize;
    let mut output = Vec::with_capacity(expected_length + delay + resampler.output_frames_max());
    let mut position = 0usize;
    while position + input_frames <= input.len() {
        let chunk = [input[position..position + input_frames].to_vec()];
        output.extend(resampler.process(&chunk, None)?[0].iter().copied());
        position += input_frames;
    }
    if position < input.len() {
        let tail = [input[position..].to_vec()];
        output.extend(
            resampler.process_partial(Some(&tail), None)?[0]
                .iter()
                .copied(),
        );
    }
    output.extend(
        resampler.process_partial::<Vec<f32>>(None, None)?[0]
            .iter()
            .copied(),
    );
    if output.len() > delay {
        output.drain(..delay);
    }
    output.resize(expected_length, 0.0);
    output.truncate(expected_length);
    Ok(output)
}

fn frequency_bin(hz: f32, n_fft: usize, sample_rate: u32) -> usize {
    ((hz * n_fft as f32) / sample_rate as f32).round() as usize
}

fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rubato_roundtrip_preserves_length_and_finiteness() {
        let input = (0..44_100)
            .map(|index| (index as f32 * 0.01).sin() * 0.25)
            .collect::<Vec<_>>();
        let upsampled = resample(&input, 44_100, 48_000).unwrap();
        let roundtrip = resample(&upsampled, 48_000, 44_100).unwrap();
        assert_eq!(roundtrip.len(), input.len());
        assert!(roundtrip.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn default_cutoff_matches_the_bundled_condition() {
        assert!(validate_model_cutoff(8000, 170, 1024, 48_000).is_ok());
    }

    #[test]
    fn incompatible_neural_cutoff_is_rejected() {
        let error = validate_model_cutoff(6000, 170, 1024, 48_000).unwrap_err();
        assert!(error.to_string().contains("only supports --fc near"));
    }

    fn sine(frequency: f32, index: usize, sample_rate: u32) -> f32 {
        (2.0 * std::f32::consts::PI * frequency * index as f32 / sample_rate as f32).sin()
    }

    #[test]
    fn neural_guard_keeps_a_balanced_candidate() {
        let sample_rate = 48_000;
        let length = sample_rate as usize;
        let mid = (0..length)
            .map(|index| {
                0.2 * sine(6_000.0, index, sample_rate) + 0.2 * sine(10_000.0, index, sample_rate)
            })
            .collect::<Vec<_>>();
        let original = (0..length)
            .map(|index| 0.1 * sine(6_000.0, index, sample_rate))
            .collect::<Vec<_>>();
        let candidate = original
            .iter()
            .enumerate()
            .map(|(index, sample)| sample + 0.1 * sine(10_000.0, index, sample_rate))
            .collect::<Vec<_>>();

        let guarded = guard_candidate(&mid, &original, &candidate, &Config::default(), sample_rate);
        assert!(
            guarded.mean_retained_mix > 0.95,
            "retained={}",
            guarded.mean_retained_mix
        );
        assert!(guarded.signal.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn neural_guard_limits_an_abnormally_loud_high_band() {
        let sample_rate = 48_000;
        let length = sample_rate as usize;
        let mid = (0..length)
            .map(|index| {
                0.2 * sine(6_000.0, index, sample_rate) + 0.2 * sine(10_000.0, index, sample_rate)
            })
            .collect::<Vec<_>>();
        let original = (0..length)
            .map(|index| 0.1 * sine(6_000.0, index, sample_rate))
            .collect::<Vec<_>>();
        let candidate = original
            .iter()
            .enumerate()
            .map(|(index, sample)| sample + 4.0 * sine(10_000.0, index, sample_rate))
            .collect::<Vec<_>>();

        let guarded = guard_candidate(&mid, &original, &candidate, &Config::default(), sample_rate);
        assert!(guarded.mean_retained_mix < 0.1);
        assert!(
            guarded
                .signal
                .iter()
                .map(|sample| sample.abs())
                .fold(0.0f32, f32::max)
                < 0.5
        );
    }

    #[test]
    fn neural_guard_does_not_cancel_existing_high_frequency_content() {
        let sample_rate = 48_000;
        let length = sample_rate as usize;
        let mid = (0..length)
            .map(|index| 0.2 * sine(10_000.0, index, sample_rate))
            .collect::<Vec<_>>();
        let original = (0..length)
            .map(|index| 0.05 * sine(10_000.0, index, sample_rate))
            .collect::<Vec<_>>();
        let candidate = original.iter().map(|sample| -*sample).collect::<Vec<_>>();

        let guarded = guard_candidate(&mid, &original, &candidate, &Config::default(), sample_rate);
        let error = guarded
            .signal
            .iter()
            .zip(&original)
            .map(|(guarded, original)| (guarded - original).powi(2))
            .sum::<f32>();
        assert!(error < 1e-10);
    }

    #[test]
    fn frame_gain_smoothing_never_exceeds_a_local_limit() {
        let raw = [1.0, 1.0, 0.1, 1.0, 1.0];
        let smoothed = smooth_gain_limits(&raw, 0.2);
        assert!(smoothed
            .iter()
            .zip(raw)
            .all(|(smoothed, raw)| *smoothed <= raw));
        assert_eq!(smoothed, vec![0.5, 0.3, 0.1, 0.3, 0.5]);
    }
}
