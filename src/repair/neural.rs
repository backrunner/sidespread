//! B route: targeted high-frequency repair with UniverSR ONNX.

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
        let sessions = Sessions::load(&config.model_onnx)?;
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
        let at_target_rate = resample(side, sample_rate, self.config.target_sr)?;
        let repaired = self.repair_48k(&at_target_rate)?;
        let mut output = resample(&repaired, self.config.target_sr, sample_rate)?;
        output.resize(side.len(), 0.0);
        output.truncate(side.len());
        Ok(output)
    }

    fn repair_48k(&mut self, side: &[f32]) -> Result<Vec<f32>> {
        if side.is_empty() {
            return Ok(Vec::new());
        }
        let hop = (self.chunk_samples / 2).max(1);
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
}
