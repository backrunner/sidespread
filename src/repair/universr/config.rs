//! UniverSR pipeline config, loaded from `models/universr_config.json`.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct UniversrConfig {
    pub n_fft: usize,
    pub hop_length: usize,
    pub window_fn: String,
    pub sampling_rate: u32,
    pub alpha: f32,
    pub beta: f32,
    pub comp_eps: f32,
    pub total_freq_bins: usize,
    pub hr_freq_bins: usize,
    pub sr_to_lr_bins: HashMap<String, usize>,
    pub sigma_min: f32,
    pub guidance_scale: f32,
    pub ode_steps: usize,
    pub ode_method: String,
    pub min_samples: usize,
    pub target_sr: u32,
    pub model_onnx: PathBuf,
}

impl UniversrConfig {
    pub fn load(models_dir: &Path) -> Result<Self> {
        let path = models_dir.join("universr_config.json");
        let s = std::fs::read_to_string(&path)
            .with_context(|| format!("reading universr config: {}", path.display()))?;
        let mut cfg: Self = serde_json::from_str(&s).context("parsing universr config")?;
        cfg.model_onnx = resolve_path(models_dir, &cfg.model_onnx);
        Ok(cfg)
    }

    pub fn lr_bin_count(&self, sr_khz: usize) -> usize {
        self.sr_to_lr_bins
            .get(&sr_khz.to_string())
            .copied()
            .expect("sr_khz in {8,12,16,24}")
    }

    pub fn hf_start_bin(&self) -> usize {
        self.total_freq_bins - self.hr_freq_bins
    }
}

fn resolve_path(models_dir: &Path, configured: &Path) -> PathBuf {
    if configured.is_absolute() || configured.exists() {
        configured.to_path_buf()
    } else {
        models_dir.join(configured)
    }
}
