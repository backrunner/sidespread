//! Configuration: defaults and CLI overrides.

use anyhow::{bail, Result};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    /// Conservatively apply DSP only where intact-band evidence is strong.
    Auto,
    /// Force DSP route (mid→side HF folding) on every segment that needs repair.
    Dsp,
    /// Force neural route (UniverSR) on every segment that needs repair.
    Nn,
    /// Skip all repair; only analyze and report.
    Skip,
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::Auto => write!(f, "auto"),
            Mode::Dsp => write!(f, "dsp"),
            Mode::Nn => write!(f, "nn"),
            Mode::Skip => write!(f, "skip"),
        }
    }
}

/// Per-segment repair route chosen by the detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Route {
    Skip,
    Dsp,
    Neural,
    Hybrid,
}

impl std::fmt::Display for Route {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Route::Skip => write!(f, "skip"),
            Route::Dsp => write!(f, "dsp"),
            Route::Neural => write!(f, "neural"),
            Route::Hybrid => write!(f, "hybrid"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    /// High-frequency cutoff in Hz (default 8000).
    pub fc: usize,
    /// R_hf below this → side HF deficient (default 0.3).
    pub rhf_threshold: f32,
    /// Smoothed intact-band correlation above this selects DSP (default 0.35).
    pub corr_high: f32,
    /// Reserved lower correlation threshold (default 0.15).
    pub corr_low: f32,
    /// User-forced mode.
    pub mode: Mode,
    /// ODE solver steps for the neural route (default 4).
    pub ode_steps: usize,
    /// Segment length in ms (default 80).
    pub segment_ms: usize,
    /// Segment overlap fraction (default 0.5).
    pub overlap: f32,
    /// STFT FFT size (default 4096).
    pub n_fft: usize,
    /// STFT hop size (default 1024).
    pub hop: usize,
    /// Path to the UniverSR ONNX model (optional until P5).
    pub model_path: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            fc: 8000,
            rhf_threshold: 0.3,
            corr_high: 0.35,
            corr_low: 0.15,
            mode: Mode::Auto,
            ode_steps: 4,
            segment_ms: 80,
            overlap: 0.5,
            n_fft: 4096,
            hop: 1024,
            model_path: None,
        }
    }
}

impl Config {
    pub fn from_detect(fc: usize, rhf_threshold: f32, (corr_high, corr_low): (f32, f32)) -> Self {
        Self {
            fc,
            rhf_threshold,
            corr_high,
            corr_low,
            ..Self::default()
        }
    }

    pub fn from_process(
        fc: usize,
        rhf_threshold: f32,
        (corr_high, corr_low): (f32, f32),
        mode: Mode,
        ode_steps: usize,
        model_path: Option<PathBuf>,
    ) -> Self {
        Self {
            fc,
            rhf_threshold,
            corr_high,
            corr_low,
            mode,
            ode_steps,
            model_path,
            ..Self::default()
        }
    }

    /// Decide the repair route for a segment given its detection metrics.
    pub fn decide(&self, needs: bool, corr_hf: f32) -> Route {
        if !needs {
            return Route::Skip;
        }
        match self.mode {
            Mode::Skip => Route::Skip,
            Mode::Dsp => Route::Dsp,
            Mode::Nn => Route::Neural,
            Mode::Auto => {
                if corr_hf >= self.corr_high {
                    Route::Dsp
                } else {
                    Route::Skip
                }
            }
        }
    }

    pub fn validate(&self, sample_rate: u32) -> Result<()> {
        if !matches!(sample_rate, 44_100 | 48_000) {
            bail!("unsupported sample rate {sample_rate} Hz; expected 44100 or 48000 Hz");
        }
        if self.fc == 0 || self.fc >= sample_rate as usize / 2 {
            bail!(
                "--fc must be between 1 Hz and Nyquist ({} Hz)",
                sample_rate / 2
            );
        }
        if !self.rhf_threshold.is_finite() || self.rhf_threshold < 0.0 {
            bail!("--rhf-threshold must be a finite non-negative number");
        }
        if !self.corr_high.is_finite()
            || !self.corr_low.is_finite()
            || !(-1.0..=1.0).contains(&self.corr_high)
            || !(-1.0..=1.0).contains(&self.corr_low)
            || self.corr_high < self.corr_low
        {
            bail!("--corr-threshold must be HIGH,LOW with -1 <= LOW <= HIGH <= 1");
        }
        if self.ode_steps == 0 {
            bail!("--ode-steps must be at least 1");
        }
        if self.segment_ms == 0 {
            bail!("segment length must be positive");
        }
        if !(0.0..1.0).contains(&self.overlap) {
            bail!("segment overlap must be in [0, 1)");
        }
        if self.n_fft == 0 || self.hop == 0 || self.hop > self.n_fft {
            bail!("invalid STFT configuration");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_mode_skips_low_confidence_repairs() {
        let config = Config::default();
        assert_eq!(config.decide(true, 0.8), Route::Dsp);
        assert_eq!(config.decide(true, 0.2), Route::Skip);
        assert_eq!(config.decide(false, 0.8), Route::Skip);
    }

    #[test]
    fn neural_mode_remains_available_explicitly() {
        let config = Config {
            mode: Mode::Nn,
            ..Config::default()
        };
        assert_eq!(config.decide(true, -0.5), Route::Neural);
    }
}
