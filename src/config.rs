//! Configuration: defaults and CLI overrides.

use anyhow::{bail, Result};
use std::path::PathBuf;

pub const DEFAULT_SMEARING_STRENGTH: f32 = 0.35;
pub const DEFAULT_BLEEDING_STRENGTH: f32 = 0.65;
pub const DEFAULT_PHASE_STRENGTH: f32 = 0.50;
pub const DEFAULT_ODE_STEPS: usize = 2;
pub const DEFAULT_HYBRID_NEURAL_MIX: f32 = 0.30;
pub const DEFAULT_HYBRID_NEURAL_DEPTH: f32 = 0.35;
pub const DEFAULT_NEURAL_MAX_HF_BOOST_DB: f32 = 0.0;
pub const DEFAULT_SCAN_START_HZ: usize = 5_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    /// Apply DSP only where near-cutoff and transition evidence is strong.
    Auto,
    /// Force DSP route (mid→side HF folding) on every segment that needs repair.
    Dsp,
    /// Force neural route (UniverSR) on every segment that needs repair.
    Nn,
    /// Use DSP normally and add guarded UniverSR detail only for deep dropouts.
    Hybrid,
    /// Skip all repair; only analyze and report.
    Skip,
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::Auto => write!(f, "auto"),
            Mode::Dsp => write!(f, "dsp"),
            Mode::Nn => write!(f, "nn"),
            Mode::Hybrid => write!(f, "hybrid"),
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
    /// Lowest frequency inspected by dynamic multi-band deficiency detection.
    pub scan_start_hz: usize,
    /// Absolute ceiling for the side/mid HF deficiency threshold (default 0.3).
    pub rhf_threshold: f32,
    /// Minimum high/intact side-energy ratio before a segment is considered deficient (0.18).
    pub rhf_relative_threshold: f32,
    /// Smoothed intact-band correlation above this selects DSP in evaluation auto mode.
    pub corr_high: f32,
    /// Smoothed outer-transition correlation required for DSP (default 0.40).
    pub corr_low: f32,
    /// Minimum smoothed S/M energy ratio in the outer cutoff transition.
    pub transition_rhf_min: f32,
    /// Scale applied to the synthesized DSP energy deficit, in [0, 3].
    pub dsp_strength: f32,
    /// Maximum smooth phase diffusion applied to synthesized DSP bins.
    pub dsp_phase_degrees: f32,
    /// Detect and perceptually extend a shared M/S brick-wall high-frequency cutoff.
    pub bandwidth_extension: bool,
    /// Maximum tail/edge energy ratio, in dB, treated as a shared hard cutoff.
    pub bandwidth_drop_db: f32,
    /// Scale applied to synthesized full-band extension energy, in [0, 2].
    pub bandwidth_strength: f32,
    /// Additional attenuation per synthesized octave above the detected cutoff.
    pub bandwidth_rolloff_db_per_octave: f32,
    /// Sharpen high-frequency onsets that have been temporally smeared.
    pub repair_smearing: bool,
    /// High-frequency transient enhancement amount, in [0, 1].
    pub smearing_strength: f32,
    /// Suppress low-level inter-harmonic high-frequency residue.
    pub repair_bleeding: bool,
    /// Harmonic debleed amount, in [0, 1].
    pub bleeding_strength: f32,
    /// Stabilize incoherent high-frequency Side phase over time.
    pub stabilize_phase: bool,
    /// Side phase stabilization amount, in [0, 1].
    pub phase_strength: f32,
    /// Allow output and report paths to be overwritten without prompting.
    pub overwrite_existing: bool,
    /// Missing-band processing strategy. Process defaults to DSP.
    pub mode: Mode,
    /// ODE solver steps for the neural route (default 2).
    pub ode_steps: usize,
    /// Neural delta mixed on top of DSP in hybrid mode, in [0, 1].
    pub hybrid_neural_mix: f32,
    /// Observed/target HF ratio below which hybrid mode invokes UniverSR.
    pub hybrid_neural_depth: f32,
    /// Maximum neural HF energy above the local M/S estimate, in dB.
    pub neural_max_hf_boost_db: f32,
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
            scan_start_hz: DEFAULT_SCAN_START_HZ,
            rhf_threshold: 0.3,
            rhf_relative_threshold: 0.18,
            corr_high: 0.35,
            corr_low: 0.40,
            transition_rhf_min: 1.0e-3,
            dsp_strength: 2.0,
            dsp_phase_degrees: 60.0,
            bandwidth_extension: true,
            bandwidth_drop_db: -30.0,
            bandwidth_strength: 0.85,
            bandwidth_rolloff_db_per_octave: 3.0,
            repair_smearing: false,
            smearing_strength: DEFAULT_SMEARING_STRENGTH,
            repair_bleeding: false,
            bleeding_strength: DEFAULT_BLEEDING_STRENGTH,
            stabilize_phase: false,
            phase_strength: DEFAULT_PHASE_STRENGTH,
            overwrite_existing: false,
            mode: Mode::Dsp,
            ode_steps: DEFAULT_ODE_STEPS,
            hybrid_neural_mix: DEFAULT_HYBRID_NEURAL_MIX,
            hybrid_neural_depth: DEFAULT_HYBRID_NEURAL_DEPTH,
            neural_max_hf_boost_db: DEFAULT_NEURAL_MAX_HF_BOOST_DB,
            segment_ms: 80,
            overlap: 0.5,
            n_fft: 4096,
            hop: 1024,
            model_path: None,
        }
    }
}

impl Config {
    pub fn from_detect(fc: usize, rhf_threshold: f32) -> Self {
        Self {
            fc,
            rhf_threshold,
            ..Self::default()
        }
    }

    pub fn from_process(fc: usize, rhf_threshold: f32) -> Self {
        Self {
            fc,
            rhf_threshold,
            repair_smearing: true,
            repair_bleeding: true,
            stabilize_phase: true,
            mode: Mode::Dsp,
            ..Self::default()
        }
    }

    pub fn from_evaluation(
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
    pub fn decide(
        &self,
        needs: bool,
        corr_intact: f32,
        corr_transition: f32,
        r_transition: f32,
    ) -> Route {
        self.decide_with_deficiency(
            needs,
            corr_intact,
            corr_transition,
            r_transition,
            f32::INFINITY,
            f32::INFINITY,
            1.0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decide_with_deficiency(
        &self,
        needs: bool,
        corr_intact: f32,
        corr_transition: f32,
        r_transition: f32,
        r_hf: f32,
        r_hf_low: f32,
        r_intact: f32,
    ) -> Route {
        if !needs {
            return Route::Skip;
        }
        match self.mode {
            Mode::Skip => Route::Skip,
            Mode::Dsp => Route::Dsp,
            Mode::Nn => Route::Neural,
            Mode::Hybrid => {
                let threshold = self.repair_threshold(r_intact);
                let depth = r_hf.min(r_hf_low) / threshold.max(1e-12);
                if self.hybrid_neural_mix > 0.0
                    && threshold > 1e-12
                    && depth <= self.hybrid_neural_depth
                {
                    Route::Hybrid
                } else {
                    Route::Dsp
                }
            }
            Mode::Auto => {
                if corr_intact >= self.corr_high
                    && corr_transition >= self.corr_low
                    && r_transition >= self.transition_rhf_min
                {
                    Route::Dsp
                } else {
                    Route::Skip
                }
            }
        }
    }

    pub fn needs_repair(&self, r_hf: f32, r_intact: f32) -> bool {
        r_hf < self.repair_threshold(r_intact)
    }

    pub fn repair_threshold(&self, r_intact: f32) -> f32 {
        self.rhf_threshold
            .min(r_intact.max(0.0) * self.rhf_relative_threshold)
    }

    pub fn artifact_processing_enabled(&self) -> bool {
        (self.repair_smearing && self.smearing_strength > 0.0)
            || (self.repair_bleeding && self.bleeding_strength > 0.0)
            || (self.stabilize_phase && self.phase_strength > 0.0)
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
        if self.scan_start_hz == 0 || self.scan_start_hz >= sample_rate as usize / 2 {
            bail!(
                "--scan-start-hz must be between 1 Hz and Nyquist ({} Hz)",
                sample_rate / 2
            );
        }
        if !self.rhf_threshold.is_finite() || self.rhf_threshold < 0.0 {
            bail!("--rhf-threshold must be a finite non-negative number");
        }
        if !self.rhf_relative_threshold.is_finite() || self.rhf_relative_threshold < 0.0 {
            bail!("relative HF threshold must be a finite non-negative number");
        }
        if !self.corr_high.is_finite()
            || !self.corr_low.is_finite()
            || !(-1.0..=1.0).contains(&self.corr_high)
            || !(-1.0..=1.0).contains(&self.corr_low)
        {
            bail!("--corr-threshold must be INTACT,TRANSITION with both values between -1 and 1");
        }
        if !self.transition_rhf_min.is_finite() || self.transition_rhf_min < 0.0 {
            bail!("transition energy threshold must be finite and non-negative");
        }
        if !self.dsp_strength.is_finite() || !(0.0..=3.0).contains(&self.dsp_strength) {
            bail!("DSP strength must be between 0 and 3");
        }
        if !self.dsp_phase_degrees.is_finite() || !(0.0..=180.0).contains(&self.dsp_phase_degrees) {
            bail!("DSP phase diffusion must be between 0 and 180 degrees");
        }
        if !self.bandwidth_drop_db.is_finite() || !(-80.0..=-12.0).contains(&self.bandwidth_drop_db)
        {
            bail!("bandwidth cutoff drop must be between -80 and -12 dB");
        }
        if !self.bandwidth_strength.is_finite() || !(0.0..=2.0).contains(&self.bandwidth_strength) {
            bail!("bandwidth extension strength must be between 0 and 2");
        }
        if !self.bandwidth_rolloff_db_per_octave.is_finite()
            || !(0.0..=24.0).contains(&self.bandwidth_rolloff_db_per_octave)
        {
            bail!("bandwidth extension rolloff must be between 0 and 24 dB per octave");
        }
        for (name, strength) in [
            ("smearing", self.smearing_strength),
            ("bleeding", self.bleeding_strength),
            ("phase", self.phase_strength),
        ] {
            if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
                bail!("{name} processing strength must be between 0 and 1");
            }
        }
        if self.ode_steps == 0 {
            bail!("--ode-steps must be at least 1");
        }
        if !self.hybrid_neural_mix.is_finite() || !(0.0..=1.0).contains(&self.hybrid_neural_mix) {
            bail!("--hybrid-neural-mix must be between 0 and 1");
        }
        if !self.hybrid_neural_depth.is_finite() || !(0.0..=1.0).contains(&self.hybrid_neural_depth)
        {
            bail!("--hybrid-neural-depth must be between 0 and 1");
        }
        if !self.neural_max_hf_boost_db.is_finite()
            || !(0.0..=12.0).contains(&self.neural_max_hf_boost_db)
        {
            bail!("--neural-max-hf-boost-db must be between 0 and 12");
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
        let config = Config {
            mode: Mode::Auto,
            ..Config::default()
        };
        assert_eq!(config.decide(true, 0.8, 0.4, 0.01), Route::Dsp);
        assert_eq!(config.decide(true, 0.2, 0.4, 0.01), Route::Skip);
        assert_eq!(config.decide(true, 0.8, 0.1, 0.01), Route::Skip);
        assert_eq!(config.decide(true, 0.8, 0.4, 1.0e-4), Route::Skip);
        assert_eq!(config.decide(false, 0.8, 0.4, 0.01), Route::Skip);
    }

    #[test]
    fn default_mode_repairs_every_deficient_segment() {
        let config = Config::default();
        assert_eq!(config.decide(true, -1.0, -1.0, 0.0), Route::Dsp);
        assert_eq!(config.decide(false, -1.0, -1.0, 0.0), Route::Skip);
    }

    #[test]
    fn deficiency_is_relative_to_the_intact_side_band() {
        let config = Config::default();
        assert!(config.needs_repair(0.004, 0.15));
        assert!(!config.needs_repair(0.10, 0.15));
        assert!(!config.needs_repair(0.0, 0.0));
        assert!(config.needs_repair(0.17, 1.0));
    }

    #[test]
    fn neural_mode_remains_available_explicitly() {
        let config = Config {
            mode: Mode::Nn,
            ..Config::default()
        };
        assert_eq!(config.decide(true, -0.5, -0.5, 0.0), Route::Neural);
    }

    #[test]
    fn hybrid_mode_reserves_neural_inference_for_deep_dropouts() {
        let config = Config {
            mode: Mode::Hybrid,
            ..Config::default()
        };
        assert_eq!(
            config.decide_with_deficiency(true, 0.0, 0.0, 0.0, 0.02, 0.01, 0.5),
            Route::Hybrid
        );
        assert_eq!(
            config.decide_with_deficiency(true, 0.0, 0.0, 0.0, 0.07, 0.06, 0.5),
            Route::Dsp
        );
    }

    #[test]
    fn independent_correlation_thresholds_need_not_be_ordered() {
        let config = Config {
            corr_high: 0.35,
            corr_low: 0.40,
            ..Config::default()
        };
        assert!(config.validate(48_000).is_ok());
    }

    #[test]
    fn artifact_strengths_are_validated_independently() {
        for config in [
            Config {
                smearing_strength: 1.01,
                ..Config::default()
            },
            Config {
                bleeding_strength: -0.01,
                ..Config::default()
            },
            Config {
                phase_strength: f32::NAN,
                ..Config::default()
            },
        ] {
            assert!(config.validate(48_000).is_err());
        }
    }

    #[test]
    fn zero_strength_does_not_request_artifact_processing() {
        let config = Config {
            repair_smearing: true,
            smearing_strength: 0.0,
            repair_bleeding: true,
            bleeding_strength: 0.0,
            stabilize_phase: true,
            phase_strength: 0.0,
            ..Config::default()
        };
        assert!(!config.artifact_processing_enabled());
    }

    #[test]
    fn process_enables_calibrated_artifact_repairs_without_changing_other_commands() {
        let process = Config::from_process(8_000, 0.3);
        assert!(process.repair_smearing);
        assert_eq!(process.smearing_strength, DEFAULT_SMEARING_STRENGTH);
        assert!(process.repair_bleeding);
        assert_eq!(process.bleeding_strength, DEFAULT_BLEEDING_STRENGTH);
        assert!(process.stabilize_phase);
        assert_eq!(process.phase_strength, DEFAULT_PHASE_STRENGTH);

        assert!(!Config::from_detect(8_000, 0.3).artifact_processing_enabled());
        assert!(
            !Config::from_evaluation(8_000, 0.3, (0.35, 0.40), Mode::Dsp, 4, None,)
                .artifact_processing_enabled()
        );
    }
}
