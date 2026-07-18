//! CLI definition and dispatch via clap.

use crate::config::Config;
use crate::pipeline;
use anyhow::Result;
use clap::{Parser, Subcommand};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "sidespread",
    version,
    about = "Repair missing high-frequency detail in AI-generated stereo music"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Process a stereo WAV: detect and repair missing high-frequency detail.
    Process {
        #[arg(value_name = "INPUT")]
        input: PathBuf,
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,
        #[arg(long, value_name = "HZ", default_value_t = 8000)]
        fc: usize,
        /// Lowest frequency scanned for dynamic multi-band Side defects.
        #[arg(
            long,
            value_name = "HZ",
            default_value_t = crate::config::DEFAULT_SCAN_START_HZ
        )]
        scan_start_hz: usize,
        #[arg(long, value_name = "THRESH", default_value_t = 0.3)]
        rhf_threshold: f32,
        /// Missing-band repair backend. UniverSR is used only by nn/hybrid.
        #[arg(long, value_enum, default_value_t = ProcessMode::Dsp)]
        mode: ProcessMode,
        /// UniverSR midpoint ODE steps. Higher is slower (default 2).
        #[arg(long, value_name = "STEPS", default_value_t = crate::config::DEFAULT_ODE_STEPS)]
        ode_steps: usize,
        /// UniverSR model file or model directory.
        #[arg(long, value_name = "ONNX")]
        model_path: Option<PathBuf>,
        /// Neural delta added on top of DSP in hybrid mode.
        #[arg(
            long,
            value_name = "0..1",
            default_value_t = crate::config::DEFAULT_HYBRID_NEURAL_MIX
        )]
        hybrid_neural_mix: f32,
        /// Invoke UniverSR below this observed/target HF ratio in hybrid mode.
        #[arg(
            long,
            value_name = "0..1",
            default_value_t = crate::config::DEFAULT_HYBRID_NEURAL_DEPTH
        )]
        hybrid_neural_depth: f32,
        /// Maximum neural HF energy above the local M/S estimate.
        #[arg(
            long,
            value_name = "DB",
            default_value_t = crate::config::DEFAULT_NEURAL_MAX_HF_BOOST_DB
        )]
        neural_max_hf_boost_db: f32,
        /// Disable the default high-frequency transient repair.
        #[arg(long)]
        no_repair_smearing: bool,
        /// Override the calibrated smearing repair strength (default 0.35).
        #[arg(long, value_name = "0..1", conflicts_with = "no_repair_smearing")]
        smearing_strength: Option<f32>,
        /// Disable the default inter-harmonic residue suppression.
        #[arg(long)]
        no_repair_bleeding: bool,
        /// Override the calibrated harmonic debleed strength (default 0.65).
        #[arg(long, value_name = "0..1", conflicts_with = "no_repair_bleeding")]
        bleeding_strength: Option<f32>,
        /// Disable the default high-frequency Side phase stabilization.
        #[arg(long)]
        no_stabilize_phase: bool,
        /// Override the calibrated phase stabilization strength (default 0.50).
        #[arg(long, value_name = "0..1", conflicts_with = "no_stabilize_phase")]
        phase_strength: Option<f32>,
        #[arg(long = "output-report", alias = "report", value_name = "PATH")]
        output_report: Option<PathBuf>,
        /// Overwrite existing output/report files without prompting.
        #[arg(long)]
        force: bool,
    },
    /// Detect only: show whether the audio needs processing and the route per segment.
    Detect {
        #[arg(value_name = "INPUT")]
        input: PathBuf,
        #[arg(long, value_name = "HZ", default_value_t = 8000)]
        fc: usize,
        #[arg(
            long,
            value_name = "HZ",
            default_value_t = crate::config::DEFAULT_SCAN_START_HZ
        )]
        scan_start_hz: usize,
        #[arg(long, value_name = "THRESH", default_value_t = 0.3)]
        rhf_threshold: f32,
        #[arg(long, hide = true, default_value_t = 0.18)]
        rhf_relative_threshold: f32,
        #[arg(long = "output-report", alias = "report", value_name = "PATH")]
        output_report: Option<PathBuf>,
        /// Overwrite an existing report without prompting.
        #[arg(long)]
        force: bool,
    },
    /// Evaluate on a synthetic degradation: clean → degrade side → repair → compare to original.
    #[command(hide = true)]
    Eval {
        #[arg(value_name = "CLEAN")]
        clean: PathBuf,
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = crate::config::Mode::Auto)]
        mode: crate::config::Mode,
        #[arg(long, value_name = "HZ", default_value_t = 8000)]
        fc: usize,
        #[arg(long, value_name = "THRESH", default_value_t = 0.3)]
        rhf_threshold: f32,
        #[arg(long, hide = true, default_value_t = 0.18)]
        rhf_relative_threshold: f32,
        #[arg(
            long,
            value_name = "INTACT,TRANSITION",
            value_delimiter = ',',
            default_value = "0.35,0.40"
        )]
        corr_threshold: Vec<f32>,
        #[arg(long, value_name = "STEPS", default_value_t = 4)]
        ode_steps: usize,
        #[arg(long = "output-report", alias = "report", value_name = "PATH")]
        output_report: Option<PathBuf>,
        #[arg(long)]
        force: bool,
        #[arg(long, value_name = "ONNX")]
        model_path: Option<PathBuf>,
        #[arg(long, hide = true, default_value_t = 2.0)]
        dsp_strength: f32,
        #[arg(long, hide = true, default_value_t = 60.0)]
        dsp_phase_degrees: f32,
    },
    /// Print WAV metadata and an M/S high-frequency energy overview.
    Info {
        #[arg(value_name = "INPUT")]
        input: PathBuf,
        #[arg(long, value_name = "HZ", default_value_t = 8000)]
        fc: usize,
    },
    /// Download or verify the optional UniverSR model.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ProcessMode {
    /// Fast deterministic DSP; does not load UniverSR.
    Dsp,
    /// Guarded UniverSR on every deficient repair span.
    Nn,
    /// DSP plus guarded UniverSR detail on deep dropouts only.
    Hybrid,
}

impl From<ProcessMode> for crate::config::Mode {
    fn from(value: ProcessMode) -> Self {
        match value {
            ProcessMode::Dsp => Self::Dsp,
            ProcessMode::Nn => Self::Nn,
            ProcessMode::Hybrid => Self::Hybrid,
        }
    }
}

#[derive(Subcommand)]
pub enum ModelCommand {
    /// Download the prebuilt model and verify its SHA-256.
    Download {
        #[arg(
            short,
            long,
            value_name = "PATH",
            default_value = "models/universr_backbone.onnx"
        )]
        output: PathBuf,
        /// Replace an existing model, even if it is already valid.
        #[arg(long)]
        force: bool,
    },
    /// Verify the size and SHA-256 of a downloaded model.
    Verify {
        #[arg(value_name = "PATH", default_value = "models/universr_backbone.onnx")]
        path: PathBuf,
    },
}

pub fn run() -> Result<()> {
    let cli = parse_args(std::env::args_os()).unwrap_or_else(|error| error.exit());
    match cli.command {
        Command::Model { command } => match command {
            ModelCommand::Download { output, force } => crate::model::download(&output, force),
            ModelCommand::Verify { path } => crate::model::verify(&path),
        },
        Command::Info { input, fc } => pipeline::info(&input, fc),
        Command::Detect {
            input,
            fc,
            scan_start_hz,
            rhf_threshold,
            rhf_relative_threshold,
            output_report,
            force,
        } => {
            let mut cfg = Config::from_detect(fc, rhf_threshold);
            cfg.scan_start_hz = scan_start_hz;
            cfg.rhf_relative_threshold = rhf_relative_threshold;
            cfg.overwrite_existing = force;
            pipeline::detect(&input, &cfg, output_report.as_deref())
        }
        Command::Process {
            input,
            output,
            fc,
            scan_start_hz,
            rhf_threshold,
            mode,
            ode_steps,
            model_path,
            hybrid_neural_mix,
            hybrid_neural_depth,
            neural_max_hf_boost_db,
            no_repair_smearing,
            smearing_strength,
            no_repair_bleeding,
            bleeding_strength,
            no_stabilize_phase,
            phase_strength,
            output_report,
            force,
        } => {
            let out = output.unwrap_or_else(|| default_output_path(&input, ".repaired"));
            let mut cfg = Config::from_process(fc, rhf_threshold);
            cfg.scan_start_hz = scan_start_hz;
            cfg.mode = mode.into();
            cfg.ode_steps = ode_steps;
            cfg.model_path = model_path;
            cfg.hybrid_neural_mix = hybrid_neural_mix;
            cfg.hybrid_neural_depth = hybrid_neural_depth;
            cfg.neural_max_hf_boost_db = neural_max_hf_boost_db;
            configure_artifact_processing(
                &mut cfg,
                no_repair_smearing,
                smearing_strength,
                no_repair_bleeding,
                bleeding_strength,
                no_stabilize_phase,
                phase_strength,
            );
            cfg.overwrite_existing = force;
            pipeline::process(&input, &out, &cfg, output_report.as_deref())
        }
        Command::Eval {
            clean,
            output,
            mode,
            fc,
            rhf_threshold,
            rhf_relative_threshold,
            corr_threshold,
            ode_steps,
            output_report,
            force,
            model_path,
            dsp_strength,
            dsp_phase_degrees,
        } => {
            let out = output.unwrap_or_else(|| default_output_path(&clean, ".eval_repaired"));
            let mut cfg = Config::from_evaluation(
                fc,
                rhf_threshold,
                parse_corr(&corr_threshold)?,
                mode,
                ode_steps,
                model_path,
            );
            cfg.dsp_strength = dsp_strength;
            cfg.dsp_phase_degrees = dsp_phase_degrees;
            cfg.rhf_relative_threshold = rhf_relative_threshold;
            cfg.overwrite_existing = force;
            pipeline::eval(&clean, &out, &cfg, output_report.as_deref())
        }
    }
}

fn parse_args<I>(args: I) -> std::result::Result<Cli, clap::Error>
where
    I: IntoIterator<Item = OsString>,
{
    let args = args.into_iter().collect::<Vec<_>>();
    let args = if should_use_process_alias(&args) {
        let mut normalized = Vec::with_capacity(args.len() + 1);
        normalized.push(args[0].clone());
        normalized.push(OsString::from("process"));
        normalized.extend(args.into_iter().skip(1));
        normalized
    } else {
        args
    };
    Cli::try_parse_from(args)
}

fn should_use_process_alias(args: &[OsString]) -> bool {
    let Some(first) = args.get(1).and_then(|value| value.to_str()) else {
        return false;
    };
    !first.starts_with('-') && !matches!(first, "process" | "detect" | "eval" | "info" | "model")
}

fn default_output_path(input: &Path, suffix: &str) -> PathBuf {
    let mut file_name = input
        .file_stem()
        .filter(|stem| !stem.is_empty())
        .unwrap_or_else(|| std::ffi::OsStr::new("output"))
        .to_os_string();
    file_name.push(suffix);
    file_name.push(".wav");
    input.with_file_name(file_name)
}

fn parse_corr(v: &[f32]) -> Result<(f32, f32)> {
    match v {
        [hi, lo] => Ok((*hi, *lo)),
        [x] => Ok((*x, *x * 0.667)),
        _ => anyhow::bail!("--corr-threshold expects INTACT,TRANSITION"),
    }
}

fn configure_artifact_processing(
    config: &mut Config,
    no_repair_smearing: bool,
    smearing_strength: Option<f32>,
    no_repair_bleeding: bool,
    bleeding_strength: Option<f32>,
    no_stabilize_phase: bool,
    phase_strength: Option<f32>,
) {
    config.repair_smearing = !no_repair_smearing;
    config.smearing_strength = smearing_strength.unwrap_or(config.smearing_strength);
    config.repair_bleeding = !no_repair_bleeding;
    config.bleeding_strength = bleeding_strength.unwrap_or(config.bleeding_strength);
    config.stabilize_phase = !no_stabilize_phase;
    config.phase_strength = phase_strength.unwrap_or(config.phase_strength);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(values: &[&str]) -> Cli {
        parse_args(values.iter().map(OsString::from)).unwrap()
    }

    #[test]
    fn process_default_output_preserves_repaired_suffix() {
        let input = Path::new("/tmp/song.wav");
        let output = default_output_path(input, ".repaired");
        assert_eq!(output, Path::new("/tmp/song.repaired.wav"));
        assert_ne!(output, input);
    }

    #[test]
    fn bare_audio_path_is_an_alias_for_process() {
        assert!(matches!(
            parse(&["sidespread", "song.wav"]).command,
            Command::Process { .. }
        ));
    }

    #[test]
    fn explicit_subcommands_are_not_rewritten() {
        assert!(matches!(
            parse(&["sidespread", "info", "song.wav"]).command,
            Command::Info { .. }
        ));
    }

    #[test]
    fn help_is_not_rewritten_as_an_audio_path() {
        let error = parse_args([OsString::from("sidespread"), OsString::from("--help")])
            .err()
            .expect("help should short-circuit parsing");
        assert_eq!(error.kind(), clap::error::ErrorKind::DisplayHelp);
    }

    #[test]
    fn process_defaults_to_dsp_and_exposes_optional_neural_modes() {
        match parse(&["sidespread", "process", "song.wav"]).command {
            Command::Process { mode, .. } => assert_eq!(mode, ProcessMode::Dsp),
            _ => panic!("expected process command"),
        }
        match parse(&["sidespread", "process", "song.wav", "--mode", "hybrid"]).command {
            Command::Process { mode, .. } => assert_eq!(mode, ProcessMode::Hybrid),
            _ => panic!("expected process command"),
        }
        assert!(
            Cli::try_parse_from(["sidespread", "process", "song.wav", "--mode", "auto"]).is_err()
        );
    }

    #[test]
    fn neural_controls_are_public_and_validated_later_by_config() {
        match parse(&[
            "sidespread",
            "process",
            "song.wav",
            "--mode",
            "nn",
            "--ode-steps",
            "4",
            "--model-path",
            "/tmp/universr.onnx",
            "--hybrid-neural-mix",
            "0.25",
            "--hybrid-neural-depth",
            "0.4",
            "--neural-max-hf-boost-db",
            "2.5",
        ])
        .command
        {
            Command::Process {
                mode,
                ode_steps,
                model_path,
                hybrid_neural_mix,
                hybrid_neural_depth,
                neural_max_hf_boost_db,
                ..
            } => {
                assert_eq!(mode, ProcessMode::Nn);
                assert_eq!(ode_steps, 4);
                assert_eq!(model_path, Some(PathBuf::from("/tmp/universr.onnx")));
                assert_eq!(hybrid_neural_mix, 0.25);
                assert_eq!(hybrid_neural_depth, 0.4);
                assert_eq!(neural_max_hf_boost_db, 2.5);
            }
            _ => panic!("expected process command"),
        }
    }

    #[test]
    fn json_report_is_opt_in() {
        match parse(&["sidespread", "process", "song.wav"]).command {
            Command::Process { output_report, .. } => assert!(output_report.is_none()),
            _ => panic!("expected process command"),
        }

        match parse(&[
            "sidespread",
            "process",
            "song.wav",
            "--output-report",
            "analysis.json",
        ])
        .command
        {
            Command::Process { output_report, .. } => {
                assert_eq!(output_report, Some(PathBuf::from("analysis.json")))
            }
            _ => panic!("expected process command"),
        }
    }

    #[test]
    fn artifact_opt_outs_and_strengths_are_independent() {
        match parse(&[
            "sidespread",
            "process",
            "song.wav",
            "--no-repair-smearing",
            "--bleeding-strength",
            "0.8",
            "--no-stabilize-phase",
        ])
        .command
        {
            Command::Process {
                no_repair_smearing,
                smearing_strength,
                no_repair_bleeding,
                bleeding_strength,
                no_stabilize_phase,
                phase_strength,
                ..
            } => {
                assert!(no_repair_smearing);
                assert_eq!(smearing_strength, None);
                assert!(!no_repair_bleeding);
                assert_eq!(bleeding_strength, Some(0.8));
                assert!(no_stabilize_phase);
                assert_eq!(phase_strength, None);
            }
            _ => panic!("expected process command"),
        }
    }

    #[test]
    fn process_artifact_repairs_default_on_and_can_be_disabled_independently() {
        let mut config = Config::from_process(8_000, 0.3);
        configure_artifact_processing(&mut config, true, None, false, Some(0.8), true, None);
        assert!(!config.repair_smearing);
        assert!(config.repair_bleeding);
        assert_eq!(config.bleeding_strength, 0.8);
        assert!(!config.stabilize_phase);
    }

    #[test]
    fn artifact_opt_out_conflicts_with_its_strength_override() {
        assert!(Cli::try_parse_from([
            "sidespread",
            "process",
            "song.wav",
            "--no-repair-smearing",
            "--smearing-strength",
            "0.4",
        ])
        .is_err());
    }

    #[test]
    fn force_is_explicitly_opt_in() {
        match parse(&["sidespread", "process", "song.wav"]).command {
            Command::Process { force, .. } => assert!(!force),
            _ => panic!("expected process command"),
        }
        match parse(&["sidespread", "process", "song.wav", "--force"]).command {
            Command::Process { force, .. } => assert!(force),
            _ => panic!("expected process command"),
        }
    }

    #[test]
    fn eval_default_output_preserves_eval_suffix() {
        let input = Path::new("/tmp/song.wav");
        let output = default_output_path(input, ".eval_repaired");
        assert_eq!(output, Path::new("/tmp/song.eval_repaired.wav"));
        assert_ne!(output, input);
    }
}
