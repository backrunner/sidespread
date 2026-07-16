//! CLI definition and dispatch via clap.

use crate::config::Config;
use crate::pipeline;
use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "sidespread",
    version,
    about = "Repair high-frequency loss in the side channel of AI-generated stereo music"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Process a stereo WAV: detect, repair side HF, output fixed WAV + report.
    Process {
        #[arg(value_name = "INPUT")]
        input: PathBuf,
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,
        #[arg(long, value_name = "HZ", default_value_t = 8000)]
        fc: usize,
        #[arg(long, value_name = "THRESH", default_value_t = 0.3)]
        rhf_threshold: f32,
        #[arg(long, value_name = "PATH", default_value = "report.json")]
        report: PathBuf,
    },
    /// Detect only: report whether the audio needs processing and recommended route per segment.
    Detect {
        #[arg(value_name = "INPUT")]
        input: PathBuf,
        #[arg(long, value_name = "HZ", default_value_t = 8000)]
        fc: usize,
        #[arg(long, value_name = "THRESH", default_value_t = 0.3)]
        rhf_threshold: f32,
        #[arg(long, hide = true, default_value_t = 0.18)]
        rhf_relative_threshold: f32,
        #[arg(long, value_name = "PATH", default_value = "report.json")]
        report: PathBuf,
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
        #[arg(long, value_name = "PATH", default_value = "report.json")]
        report: PathBuf,
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
    #[command(hide = true)]
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
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
    let cli = Cli::parse();
    match cli.command {
        Command::Model { command } => match command {
            ModelCommand::Download { output, force } => crate::model::download(&output, force),
            ModelCommand::Verify { path } => crate::model::verify(&path),
        },
        Command::Info { input, fc } => pipeline::info(&input, fc),
        Command::Detect {
            input,
            fc,
            rhf_threshold,
            rhf_relative_threshold,
            report,
        } => {
            let mut cfg = Config::from_detect(fc, rhf_threshold);
            cfg.rhf_relative_threshold = rhf_relative_threshold;
            pipeline::detect(&input, &cfg, &report)
        }
        Command::Process {
            input,
            output,
            fc,
            rhf_threshold,
            report,
        } => {
            let out = output.unwrap_or_else(|| default_output_path(&input, ".repaired"));
            let cfg = Config::from_process(fc, rhf_threshold);
            pipeline::process(&input, &out, &cfg, &report)
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
            report,
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
            pipeline::eval(&clean, &out, &cfg, &report)
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_default_output_preserves_repaired_suffix() {
        let input = Path::new("/tmp/song.wav");
        let output = default_output_path(input, ".repaired");
        assert_eq!(output, Path::new("/tmp/song.repaired.wav"));
        assert_ne!(output, input);
    }

    #[test]
    fn process_does_not_expose_repair_modes() {
        assert!(
            Cli::try_parse_from(["sidespread", "process", "song.wav", "--mode", "nn"]).is_err()
        );
    }

    #[test]
    fn eval_default_output_preserves_eval_suffix() {
        let input = Path::new("/tmp/song.wav");
        let output = default_output_path(input, ".eval_repaired");
        assert_eq!(output, Path::new("/tmp/song.eval_repaired.wav"));
        assert_ne!(output, input);
    }
}
