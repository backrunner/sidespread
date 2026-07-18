//! JSON report generation and terminal summaries.

use crate::analysis::detector::SegmentReport;
use crate::eval::metrics::{EvaluationMetrics, ProcessingMetrics};
use crate::terminal::{self, Tone};
use anyhow::Result;
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Serialize)]
pub struct Report {
    pub needs_processing: bool,
    pub segments: Vec<SegmentReport>,
    pub overall: Option<ProcessingMetrics>,
    pub evaluation: Option<EvaluationMetrics>,
}

pub fn write_json<P: AsRef<Path>>(report: &Report, path: P) -> Result<()> {
    let serialized = serde_json::to_string_pretty(report)?;
    std::fs::write(path, serialized)?;
    Ok(())
}

pub fn print_summary(report: &Report) {
    let total = report.segments.len();
    let needs = report
        .segments
        .iter()
        .filter(|segment| segment.needs_processing)
        .count();
    terminal::header("SIDESPREAD", "analysis report");
    let (state, tone) = if report.needs_processing {
        ("REPAIR NEEDED", Tone::Yellow)
    } else {
        ("HEALTHY", Tone::Green)
    };
    println!("  {:<20} {}", "status", terminal::paint(state, tone));
    println!("  {:<20} {total}", "segments analyzed");
    println!("  {:<20} {needs}", "segments flagged");

    if let Some(overall) = &report.overall {
        terminal::section("Before / after");
        println!("  {:<20} {:>12} {:>12}", "metric", "before", "after");
        println!("  {}", "-".repeat(46));
        println!(
            "  {:<20} {:>12.4} {}",
            "side / mid HF",
            overall.before.r_hf,
            terminal::paint(format!("{:>12.4}", overall.after.r_hf), Tone::Cyan)
        );
        println!(
            "  {:<20} {:>12.4} {}",
            "HF spectral distance",
            overall.before.lsd_hf,
            terminal::paint(format!("{:>12.4}", overall.after.lsd_hf), Tone::Cyan)
        );
        println!(
            "  {:<20} {:>12.4} {}",
            "cepstral distance",
            overall.before.mcd,
            terminal::paint(format!("{:>12.4}", overall.after.mcd), Tone::Cyan)
        );
        println!(
            "  {:<20} {:>12.4} {}",
            "HF correlation",
            overall.before.iccc_hf,
            terminal::paint(format!("{:>12.4}", overall.after.iccc_hf), Tone::Cyan)
        );
        println!(
            "\n  output gain  {:+.2} dB    synthesis mix  {:.1}%",
            overall.output_gain_db,
            overall.synthesis_mix * 100.0
        );
    }

    if let Some(evaluation) = &report.evaluation {
        terminal::section("Ground truth evaluation");
        println!(
            "  reference  r_hf {:.4}   lsd_hf {:.4}   iccc {:.4}",
            evaluation.reference.r_hf, evaluation.reference.lsd_hf, evaluation.reference.iccc_hf
        );
        println!(
            "  lsd_hf     {:8.4} -> {:8.4}    snr_hf {} -> {}",
            evaluation.degraded.lsd_hf,
            evaluation.repaired.lsd_hf,
            display_optional(evaluation.degraded.snr_hf_db),
            display_optional(evaluation.repaired.snr_hf_db)
        );
        println!(
            "  snr_db {} -> {}    snr_preserved {} -> {}",
            display_optional(evaluation.degraded.snr_db),
            display_optional(evaluation.repaired.snr_db),
            display_optional(evaluation.degraded.snr_preserved_db),
            display_optional(evaluation.repaired.snr_preserved_db)
        );
        println!(
            "  existing HF projection  {} dB",
            display_optional(evaluation.existing_hf_projection_db)
        );
    }

    terminal::section("Segments");
    println!(
        "  {:>15}  {:<8} {:>7} {:>7} {:>6} {:>6} {:>7}",
        "sample range", "route", "R_hf", "LSD", "corr-I", "corr-T", "R-trans"
    );
    println!("  {}", "-".repeat(68));
    let shown = if total > 20 {
        report
            .segments
            .iter()
            .take(10)
            .chain(report.segments.iter().skip(total - 10))
            .collect::<Vec<_>>()
    } else {
        report.segments.iter().collect::<Vec<_>>()
    };
    for segment in shown {
        println!(
            "  {:6}..{:<6}  {} {:>7.3} {:>7.3} {:>6.3} {:>6.3} {:>7.4}",
            segment.start,
            segment.end,
            terminal::route_label(format!("{:<8}", segment.route)),
            segment.metrics.r_hf,
            segment.metrics.lsd_hf,
            segment.metrics.corr_intact,
            segment.metrics.corr_transition,
            segment.metrics.r_transition
        );
    }
    if total > 20 {
        println!(
            "  {}",
            terminal::paint(
                format!("... {} segments omitted ...", total - 20),
                Tone::Muted
            )
        );
    }
    println!();
}

fn display_optional(value: Option<f32>) -> String {
    value
        .map(|number| format!("{number:.2}"))
        .unwrap_or_else(|| "n/a".to_string())
}
