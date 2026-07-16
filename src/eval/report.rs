//! JSON report generation and terminal summaries.

use crate::analysis::detector::SegmentReport;
use crate::eval::metrics::{EvaluationMetrics, ProcessingMetrics};
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
    println!("- sidespread report --------------------------------");
    println!("segments analyzed : {total}");
    println!("segments needing  : {needs}");

    if let Some(overall) = &report.overall {
        println!("- before / after ----------------------------------");
        println!(
            "r_hf  {:8.4} -> {:8.4}    lsd_hf {:8.4} -> {:8.4}",
            overall.before.r_hf, overall.after.r_hf, overall.before.lsd_hf, overall.after.lsd_hf
        );
        println!(
            "mcd   {:8.4} -> {:8.4}    iccc   {:8.4} -> {:8.4}",
            overall.before.mcd, overall.after.mcd, overall.before.iccc_hf, overall.after.iccc_hf
        );
        println!(
            "output gain: {:.2} dB    synthesis mix: {:.1}%",
            overall.output_gain_db,
            overall.synthesis_mix * 100.0
        );
    }

    if let Some(evaluation) = &report.evaluation {
        println!("- ground truth ------------------------------------");
        println!(
            "reference r_hf={:.4} lsd_hf={:.4} iccc={:.4}",
            evaluation.reference.r_hf, evaluation.reference.lsd_hf, evaluation.reference.iccc_hf
        );
        println!(
            "lsd_hf {:8.4} -> {:8.4}    snr_hf {} -> {}",
            evaluation.degraded.lsd_hf,
            evaluation.repaired.lsd_hf,
            display_optional(evaluation.degraded.snr_hf_db),
            display_optional(evaluation.repaired.snr_hf_db)
        );
        println!(
            "snr_db {} -> {}    snr_preserved {} -> {}",
            display_optional(evaluation.degraded.snr_db),
            display_optional(evaluation.repaired.snr_db),
            display_optional(evaluation.degraded.snr_preserved_db),
            display_optional(evaluation.repaired.snr_preserved_db)
        );
        println!(
            "existing HF projection: {} dB",
            display_optional(evaluation.existing_hf_projection_db)
        );
    }

    println!("- per-segment -------------------------------------");
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
            "[{:6}..{:6}] route={:<7} r_hf={:.3} lsd={:.3} corr_i={:.3} corr_t={:.3} r_t={:.4}",
            segment.start,
            segment.end,
            segment.route,
            segment.metrics.r_hf,
            segment.metrics.lsd_hf,
            segment.metrics.corr_intact,
            segment.metrics.corr_transition,
            segment.metrics.r_transition
        );
    }
    if total > 20 {
        println!("... ({}) segments omitted ...", total - 20);
    }
}

fn display_optional(value: Option<f32>) -> String {
    value
        .map(|number| format!("{number:.2}"))
        .unwrap_or_else(|| "n/a".to_string())
}
