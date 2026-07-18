//! JSON report generation and terminal summaries.

use crate::analysis::bandwidth::BandwidthReport;
use crate::analysis::detector::SegmentReport;
use crate::config::{Config, Route};
use crate::eval::metrics::{EvaluationMetrics, ProcessingMetrics};
use crate::terminal::{self, Tone};
use anyhow::Result;
use serde::Serialize;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Serialize)]
pub struct Report {
    pub needs_processing: bool,
    pub missing_band_processing: MissingBandProcessingReport,
    pub bandwidth: BandwidthReport,
    pub artifact_processing: ArtifactProcessingReport,
    pub segments: Vec<SegmentReport>,
    pub overall: Option<ProcessingMetrics>,
    pub evaluation: Option<EvaluationMetrics>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MissingBandProcessingReport {
    /// This route covers dynamic Side missing-band fill, not the full-track artifact stages.
    pub route_scope: &'static str,
    pub scan_start_hz: usize,
    pub band_width_hz: usize,
    pub detected_segments: usize,
    pub routed_segments: usize,
    pub multi_band_segments: usize,
    pub lowest_defect_hz: Option<f32>,
}

impl MissingBandProcessingReport {
    pub fn from_segments(config: &Config, segments: &[SegmentReport]) -> Self {
        Self {
            route_scope: "dynamic_side_missing_band_fill",
            scan_start_hz: config.scan_start_hz,
            band_width_hz: 500,
            detected_segments: segments
                .iter()
                .filter(|segment| segment.needs_processing)
                .count(),
            routed_segments: segments
                .iter()
                .filter(|segment| segment.route != Route::Skip)
                .count(),
            multi_band_segments: segments
                .iter()
                .filter(|segment| segment.needs_processing && segment.metrics.deficient_bands >= 2)
                .count(),
            lowest_defect_hz: segments
                .iter()
                .filter(|segment| segment.needs_processing)
                .filter_map(|segment| segment.metrics.repair_start_hz)
                .min_by(f32::total_cmp),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactProcessingReport {
    pub smearing: ArtifactSetting,
    pub harmonic_bleeding: ArtifactSetting,
    pub phase_incoherence: ArtifactSetting,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactSetting {
    pub enabled: bool,
    /// User-requested algorithm strength.
    pub strength: f32,
    /// Fraction retained by the runtime quality gate.
    pub retained_mix: f32,
    /// Effective strength after automatic safety backoff.
    pub applied_strength: f32,
}

impl ArtifactProcessingReport {
    pub fn from_config(config: &Config) -> Self {
        Self {
            smearing: ArtifactSetting {
                enabled: config.repair_smearing && config.smearing_strength > 0.0,
                strength: config.smearing_strength,
                retained_mix: 0.0,
                applied_strength: 0.0,
            },
            harmonic_bleeding: ArtifactSetting {
                enabled: config.repair_bleeding && config.bleeding_strength > 0.0,
                strength: config.bleeding_strength,
                retained_mix: 0.0,
                applied_strength: 0.0,
            },
            phase_incoherence: ArtifactSetting {
                enabled: config.stabilize_phase && config.phase_strength > 0.0,
                strength: config.phase_strength,
                retained_mix: 0.0,
                applied_strength: 0.0,
            },
        }
    }

    fn any_enabled(&self) -> bool {
        self.smearing.enabled || self.harmonic_bleeding.enabled || self.phase_incoherence.enabled
    }

    pub fn with_retained_mixes(
        mut self,
        smearing_mix: f32,
        bleeding_mix: f32,
        phase_mix: f32,
    ) -> Self {
        set_retained_mix(&mut self.smearing, smearing_mix);
        set_retained_mix(&mut self.harmonic_bleeding, bleeding_mix);
        set_retained_mix(&mut self.phase_incoherence, phase_mix);
        self
    }
}

pub fn write_json<P: AsRef<Path>>(report: &Report, path: P) -> Result<()> {
    let serialized = serde_json::to_string_pretty(report)?;
    std::fs::write(path, serialized)?;
    Ok(())
}

pub fn print_summary(report: &Report) {
    let total = report.segments.len();
    let missing_band = &report.missing_band_processing;
    let needs = missing_band.detected_segments;
    terminal::header("SIDESPREAD", "analysis report");
    let detected_repair = needs > 0 || report.bandwidth.needs_extension;
    let (state, tone) = if detected_repair {
        ("REPAIR NEEDED", Tone::Yellow)
    } else if report.artifact_processing.any_enabled() {
        ("OPTIONAL PROCESSING", Tone::Cyan)
    } else {
        ("HEALTHY", Tone::Green)
    };
    println!("  {:<20} {}", "status", terminal::paint(state, tone));
    println!("  {:<20} {total}", "segments analyzed");
    println!("  {:<20} {needs}", "band defects");
    println!(
        "  {:<20} {}",
        "band repairs routed", missing_band.routed_segments
    );
    println!(
        "  {:<20} {}",
        "multi-band defects", missing_band.multi_band_segments
    );
    if let Some(frequency) = missing_band.lowest_defect_hz {
        println!("  {:<20} {frequency:.0} Hz", "lowest defect");
    }
    if report.bandwidth.needs_extension {
        println!(
            "  {:<20} {:.0} Hz ({:.0}% confidence)",
            "shared HF cutoff",
            report.bandwidth.detected_cutoff_hz.unwrap_or_default(),
            report.bandwidth.confidence * 100.0
        );
    } else {
        println!("  {:<20} none", "shared HF cutoff");
    }
    if report.artifact_processing.any_enabled() {
        terminal::section("Artifact repair");
        print_artifact_setting("HF smearing", &report.artifact_processing.smearing);
        print_artifact_setting(
            "harmonic bleeding",
            &report.artifact_processing.harmonic_bleeding,
        );
        print_artifact_setting(
            "phase incoherence",
            &report.artifact_processing.phase_incoherence,
        );
    }

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

    terminal::section("Dynamic side-band scan");
    println!(
        "  scan >= {} Hz in {} Hz bands; route applies only to missing-band fill",
        missing_band.scan_start_hz, missing_band.band_width_hz
    );
    println!(
        "  {:>15}  {:<6} {:<10} {:>8} {:>9} {:>10} {:>7}",
        "sample range", "defect", "band route", "bands", "start Hz", "continuity", "R_hf"
    );
    println!("  {}", "-".repeat(80));
    let shown = shown_segments(&report.segments, 20);
    let omitted = total.saturating_sub(shown.len());
    for segment in shown {
        let defect = if segment.needs_processing {
            "yes"
        } else {
            "no"
        };
        let bands = format!(
            "D{} A{}",
            segment.metrics.deficient_bands, segment.metrics.active_bands
        );
        let start_hz = segment
            .metrics
            .repair_start_hz
            .map(|frequency| format!("{frequency:.0}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "  {:6}..{:<6}  {:<6} {} {:>8} {:>9} {:>10.3} {:>7.3}",
            segment.start,
            segment.end,
            defect,
            terminal::route_label(format!("{:<10}", segment.route)),
            bands,
            start_hz,
            segment.metrics.band_continuity,
            segment.metrics.r_hf
        );
    }
    if omitted > 0 {
        println!(
            "  {}",
            terminal::paint(format!("... {omitted} segments omitted ..."), Tone::Muted)
        );
    }
    println!();
    std::io::stdout().flush().ok();
}

fn shown_segments(segments: &[SegmentReport], limit: usize) -> Vec<&SegmentReport> {
    if segments.len() <= limit || limit == 0 {
        return segments.iter().take(limit).collect();
    }

    let mut indices = (0..3.min(segments.len())).collect::<Vec<_>>();
    indices.extend(segments.len().saturating_sub(3)..segments.len());
    let routed = segments
        .iter()
        .enumerate()
        .filter_map(|(index, segment)| (segment.route != Route::Skip).then_some(index))
        .collect::<Vec<_>>();
    let detected_skip = segments
        .iter()
        .enumerate()
        .filter_map(|(index, segment)| {
            (segment.needs_processing && segment.route == Route::Skip).then_some(index)
        })
        .collect::<Vec<_>>();
    indices.extend(evenly_spaced(&routed, 8));
    indices.extend(evenly_spaced(&detected_skip, 4));
    indices.sort_unstable();
    indices.dedup();

    let remaining = limit.saturating_sub(indices.len());
    indices.extend(evenly_spaced(
        &(0..segments.len()).collect::<Vec<_>>(),
        remaining,
    ));
    indices.sort_unstable();
    indices.dedup();
    indices.truncate(limit);
    indices.into_iter().map(|index| &segments[index]).collect()
}

fn evenly_spaced(indices: &[usize], limit: usize) -> Vec<usize> {
    if limit == 0 || indices.is_empty() {
        return Vec::new();
    }
    if indices.len() <= limit {
        return indices.to_vec();
    }
    if limit == 1 {
        return vec![indices[indices.len() / 2]];
    }
    (0..limit)
        .map(|position| indices[position * (indices.len() - 1) / (limit - 1)])
        .collect()
}

fn display_optional(value: Option<f32>) -> String {
    value
        .map(|number| format!("{number:.2}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn print_artifact_setting(label: &str, setting: &ArtifactSetting) {
    if setting.enabled {
        if setting.retained_mix > 0.0 {
            println!(
                "  {label:<20} {:.0}% -> {:.0}% after safety",
                setting.strength * 100.0,
                setting.applied_strength * 100.0
            );
        } else {
            println!(
                "  {label:<20} bypassed by safety (requested {:.0}%)",
                setting.strength * 100.0
            );
        }
    }
}

fn set_retained_mix(setting: &mut ArtifactSetting, retained_mix: f32) {
    setting.retained_mix = retained_mix.clamp(0.0, 1.0);
    setting.applied_strength = setting.strength * setting.retained_mix;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::detector::SegmentMetrics;

    fn segment(index: usize, route: Route) -> SegmentReport {
        SegmentReport {
            start: index * 100,
            end: (index + 1) * 100,
            needs_processing: route != Route::Skip,
            route,
            metrics: SegmentMetrics {
                r_hf: 0.1,
                r_hf_low: 0.1,
                lsd_hf: 1.0,
                corr_hf: 0.0,
                r_intact: 0.2,
                corr_intact: 0.0,
                corr_transition: 0.0,
                r_transition: 0.0,
                band_continuity: 0.2,
                deficient_bands: usize::from(route != Route::Skip) * 2,
                active_bands: 10,
                repair_start_hz: (route != Route::Skip).then_some(5_500.0),
            },
        }
    }

    #[test]
    fn segment_sampling_includes_repairs_from_the_middle() {
        let mut segments = (0..100)
            .map(|index| segment(index, Route::Skip))
            .collect::<Vec<_>>();
        segments[43] = segment(43, Route::Dsp);
        segments[67] = segment(67, Route::Dsp);

        let shown = shown_segments(&segments, 20);

        assert!(shown.iter().any(|segment| segment.start == 4_300));
        assert!(shown.iter().any(|segment| segment.start == 6_700));
        assert!(shown.len() <= 20);
    }
}
