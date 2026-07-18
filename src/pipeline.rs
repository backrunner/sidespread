//! Pipeline orchestration for the CLI subcommands.

use crate::analysis::detector::SegmentReport;
use crate::analysis::segment::{segments, select_safe_boundary, AdaptiveBoundary, Segment};
use crate::analysis::{self};
use crate::config::{Config, Mode, Route};
use crate::eval::metrics::{
    compare_reference, high_band_projection_db, quality_metrics, EvaluationMetrics, MetricConfig,
    ProcessingMetrics,
};
use crate::eval::report::{
    print_summary, write_json, ArtifactProcessingReport, MissingBandProcessingReport, Report,
};
use crate::eval::synthetic;
use crate::io::{lr_to_ms, ms_to_lr, read_wav, write_wav, AudioBuffer};
use crate::repair;
use crate::terminal::{self, Tone};
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::path::{Component, Path, PathBuf};

pub fn info<P: AsRef<Path>>(input: P, fc: usize) -> Result<()> {
    let input = input.as_ref();
    let buffer = read_wav(input).context("reading wav")?;
    ensure_nonempty(&buffer)?;
    let config = Config {
        fc,
        ..Config::default()
    };
    config.validate(buffer.sample_rate)?;
    let (m, s) = crate::io::mside::lr_to_ms(&buffer)?;
    let bandwidth = analysis::bandwidth::analyze(&m, &s, &config, buffer.sample_rate);
    let (_, segment_reports) = analyze_all(&m, &s, buffer.sample_rate, &config);
    let missing_band = MissingBandProcessingReport::from_segments(&config, &segment_reports);

    let stft_config = crate::analysis::stft::StftConfig::new(config.n_fft, config.hop);
    let metrics = crate::analysis::detector::compute_metrics(
        &crate::analysis::stft::stft(&m, &stft_config),
        &crate::analysis::stft::stft(&s, &stft_config),
        fc,
        buffer.sample_rate,
        config.n_fft,
    );

    terminal::header("SIDESPREAD", "audio inspection");
    terminal::section("File");
    println!("  {:<18} {}", "path", input.display());
    println!("  {:<18} {} Hz", "sample rate", buffer.sample_rate);
    println!("  {:<18} {}", "channels", buffer.channels());
    println!(
        "  {:<18} {}-bit {:?}",
        "format", buffer.bits_per_sample, buffer.sample_format
    );
    println!("  {:<18} {}", "frames", buffer.frames());
    println!("  {:<18} {:.3} s", "duration", buffer.duration_secs());

    terminal::section(&format!("M/S high-frequency analysis - {fc} Hz cutoff"));
    println!("  {:<18} {:>10}", "metric", "value");
    println!("  {}", "-".repeat(30));
    println!("  {:<18} {:>10.4}", "side / mid HF", metrics.r_hf);
    println!("  {:<18} {:>10.4} dB", "spectral distance", metrics.lsd_hf);
    println!("  {:<18} {:>10.4}", "HF correlation", metrics.corr_hf);
    println!("  {:<18} {:>10.4}", "intact ratio", metrics.r_intact);
    println!(
        "  {:<18} {:>10.4}",
        "intact correlation", metrics.corr_intact
    );
    println!(
        "  {:<18} {:>10.6}",
        "transition ratio", metrics.r_transition
    );
    println!(
        "  {:<18} {:>10.4}",
        "transition corr.", metrics.corr_transition
    );
    println!(
        "  {:<18} {:>10}",
        "band defects", missing_band.detected_segments
    );
    if let Some(frequency) = missing_band.lowest_defect_hz {
        println!("  {:<18} {:>10.0} Hz", "lowest defect", frequency);
    }
    if let Some(cutoff) = bandwidth.detected_cutoff_hz {
        println!("  {:<18} {:>10.0} Hz", "shared HF cutoff", cutoff);
    } else {
        println!("  {:<18} {:>10}", "shared HF cutoff", "none");
    }

    let needs_repair = missing_band.detected_segments > 0 || bandwidth.needs_extension;
    let (state, detail, tone) = if needs_repair {
        (
            "REPAIR",
            "High-frequency detail is deficient. Run `sidespread process`.",
            Tone::Yellow,
        )
    } else {
        (
            "HEALTHY",
            "High-frequency detail looks healthy.",
            Tone::Green,
        )
    };
    println!(
        "\n  {} {detail}\n",
        terminal::paint(format!("[{state}]"), tone)
    );
    Ok(())
}

pub fn detect<P: AsRef<Path>>(input: P, config: &Config, report_path: Option<&Path>) -> Result<()> {
    let input = input.as_ref();
    ensure_report_destination(input, None, report_path)?;
    terminal::status("READ", input.display(), Tone::Cyan);
    let buffer = read_wav(input).context("reading wav")?;
    ensure_nonempty(&buffer)?;
    config.validate(buffer.sample_rate)?;
    let (m, s) = crate::io::mside::lr_to_ms(&buffer)?;
    terminal::status(
        "ANALYZE",
        "measuring side detail and shared bandwidth",
        Tone::Cyan,
    );
    let (_, reports) = analyze_all(&m, &s, buffer.sample_rate, config);
    let bandwidth = analysis::bandwidth::analyze(&m, &s, config, buffer.sample_rate);
    let report = Report {
        needs_processing: reports.iter().any(|segment| segment.needs_processing)
            || bandwidth.needs_extension,
        missing_band_processing: MissingBandProcessingReport::from_segments(config, &reports),
        bandwidth,
        artifact_processing: ArtifactProcessingReport::from_config(config),
        segments: reports,
        overall: None,
        evaluation: None,
    };
    print_summary(&report);
    write_optional_report(&report, report_path, config.overwrite_existing)?;
    if report.needs_processing {
        terminal::status(
            "REPAIR",
            "audio needs processing; run `sidespread process` to repair it",
            Tone::Yellow,
        );
    } else {
        terminal::status("HEALTHY", "audio does not need processing", Tone::Green);
    }
    Ok(())
}

pub fn process<P: AsRef<Path>, Q: AsRef<Path>>(
    input: P,
    output: Q,
    config: &Config,
    report_path: Option<&Path>,
) -> Result<()> {
    let input = input.as_ref();
    let output = output.as_ref();
    ensure_report_destination(input, Some(output), report_path)?;
    terminal::status("READ", input.display(), Tone::Cyan);
    let buffer = read_wav(input).context("reading wav")?;
    ensure_nonempty(&buffer)?;
    config.validate(buffer.sample_rate)?;
    let (m, s) = crate::io::mside::lr_to_ms(&buffer)?;
    terminal::status("ANALYZE", "finding repair candidates", Tone::Cyan);
    let (segment_ranges, reports) = analyze_all(&m, &s, buffer.sample_rate, config);
    let bandwidth_progress = terminal::TaskProgress::new("BANDWIDTH SCAN", 1, "steps");
    let bandwidth = analysis::bandwidth::analyze(&m, &s, config, buffer.sample_rate);
    bandwidth_progress.finish();
    let needs_processing = reports.iter().any(|segment| segment.needs_processing)
        || bandwidth.needs_extension
        || config.artifact_processing_enabled();
    let has_side_repair_route = reports.iter().any(|segment| segment.route != Route::Skip);
    let has_repair_route =
        has_side_repair_route || bandwidth.needs_extension || config.artifact_processing_enabled();
    let before = quality_for_side(&buffer, &m, &s, config);
    let will_attempt_wav = needs_processing && config.mode != Mode::Skip && has_repair_route;
    let approved = if will_attempt_wav {
        approve_output_set(output, report_path, config.overwrite_existing)?
    } else if let Some(report) = report_path {
        approve_overwrite(report, config.overwrite_existing)?
    } else {
        true
    };
    if !approved {
        terminal::status(
            "CANCEL",
            "existing result files were left unchanged",
            Tone::Yellow,
        );
        return Ok(());
    }

    if !needs_processing || config.mode == Mode::Skip || !has_repair_route {
        let report = Report {
            needs_processing,
            missing_band_processing: MissingBandProcessingReport::from_segments(config, &reports),
            bandwidth,
            artifact_processing: ArtifactProcessingReport::from_config(config),
            segments: reports,
            overall: Some(ProcessingMetrics {
                before: before.clone(),
                after: before,
                output_gain_db: 0.0,
                synthesis_mix: 0.0,
            }),
            evaluation: None,
        };
        print_summary(&report);
        write_optional_report(&report, report_path, true)?;
        if needs_processing {
            if config.mode == Mode::Skip {
                terminal::status("SKIP", "repair disabled; no WAV was written", Tone::Yellow);
            } else {
                terminal::status(
                    "SKIP",
                    "no segments met repair confidence; no WAV was written",
                    Tone::Yellow,
                );
            }
        } else {
            terminal::status(
                "HEALTHY",
                "audio does not need processing; no WAV was written",
                Tone::Green,
            );
        }
        return Ok(());
    }

    terminal::status("REPAIR", "rebuilding high-frequency detail", Tone::Yellow);
    let repaired_side = if has_side_repair_route {
        repair_segments(
            &m,
            &s,
            &segment_ranges,
            &reports,
            config,
            buffer.sample_rate,
        )?
    } else {
        let label = match config.mode {
            Mode::Nn => "UNIVERSR + GUARD",
            Mode::Hybrid => "DSP / UNIVERSR",
            _ => "DSP REPAIR",
        };
        terminal::status(label, "not needed", Tone::Muted);
        s.clone()
    };
    let (repaired_mid, repaired_side) = if let Some(cutoff) = bandwidth.detected_cutoff_hz {
        repair::bandwidth::repair_pair_with_progress(
            &m,
            &repaired_side,
            cutoff,
            config,
            buffer.sample_rate,
        )
    } else {
        terminal::status("BANDWIDTH EXTEND", "not needed", Tone::Muted);
        (m.clone(), repaired_side)
    };
    let artifact_outcome = repair::artifacts::process_with_progress(
        &repaired_mid,
        &repaired_side,
        config,
        buffer.sample_rate,
    );
    let artifact_processing = ArtifactProcessingReport::from_config(config).with_retained_mixes(
        artifact_outcome.smearing_mix,
        artifact_outcome.bleeding_mix,
        artifact_outcome.phase_mix,
    );
    let artifact_applied = artifact_outcome.smearing_mix > 0.0
        || artifact_outcome.bleeding_mix > 0.0
        || artifact_outcome.phase_mix > 0.0;
    if !has_side_repair_route && !bandwidth.needs_extension && !artifact_applied {
        let report = Report {
            needs_processing,
            missing_band_processing: MissingBandProcessingReport::from_segments(config, &reports),
            bandwidth,
            artifact_processing,
            segments: reports,
            overall: Some(ProcessingMetrics {
                before: before.clone(),
                after: before,
                output_gain_db: 0.0,
                synthesis_mix: 0.0,
            }),
            evaluation: None,
        };
        print_summary(&report);
        write_optional_report(&report, report_path, true)?;
        terminal::status(
            "SKIP",
            "artifact repairs did not pass the fidelity gate; no WAV was written",
            Tone::Yellow,
        );
        return Ok(());
    }
    let headroom_progress = terminal::TaskProgress::new("HEADROOM", 1, "checks");
    let (output_mid, output_side, output_gain_db, synthesis_mix) =
        fit_output_headroom(&m, &s, &artifact_outcome.mid, &artifact_outcome.side);
    headroom_progress.finish();
    let output_buffer = ms_to_lr(&output_mid, &output_side, &buffer);
    terminal::status("WRITE", output.display(), Tone::Cyan);
    let write_progress = terminal::TaskProgress::new("WRITE", 4, "steps");
    write_wav(output, &output_buffer).context("writing output wav")?;
    write_progress.advance(1);
    let written_buffer = read_wav(output).context("reading written output wav")?;
    write_progress.advance(1);
    let (written_m, written_side) = lr_to_ms(&written_buffer)?;
    write_progress.advance(1);
    let after = quality_for_side(&written_buffer, &written_m, &written_side, config);
    write_progress.advance(1);
    write_progress.finish();

    let report = Report {
        needs_processing,
        missing_band_processing: MissingBandProcessingReport::from_segments(config, &reports),
        bandwidth,
        artifact_processing,
        segments: reports,
        overall: Some(ProcessingMetrics {
            before,
            after,
            output_gain_db,
            synthesis_mix,
        }),
        evaluation: None,
    };
    print_summary(&report);
    write_optional_report(&report, report_path, true)?;
    terminal::status("DONE", output.display(), Tone::Green);
    Ok(())
}

pub fn eval<P: AsRef<Path>, Q: AsRef<Path>>(
    clean: P,
    output: Q,
    config: &Config,
    report_path: Option<&Path>,
) -> Result<()> {
    let clean = clean.as_ref();
    let output = output.as_ref();
    ensure_report_destination(clean, Some(output), report_path)?;
    terminal::status("READ", clean.display(), Tone::Cyan);
    let buffer = read_wav(clean).context("reading clean wav")?;
    ensure_nonempty(&buffer)?;
    config.validate(buffer.sample_rate)?;
    let (original_mid, original_side) = lr_to_ms(&buffer)?;
    let synthetic_degraded_side =
        synthetic::degrade_side(&original_side, buffer.sample_rate, config.fc as f32);
    let degraded_buffer = ms_to_lr(&original_mid, &synthetic_degraded_side, &buffer);
    let (degraded_mid, degraded_side) = lr_to_ms(&degraded_buffer)?;
    let (segment_ranges, reports) =
        analyze_all(&degraded_mid, &degraded_side, buffer.sample_rate, config);
    let bandwidth =
        analysis::bandwidth::analyze(&degraded_mid, &degraded_side, config, buffer.sample_rate);
    let side_repaired = if config.mode == Mode::Skip {
        degraded_side.clone()
    } else {
        repair_segments(
            &degraded_mid,
            &degraded_side,
            &segment_ranges,
            &reports,
            config,
            buffer.sample_rate,
        )?
    };
    let (repaired_mid, repaired_side) = if config.mode != Mode::Skip {
        if let Some(cutoff) = bandwidth.detected_cutoff_hz {
            repair::bandwidth::repair_pair(
                &degraded_mid,
                &side_repaired,
                cutoff,
                config,
                buffer.sample_rate,
            )
        } else {
            (degraded_mid.clone(), side_repaired)
        }
    } else {
        (degraded_mid.clone(), side_repaired)
    };
    if !approve_output_set(output, report_path, config.overwrite_existing)? {
        terminal::status(
            "CANCEL",
            "existing result files were left unchanged",
            Tone::Yellow,
        );
        return Ok(());
    }
    let (output_mid, output_side, output_gain_db, synthesis_mix) =
        fit_output_headroom(&degraded_mid, &degraded_side, &repaired_mid, &repaired_side);
    let output_buffer = ms_to_lr(&output_mid, &output_side, &buffer);
    write_wav(output, &output_buffer).context("writing output wav")?;
    let written_buffer = read_wav(output).context("reading written output wav")?;
    let (written_mid, written_side) = lr_to_ms(&written_buffer)?;

    let report = Report {
        needs_processing: reports.iter().any(|segment| segment.needs_processing)
            || bandwidth.needs_extension,
        missing_band_processing: MissingBandProcessingReport::from_segments(config, &reports),
        bandwidth,
        artifact_processing: ArtifactProcessingReport::from_config(config),
        segments: reports,
        overall: Some(ProcessingMetrics {
            before: quality_for_side(&degraded_buffer, &degraded_mid, &degraded_side, config),
            after: quality_for_side(&written_buffer, &written_mid, &written_side, config),
            output_gain_db,
            synthesis_mix,
        }),
        evaluation: Some(EvaluationMetrics {
            reference: quality_for_side(&buffer, &original_mid, &original_side, config),
            degraded: compare_reference(
                &original_side,
                &degraded_side,
                metric_config(config, buffer.sample_rate),
            ),
            repaired: compare_reference(
                &original_side,
                &written_side,
                metric_config(config, buffer.sample_rate),
            ),
            existing_hf_projection_db: high_band_projection_db(
                &degraded_side,
                &written_side,
                metric_config(config, buffer.sample_rate),
            ),
        }),
    };
    print_summary(&report);
    write_optional_report(&report, report_path, true)?;
    terminal::status("DONE", output.display(), Tone::Green);
    Ok(())
}

fn write_optional_report(report: &Report, path: Option<&Path>, force: bool) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    if !approve_overwrite(path, force)? {
        terminal::status(
            "CANCEL",
            format!("kept existing {}", path.display()),
            Tone::Yellow,
        );
        return Ok(());
    }
    write_json(report, path).context("writing report")?;
    terminal::status("REPORT", path.display(), Tone::Green);
    Ok(())
}

fn approve_output_set(output: &Path, report: Option<&Path>, force: bool) -> Result<bool> {
    if report.is_some_and(|report| destinations_match(report, output)) {
        bail!("output WAV and JSON report must use different paths");
    }
    if !approve_overwrite(output, force)? {
        return Ok(false);
    }
    if let Some(report) = report {
        if !approve_overwrite(report, force)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn ensure_report_destination(
    input: &Path,
    output: Option<&Path>,
    report: Option<&Path>,
) -> Result<()> {
    let Some(report) = report else {
        return Ok(());
    };
    if destinations_match(report, input) {
        bail!("input WAV and JSON report must use different paths");
    }
    if output.is_some_and(|output| destinations_match(report, output)) {
        bail!("output WAV and JSON report must use different paths");
    }
    Ok(())
}

fn destinations_match(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    resolved_destination(left)
        .zip(resolved_destination(right))
        .is_some_and(|(left, right)| left == right)
}

fn resolved_destination(path: &Path) -> Option<PathBuf> {
    if let Ok(path) = std::fs::canonicalize(path) {
        return Some(path);
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if let (Ok(parent), Some(file_name)) = (std::fs::canonicalize(parent), path.file_name()) {
        return Some(parent.join(file_name));
    }

    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            component => normalized.push(component.as_os_str()),
        }
    }
    Some(normalized)
}

fn approve_overwrite(path: &Path, force: bool) -> Result<bool> {
    if force || !path.exists() {
        return Ok(true);
    }
    if !terminal::interactive() {
        bail!(
            "{} already exists; rerun with --force to overwrite it",
            path.display()
        );
    }
    terminal::confirm_default_no(&format!(
        "{} already exists. Overwrite it? [y/N]",
        path.display()
    ))
    .context("reading overwrite confirmation")
}

fn analyze_all(
    m: &[f32],
    s: &[f32],
    sample_rate: u32,
    config: &Config,
) -> (Vec<Segment>, Vec<SegmentReport>) {
    let ranges = segments(m.len(), sample_rate, config.segment_ms, config.overlap);
    let progress = terminal::TaskProgress::new("ANALYZE", ranges.len(), "windows");
    let mut reports = ranges
        .par_iter()
        .map(|segment| {
            let report = analysis::analyze(
                &m[segment.start..segment.end],
                &s[segment.start..segment.end],
                segment.start,
                segment.end,
                config,
                sample_rate,
            );
            progress.advance(1);
            report
        })
        .collect::<Vec<_>>();
    stabilize_multiband_detection(&mut reports);
    progress.finish();
    smooth_route_correlation(&mut reports, config, 9);
    (ranges, reports)
}

fn stabilize_multiband_detection(reports: &mut [SegmentReport]) {
    if reports.is_empty() {
        return;
    }
    let mut reference = reports
        .iter()
        .filter_map(|report| {
            report
                .metrics
                .band_continuity
                .is_finite()
                .then_some(report.metrics.band_continuity)
        })
        .collect::<Vec<_>>();
    reference.sort_by(f32::total_cmp);
    let reference = reference
        .get(((reference.len().saturating_sub(1)) as f32 * 0.75).round() as usize)
        .copied()
        .unwrap_or(1.0);
    let soft_threshold = (reference * 0.65).clamp(0.03, 0.55);
    let soft = reports
        .iter()
        .map(|report| {
            report.metrics.active_bands >= 8
                && report.metrics.deficient_bands >= 2
                && report.metrics.band_continuity < soft_threshold
        })
        .collect::<Vec<_>>();
    let radius = 2usize;
    let sustained = (0..reports.len())
        .map(|index| {
            let start = index.saturating_sub(radius);
            let end = (index + radius + 1).min(reports.len());
            soft[start..end].iter().filter(|value| **value).count() >= 3
        })
        .collect::<Vec<_>>();
    for (report, sustained) in reports.iter_mut().zip(&sustained) {
        report.needs_processing |= *sustained;
    }

    let marked = reports
        .iter()
        .enumerate()
        .filter_map(|(index, report)| report.needs_processing.then_some(index))
        .collect::<Vec<_>>();
    for pair in marked.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if right <= left + 1 || right - left > 12 {
            continue;
        }
        let bridge = &reports[left + 1..right];
        let near_defect = bridge
            .iter()
            .filter(|report| {
                report.metrics.active_bands >= 8
                    && report.metrics.deficient_bands > 0
                    && report.metrics.band_continuity < soft_threshold * 1.25
            })
            .count();
        if near_defect * 2 >= bridge.len() {
            for report in &mut reports[left + 1..right] {
                report.needs_processing = true;
            }
        }
    }

    let marked_after_bridge = reports
        .iter()
        .enumerate()
        .filter_map(|(index, report)| report.needs_processing.then_some(index))
        .collect::<Vec<_>>();
    for index in marked_after_bridge {
        let start = index.saturating_sub(2);
        let end = (index + 3).min(reports.len());
        for report in &mut reports[start..end] {
            if report.metrics.active_bands >= 8
                && report.metrics.deficient_bands > 0
                && report.metrics.band_continuity < soft_threshold * 1.35
            {
                report.needs_processing = true;
            }
        }
    }
}

fn smooth_route_correlation(reports: &mut [SegmentReport], config: &Config, window: usize) {
    if reports.is_empty() {
        return;
    }
    let radius = window / 2;
    let smoothed = reports
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let start = index.saturating_sub(radius);
            let end = (index + radius + 1).min(reports.len());
            let count = (end - start) as f32;
            let intact = reports[start..end]
                .iter()
                .map(|report| report.metrics.corr_intact)
                .sum::<f32>()
                / count;
            let transition = reports[start..end]
                .iter()
                .map(|report| report.metrics.corr_transition)
                .sum::<f32>()
                / count;
            let transition_ratio = reports[start..end]
                .iter()
                .map(|report| report.metrics.r_transition)
                .sum::<f32>()
                / count;
            (intact, transition, transition_ratio)
        })
        .collect::<Vec<_>>();
    for (report, (intact, transition, transition_ratio)) in reports.iter_mut().zip(smoothed) {
        report.metrics.corr_intact = intact;
        report.metrics.corr_transition = transition;
        report.metrics.r_transition = transition_ratio;
        report.route = config.decide_with_deficiency(
            report.needs_processing,
            intact,
            transition,
            transition_ratio,
            report.metrics.r_hf,
            report.metrics.r_hf_low,
            report.metrics.r_intact,
        );
    }
}

fn repair_segments(
    m: &[f32],
    s: &[f32],
    ranges: &[Segment],
    reports: &[SegmentReport],
    config: &Config,
    sample_rate: u32,
) -> Result<Vec<f32>> {
    if ranges.len() != reports.len() {
        bail!("segment ranges and reports have different lengths");
    }

    let (repair_spans, neural_target_spans) = if config.mode == Mode::Hybrid {
        let (primary_reports, neural_reports) = split_hybrid_reports(reports);
        (
            adaptive_repair_spans(m, s, ranges, &primary_reports, sample_rate),
            adaptive_repair_spans(m, s, ranges, &neural_reports, sample_rate),
        )
    } else {
        let spans = adaptive_repair_spans(m, s, ranges, reports, sample_rate);
        let neural = spans.clone();
        (spans, neural)
    };
    let neural_spans = neural_context_spans(&neural_target_spans, sample_rate as usize / 4);
    let neural_candidate = if neural_spans.is_empty() {
        None
    } else {
        let mut state = repair::neural::NeuralState::from_config(config)
            .context("loading UniverSR repair state")?;
        let mut candidate = s.to_vec();
        let total = neural_spans.len();
        let total_chunks = neural_spans
            .iter()
            .map(|span| state.chunk_count(span.end - span.start, sample_rate))
            .sum::<usize>();
        let progress = terminal::TaskProgress::new(
            "UNIVERSR + GUARD",
            total_chunks.saturating_add(total),
            "steps",
        );
        let mut limited_spans = 0usize;
        for span in neural_spans {
            let repaired = state
                .repair_signal_with_progress(&s[span.start..span.end], sample_rate, || {
                    progress.advance(1);
                })
                .context("UniverSR repair failed")?;
            if repaired.len() != span.end - span.start {
                bail!("neural repair returned an unexpected routed span length");
            }
            let guarded = repair::neural::guard_candidate(
                &m[span.start..span.end],
                &s[span.start..span.end],
                &repaired,
                config,
                sample_rate,
            );
            if guarded.minimum_retained_mix < 0.999 {
                limited_spans += 1;
            }
            candidate[span.start..span.end].copy_from_slice(&guarded.signal);
            progress.advance(1);
        }
        progress.finish();
        if limited_spans > 0 {
            terminal::status(
                "GUARD",
                format!("limited neural HF energy in {limited_spans}/{total} routed spans"),
                Tone::Green,
            );
        }
        Some(candidate)
    };
    let mut output = s.to_vec();
    let primary_label = if config.mode == Mode::Nn {
        "NEURAL MERGE"
    } else {
        "DSP REPAIR"
    };
    let primary_uses_dsp = config.mode != Mode::Nn;
    let progress_total = if primary_uses_dsp {
        let settings = crate::analysis::stft::StftConfig::new(config.n_fft, config.hop);
        repair_spans
            .iter()
            .map(|span| settings.num_frames(span.end - span.start))
            .sum()
    } else {
        repair_spans.len()
    };
    let progress_unit = if primary_uses_dsp { "frames" } else { "spans" };
    let repair_progress = terminal::TaskProgress::new(primary_label, progress_total, progress_unit);
    for span in repair_spans {
        let mid = &m[span.start..span.end];
        let original = &s[span.start..span.end];
        let repaired = match span.route {
            Route::Skip => continue,
            Route::Dsp => {
                conservative_dsp_repair(mid, original, config, sample_rate, Some(&repair_progress))
            }
            Route::Neural => neural_candidate.as_ref().expect("neural candidate exists")
                [span.start..span.end]
                .to_vec(),
            Route::Hybrid => {
                let dsp = conservative_dsp_repair(
                    mid,
                    original,
                    config,
                    sample_rate,
                    Some(&repair_progress),
                );
                let neural = &neural_candidate.as_ref().expect("neural candidate exists")
                    [span.start..span.end];
                let combined = dsp
                    .iter()
                    .zip(original)
                    .zip(neural)
                    .map(|((dsp, original), neural)| {
                        dsp + config.hybrid_neural_mix * (neural - original)
                    })
                    .collect::<Vec<_>>();
                repair::neural::guard_candidate(mid, original, &combined, config, sample_rate)
                    .signal
            }
        };
        if repaired.len() != span.end - span.start {
            bail!("repair route returned an unexpected adaptive span length");
        }
        apply_repair_span(&mut output, s, &repaired, span);
        if !primary_uses_dsp {
            repair_progress.advance(1);
        }
    }
    repair_progress.finish();
    if config.mode == Mode::Hybrid && !neural_target_spans.is_empty() {
        let hybrid_progress =
            terminal::TaskProgress::new("HYBRID NEURAL MERGE", neural_target_spans.len(), "spans");
        let dsp_output = output.clone();
        for span in neural_target_spans {
            let mid = &m[span.start..span.end];
            let original = &dsp_output[span.start..span.end];
            let generated =
                &neural_candidate.as_ref().expect("neural candidate exists")[span.start..span.end];
            let input = &s[span.start..span.end];
            let combined = mix_hybrid_delta(original, input, generated, config.hybrid_neural_mix);
            let guarded = repair::neural::guard_candidate_with_detection(
                mid,
                original,
                input,
                &combined,
                config,
                sample_rate,
            );
            apply_repair_span(&mut output, &dsp_output, &guarded.signal, span);
            hybrid_progress.advance(1);
        }
        hybrid_progress.finish();
    } else if config.mode == Mode::Hybrid {
        terminal::status("HYBRID NEURAL MERGE", "not needed", Tone::Muted);
    }
    Ok(output)
}

fn split_hybrid_reports(reports: &[SegmentReport]) -> (Vec<SegmentReport>, Vec<SegmentReport>) {
    let mut primary = reports.to_vec();
    let mut neural = reports.to_vec();
    for report in &mut primary {
        if report.route != Route::Skip {
            report.route = Route::Dsp;
        }
    }
    for report in &mut neural {
        if report.route != Route::Hybrid {
            report.route = Route::Skip;
        }
    }
    (primary, neural)
}

fn mix_hybrid_delta(dsp: &[f32], input: &[f32], generated: &[f32], mix: f32) -> Vec<f32> {
    dsp.iter()
        .zip(input)
        .zip(generated)
        .map(|((dsp, input), generated)| dsp + mix * (generated - input))
        .collect()
}

fn conservative_dsp_repair(
    mid: &[f32],
    side: &[f32],
    config: &Config,
    sample_rate: u32,
    progress: Option<&terminal::TaskProgress>,
) -> Vec<f32> {
    const ADAPTIVE_SPAN_MIX: f32 = 0.75;
    repair::dsp::repair_with_progress(mid, side, config, sample_rate, || {
        if let Some(progress) = progress {
            progress.advance(1);
        }
    })
    .into_iter()
    .zip(side)
    .map(|(repaired, original)| original + ADAPTIVE_SPAN_MIX * (repaired - original))
    .collect()
}

#[derive(Debug, Clone, Copy)]
struct RepairSpan {
    start: usize,
    end: usize,
    route: Route,
    start_fade: usize,
    end_fade: usize,
}

#[derive(Debug, Clone, Copy)]
struct RawRepairSpan {
    start: usize,
    end: usize,
    route: Route,
}

fn adaptive_repair_spans(
    mid: &[f32],
    side: &[f32],
    ranges: &[Segment],
    reports: &[SegmentReport],
    sample_rate: u32,
) -> Vec<RepairSpan> {
    let length = mid.len().min(side.len());
    if length == 0 {
        return Vec::new();
    }
    let mut raw: Vec<RawRepairSpan> = Vec::new();
    let dsp_bridge_gap = sample_rate as usize * 750 / 1_000;
    for (segment, report) in ranges.iter().zip(reports) {
        if report.route == Route::Skip {
            continue;
        }
        if let Some(last) = raw.last_mut() {
            let bridges_dynamic_dsp = last.route == Route::Dsp
                && report.route == Route::Dsp
                && segment.start <= last.end.saturating_add(dsp_bridge_gap);
            if segment.start <= last.end || bridges_dynamic_dsp {
                last.end = last.end.max(segment.end);
                last.route = combined_route(last.route, report.route);
                continue;
            }
        }
        raw.push(RawRepairSpan {
            start: segment.start,
            end: segment.end,
            route: report.route,
        });
    }

    let search_radius = sample_rate as usize * 20 / 1_000;
    raw.iter()
        .enumerate()
        .filter_map(|(index, span)| {
            let start_boundary = if span.start == 0 {
                AdaptiveBoundary {
                    index: 0,
                    fade_samples: 0,
                    score: 0.0,
                }
            } else {
                let neighbor_limit = raw
                    .get(index.wrapping_sub(1))
                    .map(|previous| previous.end + (span.start - previous.end) / 2)
                    .unwrap_or(0);
                select_safe_boundary(
                    mid,
                    side,
                    span.start.saturating_sub(search_radius).max(neighbor_limit),
                    span.start,
                    sample_rate,
                )
            };
            let end_boundary = if span.end >= length {
                AdaptiveBoundary {
                    index: length,
                    fade_samples: 0,
                    score: 0.0,
                }
            } else {
                let neighbor_limit = raw
                    .get(index + 1)
                    .map(|next| span.end + (next.start - span.end) / 2)
                    .unwrap_or(length - 1);
                select_safe_boundary(
                    mid,
                    side,
                    span.end,
                    span.end
                        .saturating_add(search_radius)
                        .min(neighbor_limit)
                        .min(length - 1),
                    sample_rate,
                )
            };
            let span_length = end_boundary.index.saturating_sub(start_boundary.index);
            if span_length == 0 {
                return None;
            }
            let maximum_fade = span_length / 2;
            Some(RepairSpan {
                start: start_boundary.index,
                end: end_boundary.index,
                route: span.route,
                start_fade: start_boundary.fade_samples.min(maximum_fade),
                end_fade: end_boundary.fade_samples.min(maximum_fade),
            })
        })
        .collect()
}

fn combined_route(left: Route, right: Route) -> Route {
    match (left, right) {
        (Route::Skip, route) | (route, Route::Skip) => route,
        (Route::Dsp, Route::Dsp) => Route::Dsp,
        (Route::Neural, Route::Neural) => Route::Neural,
        _ => Route::Hybrid,
    }
}

fn neural_context_spans(repair_spans: &[RepairSpan], maximum_gap: usize) -> Vec<Segment> {
    let mut spans: Vec<Segment> = Vec::new();
    for span in repair_spans {
        if !matches!(span.route, Route::Neural | Route::Hybrid) {
            continue;
        }
        if let Some(last) = spans.last_mut() {
            if span.start <= last.end.saturating_add(maximum_gap) {
                last.end = last.end.max(span.end);
                continue;
            }
        }
        spans.push(Segment {
            start: span.start,
            end: span.end,
        });
    }
    spans
}

fn apply_repair_span(output: &mut [f32], original: &[f32], repaired: &[f32], span: RepairSpan) {
    let length = span.end - span.start;
    for (local, repaired_sample) in repaired.iter().take(length).enumerate() {
        let mut mix = 1.0f32;
        if span.start > 0 && span.start_fade > 0 {
            mix *= smoothstep(local as f32 / span.start_fade as f32);
        }
        if span.end < original.len() && span.end_fade > 0 {
            let remaining = length.saturating_sub(local + 1);
            mix *= smoothstep(remaining as f32 / span.end_fade as f32);
        }
        let index = span.start + local;
        output[index] = original[index] + mix * (*repaired_sample - original[index]);
    }
}

fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

/// Preserve the complete repair when possible. For unusually hot masters, cap fixed output
/// attenuation at 3 dB and reduce the synthesized M/S deltas together enough to avoid clipping.
fn fit_output_headroom(
    original_mid: &[f32],
    original_side: &[f32],
    repaired_mid: &[f32],
    repaired_side: &[f32],
) -> (Vec<f32>, Vec<f32>, f32, f32) {
    const PEAK_LIMIT: f32 = 0.998;
    const MIN_OUTPUT_GAIN_DB: f32 = -3.0;
    let length = original_mid
        .len()
        .min(original_side.len())
        .min(repaired_mid.len())
        .min(repaired_side.len());
    let peak_for_mix = |mix: f32| {
        (0..length).fold(0.0f32, |peak, index| {
            let mid = original_mid[index] + mix * (repaired_mid[index] - original_mid[index]);
            let side = original_side[index] + mix * (repaired_side[index] - original_side[index]);
            peak.max((mid + side).abs()).max((mid - side).abs())
        })
    };
    let full_peak = peak_for_mix(1.0);
    let full_gain = if full_peak > PEAK_LIMIT {
        PEAK_LIMIT / full_peak
    } else {
        1.0
    };
    let minimum_gain = 10.0f32.powf(MIN_OUTPUT_GAIN_DB / 20.0);
    let (gain, synthesis_mix) = if full_gain >= minimum_gain {
        (full_gain, 1.0)
    } else {
        let original_peak = peak_for_mix(0.0);
        let gain = minimum_gain.min(if original_peak > PEAK_LIMIT {
            PEAK_LIMIT / original_peak
        } else {
            1.0
        });
        let allowed_peak = PEAK_LIMIT / gain;
        let mut low = 0.0f32;
        let mut high = 1.0f32;
        for _ in 0..24 {
            let middle = (low + high) * 0.5;
            if peak_for_mix(middle) <= allowed_peak {
                low = middle;
            } else {
                high = middle;
            }
        }
        (gain, low)
    };
    let output_gain_db = 20.0 * gain.log10();
    (
        original_mid[..length]
            .iter()
            .zip(&repaired_mid[..length])
            .map(|(original, repaired)| gain * (original + synthesis_mix * (repaired - original)))
            .collect(),
        original_side[..length]
            .iter()
            .zip(&repaired_side[..length])
            .map(|(original, repaired)| gain * (original + synthesis_mix * (repaired - original)))
            .collect(),
        output_gain_db,
        synthesis_mix,
    )
}

fn quality_for_side(
    buffer: &AudioBuffer,
    m: &[f32],
    s: &[f32],
    config: &Config,
) -> crate::eval::QualityMetrics {
    let (l, r) = buffer.stereo().expect("validated stereo buffer");
    quality_metrics(m, s, l, r, metric_config(config, buffer.sample_rate))
}

fn metric_config(config: &Config, sample_rate: u32) -> MetricConfig {
    MetricConfig {
        fc_hz: config.fc,
        protected_end_hz: config.scan_start_hz.saturating_sub(500),
        n_fft: config.n_fft,
        hop: config.hop,
        sample_rate,
    }
}

fn ensure_nonempty(buffer: &AudioBuffer) -> Result<()> {
    crate::io::mside::require_stereo(buffer)?;
    if buffer.frames() == 0 {
        bail!("input WAV contains no audio frames");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::detector::SegmentMetrics;

    fn report(route: Route, segment: Segment) -> SegmentReport {
        SegmentReport {
            start: segment.start,
            end: segment.end,
            needs_processing: route != Route::Skip,
            route,
            metrics: SegmentMetrics {
                r_hf: 0.0,
                r_hf_low: 0.0,
                lsd_hf: 0.0,
                corr_hf: 0.0,
                r_intact: 0.0,
                corr_intact: 0.0,
                corr_transition: 0.0,
                r_transition: 0.0,
                band_continuity: 1.0,
                deficient_bands: 0,
                active_bands: 0,
                repair_start_hz: None,
            },
        }
    }

    #[test]
    fn adaptive_span_blends_only_the_repair_delta() {
        let original = vec![0.0f32; 8];
        let repaired = vec![1.0f32; 6];
        let mut output = original.clone();
        apply_repair_span(
            &mut output,
            &original,
            &repaired,
            RepairSpan {
                start: 1,
                end: 7,
                route: Route::Dsp,
                start_fade: 2,
                end_fade: 2,
            },
        );
        assert_eq!(output[0], 0.0);
        assert_eq!(output[1], 0.0);
        assert_eq!(output[6], 0.0);
        assert_eq!(output[7], 0.0);
        assert!(output[2] > 0.0 && output[3] == 1.0 && output[4] == 1.0);
    }

    #[test]
    fn sparse_neural_routes_form_only_local_spans() {
        let ranges = [
            Segment { start: 0, end: 100 },
            Segment {
                start: 50,
                end: 150,
            },
            Segment {
                start: 200,
                end: 300,
            },
            Segment {
                start: 250,
                end: 350,
            },
        ];
        let reports = [
            report(Route::Neural, ranges[0]),
            report(Route::Skip, ranges[1]),
            report(Route::Hybrid, ranges[2]),
            report(Route::Neural, ranges[3]),
        ];

        let mid = vec![0.0f32; 500];
        let side = mid.clone();
        let adaptive = adaptive_repair_spans(&mid, &side, &ranges, &reports, 1_000);
        assert_eq!(adaptive.len(), 2);
        assert_eq!(adaptive[0].start, 0);
        assert!(adaptive[0].end >= 100);
        assert!(adaptive[1].start <= 200 && adaptive[1].end >= 350);
        assert_eq!(adaptive[1].route, Route::Hybrid);

        let spans = neural_context_spans(&adaptive, 0);
        assert_eq!(spans.len(), 2);
        let contextual = neural_context_spans(&adaptive, 100);
        assert_eq!(contextual.len(), 1);
        assert_eq!(contextual[0].start, 0);
        assert!(contextual[0].end >= 350);
    }

    #[test]
    fn nearby_dsp_regions_share_one_continuous_dynamic_span() {
        let ranges = [
            Segment { start: 0, end: 100 },
            Segment {
                start: 700,
                end: 800,
            },
        ];
        let dsp_reports = [report(Route::Dsp, ranges[0]), report(Route::Dsp, ranges[1])];
        let neural_reports = [
            report(Route::Neural, ranges[0]),
            report(Route::Neural, ranges[1]),
        ];
        let signal = vec![0.0f32; 2_000];

        let dsp = adaptive_repair_spans(&signal, &signal, &ranges, &dsp_reports, 1_000);
        let neural = adaptive_repair_spans(&signal, &signal, &ranges, &neural_reports, 1_000);

        assert_eq!(dsp.len(), 1);
        assert_eq!(neural.len(), 2);
    }

    #[test]
    fn hybrid_routing_keeps_dsp_continuous_and_neural_targets_sparse() {
        let ranges = [
            Segment { start: 0, end: 100 },
            Segment {
                start: 50,
                end: 150,
            },
            Segment {
                start: 100,
                end: 200,
            },
        ];
        let reports = [
            report(Route::Dsp, ranges[0]),
            report(Route::Hybrid, ranges[1]),
            report(Route::Dsp, ranges[2]),
        ];
        let (primary, neural) = split_hybrid_reports(&reports);
        assert!(primary.iter().all(|report| report.route == Route::Dsp));
        assert_eq!(
            neural.iter().map(|report| report.route).collect::<Vec<_>>(),
            vec![Route::Skip, Route::Hybrid, Route::Skip]
        );

        let signal = vec![0.0f32; 300];
        let primary_spans = adaptive_repair_spans(&signal, &signal, &ranges, &primary, 1_000);
        let neural_spans = adaptive_repair_spans(&signal, &signal, &ranges, &neural, 1_000);
        assert_eq!(primary_spans.len(), 1);
        assert_eq!(primary_spans[0].route, Route::Dsp);
        assert_eq!(neural_spans.len(), 1);
        assert_eq!(neural_spans[0].route, Route::Hybrid);
        assert!(neural_spans[0].start <= 50 && neural_spans[0].end >= 150);
        assert!(neural_spans[0].start > 0 && neural_spans[0].end < 200);
    }

    #[test]
    fn hybrid_mix_preserves_the_complete_dsp_result() {
        let input = [0.1, -0.2, 0.3];
        let dsp = [0.2, -0.1, 0.4];
        let generated = [0.5, 0.2, -0.1];
        assert_eq!(mix_hybrid_delta(&dsp, &input, &generated, 0.0), dsp);
        assert_eq!(
            mix_hybrid_delta(&dsp, &input, &input, 0.3),
            dsp,
            "a neutral neural candidate must not attenuate DSP"
        );
        let mixed = mix_hybrid_delta(&dsp, &input, &generated, 0.5);
        for (actual, expected) in mixed.iter().zip([0.4, 0.1, 0.2]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn sustained_multiband_defects_are_promoted_without_isolated_flicker() {
        let ranges = (0..9)
            .map(|index| Segment {
                start: index * 100,
                end: (index + 1) * 100,
            })
            .collect::<Vec<_>>();
        let mut reports = ranges
            .iter()
            .map(|segment| report(Route::Skip, *segment))
            .collect::<Vec<_>>();
        for index in [1usize, 2, 3, 5, 6, 7] {
            reports[index].metrics.band_continuity = 0.35;
            reports[index].metrics.deficient_bands = 3;
            reports[index].metrics.active_bands = 12;
        }
        reports[4].metrics.band_continuity = 0.60;
        reports[4].metrics.deficient_bands = 1;
        reports[4].metrics.active_bands = 12;

        stabilize_multiband_detection(&mut reports);

        assert!(reports[1..8].iter().all(|report| report.needs_processing));
        assert!(!reports[0].needs_processing);
        assert!(!reports[8].needs_processing);

        let mut isolated = ranges
            .iter()
            .map(|segment| report(Route::Skip, *segment))
            .collect::<Vec<_>>();
        isolated[4].metrics.band_continuity = 0.2;
        isolated[4].metrics.deficient_bands = 3;
        isolated[4].metrics.active_bands = 12;
        stabilize_multiband_detection(&mut isolated);
        assert!(isolated.iter().all(|report| !report.needs_processing));
    }

    #[test]
    fn dsp_gain_remains_dynamic_inside_an_adaptive_span() {
        let sample_rate = 48_000;
        let segment_length = 8192;
        let total_length = segment_length * 2;
        let mut m = vec![0.0f32; total_length];
        let mut s = vec![0.0f32; total_length];
        for index in 0..total_length {
            let time = index as f32 / sample_rate as f32;
            m[index] = 0.2 * (2.0 * std::f32::consts::PI * 6_000.0 * time).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 10_000.0 * time).sin();
            let side_scale = if index < segment_length { 0.02 } else { 0.1 };
            s[index] = side_scale * (2.0 * std::f32::consts::PI * 6_000.0 * time).sin();
        }
        let ranges = [
            Segment {
                start: 0,
                end: segment_length,
            },
            Segment {
                start: segment_length,
                end: total_length,
            },
        ];
        let reports = [report(Route::Dsp, ranges[0]), report(Route::Dsp, ranges[1])];
        let config = Config {
            mode: Mode::Dsp,
            segment_ms: 170,
            overlap: 0.0,
            ..Config::default()
        };

        let repaired = repair_segments(&m, &s, &ranges, &reports, &config, sample_rate).unwrap();
        let delta_rms = |range: std::ops::Range<usize>| {
            let sum = range
                .clone()
                .map(|index| (repaired[index] - s[index]).powi(2))
                .sum::<f32>();
            (sum / range.len() as f32).sqrt()
        };
        let margin = segment_length / 4;
        let quiet_delta = delta_rms(margin..segment_length - margin);
        let loud_delta = delta_rms(segment_length + margin..total_length - margin);
        assert!(
            loud_delta > quiet_delta * 1.5,
            "expected frame-local DSP gains, got quiet={quiet_delta}, loud={loud_delta}"
        );
    }

    #[test]
    fn output_headroom_uses_one_transparent_gain() {
        let mid = vec![0.8, 0.2, -0.7, 0.0];
        let original_side = vec![0.0; 4];
        let repaired_side = vec![0.6, -0.2, -0.7, 0.2];

        let (limited_mid, limited_side, gain_db, synthesis_mix) =
            fit_output_headroom(&mid, &original_side, &mid, &repaired_side);

        let gain = 10.0f32.powf(gain_db / 20.0);
        assert_eq!(synthesis_mix, 1.0);
        for index in 0..mid.len() {
            assert!((limited_mid[index] - mid[index] * gain).abs() < 1e-6);
            assert!((limited_side[index] - repaired_side[index] * gain).abs() < 1e-6);
            assert!((limited_mid[index] + limited_side[index]).abs() <= 0.998_001);
            assert!((limited_mid[index] - limited_side[index]).abs() <= 0.998_001);
        }

        let (safe_mid, safe_side, safe_gain_db, safe_mix) = fit_output_headroom(
            &vec![0.1; 256],
            &vec![0.0; 256],
            &vec![0.1; 256],
            &vec![0.2; 256],
        );
        assert_eq!(safe_mid, vec![0.1; 256]);
        assert_eq!(safe_side, vec![0.2; 256]);
        assert_eq!(safe_gain_db, 0.0);
        assert_eq!(safe_mix, 1.0);
    }

    #[test]
    fn output_headroom_caps_loudness_loss_before_reducing_synthesis() {
        let mid = vec![0.95; 256];
        let original_side = vec![0.0; 256];
        let repaired_side = vec![0.95; 256];

        let (limited_mid, limited_side, gain_db, synthesis_mix) =
            fit_output_headroom(&mid, &original_side, &mid, &repaired_side);

        assert!((gain_db - -3.0).abs() < 1e-5);
        assert!(synthesis_mix > 0.0 && synthesis_mix < 1.0);
        for (mid, side) in limited_mid.iter().zip(limited_side) {
            assert!((mid + side).abs() <= 0.998_001);
            assert!((mid - side).abs() <= 0.998_001);
        }
    }

    #[test]
    fn existing_results_require_force_when_non_interactive() {
        let path = std::env::temp_dir().join(format!(
            "sidespread-overwrite-{}-{}.wav",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, b"keep me").unwrap();
        assert!(approve_overwrite(&path, true).unwrap());
        if !terminal::interactive() {
            assert!(approve_overwrite(&path, false).is_err());
        }
        assert_eq!(std::fs::read(&path).unwrap(), b"keep me");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn output_and_report_paths_must_differ() {
        let path = Path::new("same-result-path");
        assert!(approve_output_set(path, Some(path), true).is_err());
    }

    #[test]
    fn path_aliases_cannot_bypass_report_destination_checks() {
        let output = Path::new("/tmp/sidespread-result.wav");
        let alias = Path::new("/tmp/./sidespread-result.wav");
        assert!(destinations_match(output, alias));
        assert!(approve_output_set(output, Some(alias), true).is_err());
        assert!(ensure_report_destination(output, None, Some(alias)).is_err());
    }
}
