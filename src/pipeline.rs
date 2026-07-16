//! Pipeline orchestration for the CLI subcommands.

use crate::analysis::detector::SegmentReport;
use crate::analysis::segment::{segments, Segment};
use crate::analysis::{self};
use crate::config::{Config, Mode, Route};
use crate::eval::metrics::{
    compare_reference, high_band_projection_db, quality_metrics, EvaluationMetrics, MetricConfig,
    ProcessingMetrics,
};
use crate::eval::report::{print_summary, write_json, Report};
use crate::eval::synthetic;
use crate::io::{lr_to_ms, ms_to_lr, read_wav, write_wav, AudioBuffer};
use crate::repair;
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::path::Path;

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

    let stft_config = crate::analysis::stft::StftConfig::new(config.n_fft, config.hop);
    let metrics = crate::analysis::detector::compute_metrics(
        &crate::analysis::stft::stft(&m, &stft_config),
        &crate::analysis::stft::stft(&s, &stft_config),
        fc,
        buffer.sample_rate,
        config.n_fft,
    );

    println!("- file --------------------------------------------");
    println!("path        : {}", input.display());
    println!("sample_rate : {} Hz", buffer.sample_rate);
    println!("channels    : {}", buffer.channels());
    println!("bits/sample : {}", buffer.bits_per_sample);
    println!("sample fmt  : {:?}", buffer.sample_format);
    println!("frames      : {}", buffer.frames());
    println!("duration    : {:.3} s", buffer.duration_secs());
    println!("- M/S high-frequency analysis (fc={fc} Hz) -------");
    println!("R_hf    : {:.4}", metrics.r_hf);
    println!("LSD_hf  : {:.4} dB", metrics.lsd_hf);
    println!("corr_hf : {:.4}", metrics.corr_hf);
    println!("R_intact: {:.4}", metrics.r_intact);
    println!("corr_int: {:.4}", metrics.corr_intact);
    println!("R_trans : {:.6}", metrics.r_transition);
    println!("corr_trn : {:.4}", metrics.corr_transition);
    if config.needs_repair(metrics.r_hf, metrics.r_intact) {
        println!("side HF appears deficient; run `sidespread process`.");
    } else {
        println!("side HF looks healthy; no repair is likely needed.");
    }
    Ok(())
}

pub fn detect<P: AsRef<Path>, Q: AsRef<Path>>(
    input: P,
    config: &Config,
    report_path: Q,
) -> Result<()> {
    let buffer = read_wav(input.as_ref()).context("reading wav")?;
    ensure_nonempty(&buffer)?;
    config.validate(buffer.sample_rate)?;
    let (m, s) = crate::io::mside::lr_to_ms(&buffer)?;
    let (_, reports) = analyze_all(&m, &s, buffer.sample_rate, config);
    let report = Report {
        needs_processing: reports.iter().any(|segment| segment.needs_processing),
        segments: reports,
        overall: None,
        evaluation: None,
    };
    print_summary(&report);
    write_json(&report, report_path).context("writing report")?;
    if report.needs_processing {
        eprintln!("audio needs processing; run `sidespread process` to repair it.");
    } else {
        eprintln!("audio does not need processing.");
    }
    Ok(())
}

pub fn process<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
    input: P,
    output: Q,
    config: &Config,
    report_path: R,
) -> Result<()> {
    let input = input.as_ref();
    let output = output.as_ref();
    let buffer = read_wav(input).context("reading wav")?;
    ensure_nonempty(&buffer)?;
    config.validate(buffer.sample_rate)?;
    let (m, s) = crate::io::mside::lr_to_ms(&buffer)?;
    let (segment_ranges, reports) = analyze_all(&m, &s, buffer.sample_rate, config);
    let needs_processing = reports.iter().any(|segment| segment.needs_processing);
    let has_repair_route = reports.iter().any(|segment| segment.route != Route::Skip);
    let before = quality_for_side(&buffer, &m, &s, config);

    if !needs_processing || config.mode == Mode::Skip || !has_repair_route {
        let report = Report {
            needs_processing,
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
        write_json(&report, report_path).context("writing report")?;
        if needs_processing {
            if config.mode == Mode::Skip {
                eprintln!("repair skipped by --mode skip; no WAV was written.");
            } else {
                eprintln!("no segments met repair confidence; no WAV was written.");
            }
        } else {
            eprintln!("audio does not need processing; only the report was written.");
        }
        return Ok(());
    }

    let repaired_side = repair_segments(
        &m,
        &s,
        &segment_ranges,
        &reports,
        config,
        buffer.sample_rate,
    )?;
    let (output_mid, output_side, output_gain_db, synthesis_mix) =
        fit_output_headroom(&m, &s, &repaired_side);
    let output_buffer = ms_to_lr(&output_mid, &output_side, &buffer);
    write_wav(output, &output_buffer).context("writing output wav")?;
    let written_buffer = read_wav(output).context("reading written output wav")?;
    let (written_m, written_side) = lr_to_ms(&written_buffer)?;
    let after = quality_for_side(&written_buffer, &written_m, &written_side, config);

    let report = Report {
        needs_processing,
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
    write_json(&report, report_path).context("writing report")?;
    eprintln!("wrote {}", output.display());
    Ok(())
}

pub fn eval<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
    clean: P,
    output: Q,
    config: &Config,
    report_path: R,
) -> Result<()> {
    let clean = clean.as_ref();
    let output = output.as_ref();
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
    let repaired_side = if config.mode == Mode::Skip {
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
    let (output_mid, output_side, output_gain_db, synthesis_mix) =
        fit_output_headroom(&degraded_mid, &degraded_side, &repaired_side);
    let output_buffer = ms_to_lr(&output_mid, &output_side, &buffer);
    write_wav(output, &output_buffer).context("writing output wav")?;
    let written_buffer = read_wav(output).context("reading written output wav")?;
    let (written_mid, written_side) = lr_to_ms(&written_buffer)?;

    let report = Report {
        needs_processing: reports.iter().any(|segment| segment.needs_processing),
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
    write_json(&report, report_path).context("writing report")?;
    eprintln!("wrote {}", output.display());
    Ok(())
}

fn analyze_all(
    m: &[f32],
    s: &[f32],
    sample_rate: u32,
    config: &Config,
) -> (Vec<Segment>, Vec<SegmentReport>) {
    let ranges = segments(m.len(), sample_rate, config.segment_ms, config.overlap);
    let mut reports = ranges
        .par_iter()
        .map(|segment| {
            analysis::analyze(
                &m[segment.start..segment.end],
                &s[segment.start..segment.end],
                segment.start,
                segment.end,
                config,
                sample_rate,
            )
        })
        .collect::<Vec<_>>();
    smooth_route_correlation(&mut reports, config, 9);
    (ranges, reports)
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
        report.route = config.decide(
            report.needs_processing,
            intact,
            transition,
            transition_ratio,
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

    let neural_spans = neural_routed_spans(ranges, reports);
    let neural_candidate = if neural_spans.is_empty() {
        None
    } else {
        let mut state = repair::neural::NeuralState::from_config(config)
            .context("loading UniverSR repair state")?;
        let mut candidate = s.to_vec();
        for span in neural_spans {
            let repaired = state
                .repair_signal(&s[span.start..span.end], sample_rate)
                .context("UniverSR repair failed")?;
            if repaired.len() != span.end - span.start {
                bail!("neural repair returned an unexpected routed span length");
            }
            candidate[span.start..span.end].copy_from_slice(&repaired);
        }
        Some(candidate)
    };

    let segment_length = (config.segment_ms * sample_rate as usize) / 1000;
    let segment_hop = ((segment_length as f32) * (1.0 - config.overlap))
        .round()
        .max(1.0) as usize;
    let fade_length = segment_length.saturating_sub(segment_hop);
    let mut accumulated = vec![0.0f32; s.len()];
    let mut weights = vec![0.0f32; s.len()];

    let repaired_segments = ranges
        .par_iter()
        .zip(reports.par_iter())
        .map(|(segment, report)| {
            let s_segment = &s[segment.start..segment.end];
            let m_segment = &m[segment.start..segment.end];
            let repaired = match report.route {
                Route::Skip => s_segment.to_vec(),
                Route::Dsp => repair::dsp::repair(m_segment, s_segment, config, sample_rate),
                Route::Neural => neural_candidate.as_ref().expect("neural candidate exists")
                    [segment.start..segment.end]
                    .to_vec(),
                Route::Hybrid => {
                    let dsp = repair::dsp::repair(m_segment, s_segment, config, sample_rate);
                    let neural = &neural_candidate.as_ref().expect("neural candidate exists")
                        [segment.start..segment.end];
                    dsp.iter()
                        .zip(neural)
                        .map(|(dsp, neural)| 0.7 * dsp + 0.3 * neural)
                        .collect()
                }
            };
            if repaired.len() != segment.end - segment.start {
                bail!("repair route returned an unexpected segment length");
            }
            Ok(repaired)
        })
        .collect::<Vec<Result<Vec<f32>>>>()
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    for (segment, repaired) in ranges.iter().zip(repaired_segments) {
        add_segment(
            &mut accumulated,
            &mut weights,
            &repaired,
            *segment,
            s.len(),
            fade_length,
        );
    }

    Ok(accumulated
        .into_iter()
        .zip(weights)
        .zip(s)
        .map(|((value, weight), original)| {
            if weight > 1e-8 {
                value / weight
            } else {
                *original
            }
        })
        .collect())
}

fn neural_routed_spans(ranges: &[Segment], reports: &[SegmentReport]) -> Vec<Segment> {
    let mut spans: Vec<Segment> = Vec::new();
    for (segment, report) in ranges.iter().zip(reports) {
        if !matches!(report.route, Route::Neural | Route::Hybrid) {
            continue;
        }
        if let Some(last) = spans.last_mut() {
            if segment.start <= last.end {
                last.end = last.end.max(segment.end);
                continue;
            }
        }
        spans.push(*segment);
    }
    spans
}

fn add_segment(
    accumulated: &mut [f32],
    weights: &mut [f32],
    segment_samples: &[f32],
    segment: Segment,
    total_length: usize,
    fade_length: usize,
) {
    let length = segment_samples.len();
    for (local, sample) in segment_samples.iter().enumerate() {
        let mut weight = 1.0f32;
        if segment.start > 0 && local < fade_length {
            weight *= smoothstep(local as f32 / fade_length.max(1) as f32);
        }
        if segment.end < total_length && local + fade_length >= length {
            let remaining = length.saturating_sub(local + 1);
            weight *= smoothstep(remaining as f32 / fade_length.max(1) as f32);
        }
        let index = segment.start + local;
        accumulated[index] += sample * weight;
        weights[index] += weight;
    }
}

fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

/// Preserve the complete repair when possible. For unusually hot masters, cap fixed output
/// attenuation at 3 dB and reduce only the synthesized side delta enough to avoid clipping.
fn fit_output_headroom(
    mid: &[f32],
    original_side: &[f32],
    repaired_side: &[f32],
) -> (Vec<f32>, Vec<f32>, f32, f32) {
    const PEAK_LIMIT: f32 = 0.998;
    const MIN_OUTPUT_GAIN_DB: f32 = -3.0;
    let length = mid.len().min(original_side.len()).min(repaired_side.len());
    let peak_for_mix = |mix: f32| {
        (0..length).fold(0.0f32, |peak, index| {
            let side = original_side[index] + mix * (repaired_side[index] - original_side[index]);
            peak.max((mid[index] + side).abs())
                .max((mid[index] - side).abs())
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
        mid[..length].iter().map(|sample| sample * gain).collect(),
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
                lsd_hf: 0.0,
                corr_hf: 0.0,
                r_intact: 0.0,
                corr_intact: 0.0,
                corr_transition: 0.0,
                r_transition: 0.0,
            },
        }
    }

    #[test]
    fn overlap_add_is_continuous_and_normalized() {
        let ranges = [Segment { start: 0, end: 4 }, Segment { start: 2, end: 6 }];
        let mut accumulated = vec![0.0f32; 6];
        let mut weights = vec![0.0f32; 6];
        add_segment(&mut accumulated, &mut weights, &[1.0; 4], ranges[0], 6, 2);
        add_segment(&mut accumulated, &mut weights, &[1.0; 4], ranges[1], 6, 2);
        let result = accumulated
            .iter()
            .zip(&weights)
            .map(|(value, weight)| value / weight)
            .collect::<Vec<_>>();
        assert!(result.iter().all(|value| (*value - 1.0).abs() < 1e-6));
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

        let spans = neural_routed_spans(&ranges, &reports);
        assert_eq!(spans.len(), 2);
        assert_eq!((spans[0].start, spans[0].end), (0, 100));
        assert_eq!((spans[1].start, spans[1].end), (200, 350));
    }

    #[test]
    fn dsp_gain_is_estimated_independently_for_each_segment() {
        let sample_rate = 48_000;
        let segment_length = 8192;
        let total_length = segment_length * 2;
        let mut m = vec![0.0f32; total_length];
        let mut s = vec![0.0f32; total_length];
        for index in 0..total_length {
            let time = index as f32 / sample_rate as f32;
            m[index] = 0.2 * (2.0 * std::f32::consts::PI * 2_000.0 * time).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 10_000.0 * time).sin();
            let side_scale = if index < segment_length { 0.02 } else { 0.1 };
            s[index] = side_scale * (2.0 * std::f32::consts::PI * 2_000.0 * time).sin();
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
        let quiet_delta = delta_rms(0..segment_length);
        let loud_delta = delta_rms(segment_length..total_length);
        assert!(
            loud_delta > quiet_delta * 3.0,
            "expected independent DSP gains, got quiet={quiet_delta}, loud={loud_delta}"
        );
    }

    #[test]
    fn output_headroom_uses_one_transparent_gain() {
        let mid = vec![0.8, 0.2, -0.7, 0.0];
        let original_side = vec![0.0; 4];
        let repaired_side = vec![0.6, -0.2, -0.7, 0.2];

        let (limited_mid, limited_side, gain_db, synthesis_mix) =
            fit_output_headroom(&mid, &original_side, &repaired_side);

        let gain = 10.0f32.powf(gain_db / 20.0);
        assert_eq!(synthesis_mix, 1.0);
        for index in 0..mid.len() {
            assert!((limited_mid[index] - mid[index] * gain).abs() < 1e-6);
            assert!((limited_side[index] - repaired_side[index] * gain).abs() < 1e-6);
            assert!((limited_mid[index] + limited_side[index]).abs() <= 0.998_001);
            assert!((limited_mid[index] - limited_side[index]).abs() <= 0.998_001);
        }

        let (safe_mid, safe_side, safe_gain_db, safe_mix) =
            fit_output_headroom(&vec![0.1; 256], &vec![0.0; 256], &vec![0.2; 256]);
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
            fit_output_headroom(&mid, &original_side, &repaired_side);

        assert!((gain_db - -3.0).abs() < 1e-5);
        assert!(synthesis_mix > 0.0 && synthesis_mix < 1.0);
        for (mid, side) in limited_mid.iter().zip(limited_side) {
            assert!((mid + side).abs() <= 0.998_001);
            assert!((mid - side).abs() <= 0.998_001);
        }
    }
}
