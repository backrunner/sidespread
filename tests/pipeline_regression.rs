use sidespread::analysis::spectrum::bin_of;
use sidespread::analysis::stft::{istft, stft, StftConfig};
use sidespread::config::{Config, Mode};
use sidespread::eval::metrics::{
    compare_reference, high_band_projection_db, quality_metrics, MetricConfig,
};
use sidespread::io::{lr_to_ms, read_wav, write_wav};
use sidespread::pipeline;
use std::path::{Path, PathBuf};

fn temp_path(label: &str, extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "sidespread-{label}-{}.{extension}",
        std::process::id()
    ))
}

fn remove_if_exists(path: &Path) {
    let _ = std::fs::remove_file(path);
}

fn write_stereo<F>(path: &Path, sample_rate: u32, frames: usize, mut sample: F)
where
    F: FnMut(usize) -> (f32, f32),
{
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for index in 0..frames {
        let (left, right) = sample(index);
        writer
            .write_sample((left.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .unwrap();
        writer
            .write_sample((right.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .unwrap();
    }
    writer.finalize().unwrap();
}

fn brickwall_noise(length: usize, cutoff_hz: f32, seed: u32) -> Vec<f32> {
    let mut state = seed;
    let noise = (0..length)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 16) as f32 / 32_768.0 - 1.0) * 0.15
        })
        .collect::<Vec<_>>();
    let config = StftConfig::new(4096, 1024);
    let mut spectrum = stft(&noise, &config);
    let cutoff = bin_of(cutoff_hz, 4096, 48_000);
    for frame in &mut spectrum {
        for bin in cutoff..frame.n_bins {
            frame.cplx[2 * bin] = 0.0;
            frame.cplx[2 * bin + 1] = 0.0;
        }
    }
    istft(&spectrum, &config, length)
}

#[test]
fn healthy_process_writes_only_report() {
    let input = temp_path("healthy-input", "wav");
    let output = temp_path("healthy-output", "wav");
    let report = temp_path("healthy-report", "json");
    remove_if_exists(&output);
    write_stereo(&input, 48_000, 48_000, |index| {
        let phase = 2.0 * std::f32::consts::PI * 10_000.0 * index as f32 / 48_000.0;
        (phase.sin() * 0.25, phase.cos() * 0.25)
    });

    pipeline::process(&input, &output, &Config::default(), Some(report.as_path())).unwrap();
    assert!(!output.exists());
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(json["needs_processing"], false);
    assert!(json["overall"]["before"]["mcd"].is_number());

    remove_if_exists(&input);
    remove_if_exists(&report);
}

#[test]
fn optional_artifact_repairs_process_healthy_audio_and_are_reported() {
    let input = temp_path("artifact-input", "wav");
    let output = temp_path("artifact-output", "wav");
    let report = temp_path("artifact-report", "json");
    remove_if_exists(&output);
    write_stereo(&input, 48_000, 48_000, |index| {
        let time = index as f32 / 48_000.0;
        let carrier = (2.0 * std::f32::consts::PI * 10_000.0 * time).sin() * 0.16;
        let overtone = (2.0 * std::f32::consts::PI * 13_000.0 * time).sin() * 0.05;
        let phase = if (index / 8_000) % 2 == 0 { 0.4 } else { -1.2 };
        let width = (2.0 * std::f32::consts::PI * 11_000.0 * time + phase).sin() * 0.05;
        (carrier + overtone + width, carrier + overtone - width)
    });
    let config = Config {
        repair_smearing: true,
        smearing_strength: 0.4,
        repair_bleeding: true,
        bleeding_strength: 0.6,
        stabilize_phase: true,
        phase_strength: 0.8,
        ..Config::default()
    };

    pipeline::process(&input, &output, &config, Some(report.as_path())).unwrap();

    assert!(output.exists());
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(json["needs_processing"], true);
    assert_eq!(json["artifact_processing"]["smearing"]["enabled"], true);
    assert_eq!(
        json["artifact_processing"]["harmonic_bleeding"]["strength"],
        0.6
    );
    assert_eq!(
        json["artifact_processing"]["phase_incoherence"]["strength"],
        0.8
    );
    for processor in ["smearing", "harmonic_bleeding", "phase_incoherence"] {
        let setting = &json["artifact_processing"][processor];
        let requested = setting["strength"].as_f64().unwrap();
        let applied = setting["applied_strength"].as_f64().unwrap();
        assert!(applied > 0.0 && applied <= requested);
    }
    let before = &json["overall"]["before"];
    let after = &json["overall"]["after"];
    assert!(after["lsd_hf"].as_f64().unwrap() <= before["lsd_hf"].as_f64().unwrap());
    assert!(after["mcd"].as_f64().unwrap() <= before["mcd"].as_f64().unwrap());
    assert!(after["iccc_hf"].as_f64().unwrap() >= before["iccc_hf"].as_f64().unwrap());

    remove_if_exists(&input);
    remove_if_exists(&output);
    remove_if_exists(&report);
}

#[test]
fn zero_strength_artifact_switches_remain_a_pipeline_bypass() {
    let input = temp_path("artifact-zero-input", "wav");
    let output = temp_path("artifact-zero-output", "wav");
    remove_if_exists(&output);
    write_stereo(&input, 48_000, 48_000, |index| {
        let phase = 2.0 * std::f32::consts::PI * 10_000.0 * index as f32 / 48_000.0;
        (phase.sin() * 0.25, phase.cos() * 0.25)
    });
    let config = Config {
        repair_smearing: true,
        smearing_strength: 0.0,
        repair_bleeding: true,
        bleeding_strength: 0.0,
        stabilize_phase: true,
        phase_strength: 0.0,
        ..Config::default()
    };

    pipeline::process(&input, &output, &config, None).unwrap();
    assert!(!output.exists());

    remove_if_exists(&input);
}

#[test]
fn fidelity_rejected_optional_repairs_do_not_write_audio() {
    let input = temp_path("artifact-rejected-input", "wav");
    let output = temp_path("artifact-rejected-output", "wav");
    let report = temp_path("artifact-rejected-report", "json");
    remove_if_exists(&output);
    write_stereo(&input, 48_000, 48_000, |_| (0.0, 0.0));
    let config = Config {
        repair_smearing: true,
        smearing_strength: 1.0,
        repair_bleeding: true,
        bleeding_strength: 1.0,
        stabilize_phase: true,
        phase_strength: 1.0,
        ..Config::default()
    };

    pipeline::process(&input, &output, &config, Some(report.as_path())).unwrap();
    assert!(!output.exists());
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    for processor in ["smearing", "harmonic_bleeding", "phase_incoherence"] {
        assert_eq!(json["artifact_processing"][processor]["retained_mix"], 0.0);
    }

    remove_if_exists(&input);
    remove_if_exists(&report);
}

#[test]
fn shared_brickwall_cutoff_triggers_full_band_extension() {
    let input = temp_path("brickwall-input", "wav");
    let output = temp_path("brickwall-output", "wav");
    let report = temp_path("brickwall-report", "json");
    remove_if_exists(&output);
    let mid = brickwall_noise(48_000, 14_000.0, 7);
    let side = brickwall_noise(48_000, 14_000.0, 19)
        .into_iter()
        .map(|sample| sample * 0.4)
        .collect::<Vec<_>>();
    write_stereo(&input, 48_000, 48_000, |index| {
        (mid[index] + side[index], mid[index] - side[index])
    });

    pipeline::process(&input, &output, &Config::default(), Some(report.as_path())).unwrap();

    assert!(output.exists());
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(json["bandwidth"]["needs_extension"], true);
    let cutoff = json["bandwidth"]["detected_cutoff_hz"].as_f64().unwrap();
    assert!((cutoff - 14_000.0).abs() < 500.0, "cutoff={cutoff}");

    remove_if_exists(&input);
    remove_if_exists(&output);
    remove_if_exists(&report);
}

#[test]
fn low_confidence_process_does_not_rewrite_audio() {
    let input = temp_path("low-confidence-input", "wav");
    let output = temp_path("low-confidence-output", "wav");
    let report = temp_path("low-confidence-report", "json");
    remove_if_exists(&output);
    write_stereo(&input, 48_000, 48_000, |index| {
        let time = index as f32 / 48_000.0;
        let mid = 0.25 * (2.0 * std::f32::consts::PI * 10_000.0 * time).sin();
        let side = 0.1 * (2.0 * std::f32::consts::PI * 1_000.0 * time + 0.7).sin();
        (mid + side, mid - side)
    });
    let config = Config {
        mode: Mode::Auto,
        corr_high: 1.0,
        ..Config::default()
    };

    pipeline::process(&input, &output, &config, Some(report.as_path())).unwrap();

    assert!(!output.exists());
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(json["needs_processing"], true);
    assert!(
        json["missing_band_processing"]["detected_segments"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(json["missing_band_processing"]["routed_segments"], 0);
    assert_eq!(
        json["missing_band_processing"]["route_scope"],
        "dynamic_side_missing_band_fill"
    );
    assert!(json["segments"]
        .as_array()
        .unwrap()
        .iter()
        .all(|segment| segment["route"] == "skip"));
    remove_if_exists(&input);
    remove_if_exists(&report);
}

#[test]
fn short_tail_detect_is_safe() {
    let input = temp_path("short-tail-input", "wav");
    let report = temp_path("short-tail-report", "json");
    write_stereo(&input, 48_000, 9_601, |index| {
        let phase = 2.0 * std::f32::consts::PI * 1_000.0 * index as f32 / 48_000.0;
        (phase.sin() * 0.25, phase.sin() * 0.125)
    });
    pipeline::detect(&input, &Config::default(), Some(report.as_path())).unwrap();
    assert!(report.exists());
    remove_if_exists(&input);
    remove_if_exists(&report);
}

#[test]
fn detect_supports_44100_hz() {
    let input = temp_path("44100-input", "wav");
    let report = temp_path("44100-report", "json");
    write_stereo(&input, 44_100, 44_100, |index| {
        let left_phase = 2.0 * std::f32::consts::PI * 9_000.0 * index as f32 / 44_100.0;
        let right_phase = 2.0 * std::f32::consts::PI * 11_000.0 * index as f32 / 44_100.0;
        (left_phase.sin() * 0.2, right_phase.sin() * 0.2)
    });
    pipeline::detect(&input, &Config::default(), Some(report.as_path())).unwrap();
    assert!(report.exists());
    remove_if_exists(&input);
    remove_if_exists(&report);
}

#[test]
fn neural_failure_does_not_write_output() {
    let input = temp_path("missing-model-input", "wav");
    let output = temp_path("missing-model-output", "wav");
    let report = temp_path("missing-model-report", "json");
    remove_if_exists(&output);
    write_stereo(&input, 48_000, 48_000, |index| {
        let time = index as f32 / 48_000.0;
        let mid = (2.0 * std::f32::consts::PI * 1_000.0 * time).sin() * 0.15
            + (2.0 * std::f32::consts::PI * 10_000.0 * time).sin() * 0.10;
        let side = (2.0 * std::f32::consts::PI * 6_000.0 * time).sin() * 0.05;
        (mid + side, mid - side)
    });
    let config = Config {
        mode: Mode::Nn,
        model_path: Some(temp_path("missing-guided", "onnx")),
        ..Config::default()
    };
    assert!(pipeline::process(&input, &output, &config, Some(report.as_path())).is_err());
    assert!(!output.exists());
    remove_if_exists(&input);
    remove_if_exists(&report);
}

#[test]
fn thirty_two_bit_pcm_format_is_preserved() {
    let input = temp_path("pcm32-input", "wav");
    let output = temp_path("pcm32-output", "wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 48_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&input, spec).unwrap();
    for _ in 0..1024 {
        writer.write_sample(0i32).unwrap();
        writer.write_sample(0i32).unwrap();
    }
    writer.finalize().unwrap();
    let buffer = read_wav(&input).unwrap();
    write_wav(&output, &buffer).unwrap();
    let output_spec = hound::WavReader::open(&output).unwrap().spec();
    assert_eq!(output_spec.bits_per_sample, 32);
    assert_eq!(output_spec.sample_format, hound::SampleFormat::Int);
    remove_if_exists(&input);
    remove_if_exists(&output);
}

#[test]
fn eval_reports_enrichment_without_protected_band_damage() {
    let input = temp_path("eval-input", "wav");
    let output = temp_path("eval-output", "wav");
    let report = temp_path("eval-report", "json");
    write_stereo(&input, 48_000, 48_000, |index| {
        let time = index as f32 / 48_000.0;
        let mid = 0.2 * (2.0 * std::f32::consts::PI * 1_000.0 * time).sin()
            + 0.15 * (2.0 * std::f32::consts::PI * 10_000.0 * time).sin();
        let side = 0.1 * (2.0 * std::f32::consts::PI * 1_000.0 * time + 0.2).sin()
            + 0.075 * (2.0 * std::f32::consts::PI * 10_000.0 * time + 0.2).sin();
        (mid + side, mid - side)
    });
    let config = Config {
        mode: Mode::Dsp,
        ..Config::default()
    };
    pipeline::eval(&input, &output, &config, Some(report.as_path())).unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    let degraded_lsd = json["evaluation"]["degraded"]["lsd_hf"].as_f64().unwrap();
    let repaired_lsd = json["evaluation"]["repaired"]["lsd_hf"].as_f64().unwrap();
    let degraded_hf_snr = json["evaluation"]["degraded"]["snr_hf_db"]
        .as_f64()
        .unwrap();
    let repaired_hf_snr = json["evaluation"]["repaired"]["snr_hf_db"]
        .as_f64()
        .unwrap();
    let repaired_preserved_snr = json["evaluation"]["repaired"]["snr_preserved_db"]
        .as_f64()
        .unwrap();
    let before_r_hf = json["overall"]["before"]["r_hf"].as_f64().unwrap();
    let after_r_hf = json["overall"]["after"]["r_hf"].as_f64().unwrap();
    assert!(repaired_lsd < degraded_lsd);
    assert!(after_r_hf > before_r_hf + 0.01);
    assert!(repaired_preserved_snr > 40.0);
    assert!(repaired_hf_snr - degraded_hf_snr > -3.0);
    remove_if_exists(&input);
    remove_if_exists(&output);
    remove_if_exists(&report);
}

#[test]
fn skipped_eval_applies_the_limiter_only_once() {
    let input = temp_path("eval-skip-input", "wav");
    let output = temp_path("eval-skip-output", "wav");
    let report = temp_path("eval-skip-report", "json");
    write_stereo(&input, 48_000, 48_000, |index| {
        let time = index as f32 / 48_000.0;
        let mid = 0.8 * (2.0 * std::f32::consts::PI * 1_000.0 * time).sin()
            + 0.15 * (2.0 * std::f32::consts::PI * 10_000.0 * time).sin();
        let side = 0.12 * (2.0 * std::f32::consts::PI * 2_000.0 * time + 0.4).sin()
            + 0.08 * (2.0 * std::f32::consts::PI * 10_000.0 * time + 0.4).sin();
        (mid + side, mid - side)
    });
    let config = Config {
        mode: Mode::Skip,
        ..Config::default()
    };

    pipeline::eval(&input, &output, &config, Some(report.as_path())).unwrap();

    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    let degraded = json["evaluation"]["degraded"]["snr_hf_db"]
        .as_f64()
        .unwrap();
    let repaired = json["evaluation"]["repaired"]["snr_hf_db"]
        .as_f64()
        .unwrap();
    assert!(
        (repaired - degraded).abs() < 0.02,
        "skip changed HF-SNR by {} dB",
        repaired - degraded
    );
    remove_if_exists(&input);
    remove_if_exists(&output);
    remove_if_exists(&report);
}

#[test]
fn process_report_measures_the_written_waveform() {
    let input = temp_path("clipped-metrics-input", "wav");
    let output = temp_path("clipped-metrics-output", "wav");
    let report = temp_path("clipped-metrics-report", "json");
    write_stereo(&input, 48_000, 48_000, |index| {
        let time = index as f32 / 48_000.0;
        let mid = 0.6 * (2.0 * std::f32::consts::PI * 10_000.0 * time).sin()
            + 0.1 * (2.0 * std::f32::consts::PI * 6_000.0 * time).sin();
        let side = 0.25 * (2.0 * std::f32::consts::PI * 6_000.0 * time + 0.3).sin();
        (mid + side, mid - side)
    });
    let config = Config {
        mode: Mode::Dsp,
        ..Config::default()
    };

    pipeline::process(&input, &output, &config, Some(report.as_path())).unwrap();
    let written = read_wav(&output).unwrap();
    assert!(
        written
            .samples
            .iter()
            .flatten()
            .any(|sample| sample.abs() > 0.96),
        "fixture should exercise the limiter knee"
    );
    let (written_mid, written_side) = lr_to_ms(&written).unwrap();
    let (left, right) = written.stereo().unwrap();
    let recomputed = quality_metrics(
        &written_mid,
        &written_side,
        left,
        right,
        MetricConfig {
            fc_hz: config.fc,
            protected_end_hz: config.scan_start_hz.saturating_sub(500),
            n_fft: config.n_fft,
            hop: config.hop,
            sample_rate: written.sample_rate,
        },
    );
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    let after = &json["overall"]["after"];
    let assert_metric = |name: &str, expected: f32| {
        let reported = after[name].as_f64().unwrap() as f32;
        assert!(
            (reported - expected).abs() < 1e-5,
            "{name}: reported={reported}, recomputed={expected}"
        );
    };
    assert_metric("r_hf", recomputed.r_hf);
    assert_metric("lsd_hf", recomputed.lsd_hf);
    assert_metric("mcd", recomputed.mcd);
    assert_metric("iccc_hf", recomputed.iccc_hf);

    remove_if_exists(&input);
    remove_if_exists(&output);
    remove_if_exists(&report);
}

#[test]
#[ignore = "manual real-audio audit; set SIDESPREAD_AUDIT_INPUT and SIDESPREAD_AUDIT_OUTPUT"]
fn audit_external_process_fidelity() {
    let input = std::env::var("SIDESPREAD_AUDIT_INPUT").unwrap();
    let output = std::env::var("SIDESPREAD_AUDIT_OUTPUT").unwrap();
    let input = read_wav(input).unwrap();
    let output = read_wav(output).unwrap();
    assert_eq!(input.sample_rate, output.sample_rate);
    assert_eq!(input.frames(), output.frames());
    let settings = MetricConfig {
        fc_hz: 8_000,
        protected_end_hz: 4_500,
        n_fft: 4096,
        hop: 1024,
        sample_rate: input.sample_rate,
    };
    let mut minimum_full_snr = f32::INFINITY;
    let mut minimum_protected_snr = f32::INFINITY;
    let mut minimum_hf_snr = f32::INFINITY;
    let mut minimum_hf_projection = f32::INFINITY;
    for (reference, candidate) in input.samples.iter().zip(&output.samples) {
        let metrics = compare_reference(reference, candidate, settings);
        minimum_full_snr = minimum_full_snr.min(metrics.snr_db.unwrap());
        minimum_protected_snr = minimum_protected_snr.min(metrics.snr_preserved_db.unwrap());
        minimum_hf_snr = minimum_hf_snr.min(metrics.snr_hf_db.unwrap());
        minimum_hf_projection = minimum_hf_projection
            .min(high_band_projection_db(reference, candidate, settings).unwrap());
    }
    let peak = output
        .samples
        .iter()
        .flatten()
        .map(|sample| sample.abs())
        .fold(0.0f32, f32::max);
    println!(
        "PROCESS_FIDELITY full_snr_db={minimum_full_snr:.3} protected_snr_db={minimum_protected_snr:.3} hf_snr_db={minimum_hf_snr:.3} hf_projection_db={minimum_hf_projection:.3} peak={peak:.6}"
    );
    assert!(minimum_protected_snr >= 50.0);
    assert!(minimum_hf_projection >= -0.5);
    assert!(peak < 1.0);
}

#[test]
fn dsp_only_processing_accepts_a_custom_cutoff() {
    let input = temp_path("custom-dsp-cutoff-input", "wav");
    let output = temp_path("custom-dsp-cutoff-output", "wav");
    let report = temp_path("custom-dsp-cutoff-report", "json");
    write_stereo(&input, 48_000, 24_000, |index| {
        let time = index as f32 / 48_000.0;
        let mid = 0.25 * (2.0 * std::f32::consts::PI * 10_000.0 * time).sin()
            + 0.15 * (2.0 * std::f32::consts::PI * 3_000.0 * time).sin();
        let side = 0.05 * (2.0 * std::f32::consts::PI * 3_000.0 * time + 0.2).sin();
        (mid + side, mid - side)
    });
    let config = Config {
        fc: 6000,
        mode: Mode::Dsp,
        model_path: Some(temp_path("model-must-not-load", "onnx")),
        ..Config::default()
    };

    pipeline::process(&input, &output, &config, Some(report.as_path())).unwrap();
    assert!(output.exists());

    remove_if_exists(&input);
    remove_if_exists(&output);
    remove_if_exists(&report);
}
