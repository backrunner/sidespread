use sidespread::config::{Config, Mode};
use sidespread::eval::metrics::{quality_metrics, MetricConfig};
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

    pipeline::process(&input, &output, &Config::default(), &report).unwrap();
    assert!(!output.exists());
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(json["needs_processing"], false);
    assert!(json["overall"]["before"]["mcd"].is_number());

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
    pipeline::detect(&input, &Config::default(), &report).unwrap();
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
    pipeline::detect(&input, &Config::default(), &report).unwrap();
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
        let phase = 2.0 * std::f32::consts::PI * 1_000.0 * index as f32 / 48_000.0;
        let mono = phase.sin() * 0.25;
        (mono, mono)
    });
    let config = Config {
        mode: Mode::Nn,
        model_path: Some(temp_path("missing-guided", "onnx")),
        ..Config::default()
    };
    assert!(pipeline::process(&input, &output, &config, &report).is_err());
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
fn eval_reports_ground_truth_improvement() {
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
    pipeline::eval(&input, &output, &config, &report).unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    let degraded_lsd = json["evaluation"]["degraded"]["lsd_hf"].as_f64().unwrap();
    let repaired_lsd = json["evaluation"]["repaired"]["lsd_hf"].as_f64().unwrap();
    let degraded_snr = json["evaluation"]["degraded"]["snr_db"].as_f64().unwrap();
    let repaired_snr = json["evaluation"]["repaired"]["snr_db"].as_f64().unwrap();
    assert!(repaired_lsd < degraded_lsd);
    assert!(repaired_snr > degraded_snr);
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

    pipeline::process(&input, &output, &config, &report).unwrap();
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

    pipeline::process(&input, &output, &config, &report).unwrap();
    assert!(output.exists());

    remove_if_exists(&input);
    remove_if_exists(&output);
    remove_if_exists(&report);
}
