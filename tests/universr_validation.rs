//! Numerical validation tests for the UniverSR integration against Python fixtures.

use sidespread::repair::universr::frontend::Frontend;
use sidespread::repair::universr::istft::Backend;

/// Minimal NPY (.npz) reader using the `zip` crate + a hand-rolled .npy parser.
mod npz {
    use std::collections::HashMap;
    use std::io::Read;

    /// Read a .npz file into a map of name (without .npy suffix) → (shape, f32 data).
    pub fn read(path: &str) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return HashMap::new(),
        };
        let mut zip = match zip::ZipArchive::new(file) {
            Ok(z) => z,
            Err(_) => return HashMap::new(),
        };
        let mut out = HashMap::new();
        for i in 0..zip.len() {
            let mut entry = match zip.by_index(i) {
                Ok(e) => e,
                Err(_) => continue,
            };
            let name = entry.name().trim_end_matches(".npy").to_string();
            let mut data = Vec::new();
            if entry.read_to_end(&mut data).is_err() {
                continue;
            }
            if let Some((shape, vals)) = parse_npy(&data) {
                out.insert(name, (shape, vals));
            }
        }
        out
    }

    /// Parse a .npy byte stream (header + data) for float32/float64 arrays → f32.
    /// Handles `fortran_order: True` by returning the data in **C-order** (row-major),
    /// so Rust code can always index with C-order strides regardless of how numpy
    /// decided to save it.
    fn parse_npy(data: &[u8]) -> Option<(Vec<usize>, Vec<f32>)> {
        if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
            return None;
        }
        let header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
        let header_str = std::str::from_utf8(&data[10..10 + header_len]).ok()?;
        let is_f64 = header_str.contains("'<f8'");
        let is_i8 = header_str.contains("'<i8'");
        let is_u8 = header_str.contains("'<u8'");
        let fortran_order = header_str.contains("'fortran_order': True");
        // shape
        let shape_start = header_str.find("shape':")? + "shape':".len();
        let rest = &header_str[shape_start..];
        let paren_open = rest.find('(')?;
        let paren_close = rest[paren_open..].find(')')?;
        let shape_str = &rest[paren_open + 1..paren_open + paren_close];
        let shape: Vec<usize> = shape_str
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        let data_start = 10 + header_len;
        let count: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        let elem_size = if is_f64 || is_i8 || is_u8 { 8 } else { 4 };
        if data_start + count * elem_size > data.len() {
            return None;
        }
        let mut vals = Vec::with_capacity(count);
        for i in 0..count {
            let off = data_start + i * elem_size;
            let v = if is_f64 {
                f64::from_le_bytes([
                    data[off],
                    data[off + 1],
                    data[off + 2],
                    data[off + 3],
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]) as f32
            } else if is_i8 {
                i64::from_le_bytes([
                    data[off],
                    data[off + 1],
                    data[off + 2],
                    data[off + 3],
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]) as f32
            } else if is_u8 {
                u64::from_le_bytes([
                    data[off],
                    data[off + 1],
                    data[off + 2],
                    data[off + 3],
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]) as f32
            } else {
                f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
            };
            vals.push(v);
        }
        // If saved in Fortran order, convert to C-order so Rust indexing is consistent.
        if fortran_order && shape.len() > 1 {
            vals = fortran_to_c(&shape, &vals);
        }
        Some((shape, vals))
    }

    /// Transpose a flat Fortran-order (column-major) array to C-order (row-major).
    fn fortran_to_c(shape: &[usize], fortran: &[f32]) -> Vec<f32> {
        let count: usize = shape.iter().product();
        if count == 0 || fortran.len() < count {
            return fortran.to_vec();
        }
        let mut c = vec![0.0f32; count];
        let ndim = shape.len();
        // Multi-index → flat fortran index: sum(idx_i * stride_i_f), stride_f[0]=1, stride_f[i]=stride_f[i-1]*shape[i-1].
        // Multi-index → flat C index: sum(idx_i * stride_i_c), stride_c[ndim-1]=1, stride_c[i]=stride_c[i+1]*shape[i+1].
        let mut strides_f = vec![1usize; ndim];
        for i in 1..ndim {
            strides_f[i] = strides_f[i - 1] * shape[i - 1];
        }
        let mut strides_c = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            strides_c[i] = strides_c[i + 1] * shape[i + 1];
        }
        let mut idx = vec![0usize; ndim];
        for _ in 0..count {
            let fi: usize = idx.iter().zip(strides_f.iter()).map(|(i, s)| i * s).sum();
            let ci: usize = idx.iter().zip(strides_c.iter()).map(|(i, s)| i * s).sum();
            c[ci] = fortran[fi];
            // Increment multi-index in C-order iteration (last dim fastest) —
            // iteration order doesn't matter for the final mapping, only the index
            // computation, so we just walk all multi-indices.
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        c
    }
}

type FixtureMap = std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>;

fn fixtures() -> Option<FixtureMap> {
    let p = "tests/fixtures/universr_ref.npz";
    let m = npz::read(p);
    if m.is_empty() {
        eprintln!("[universr tests] fixtures not found at {p}; skipping (run scripts/export_universr_onnx.py)");
        None
    } else {
        Some(m)
    }
}

fn get(m: &FixtureMap, k: &str) -> (Vec<usize>, Vec<f32>) {
    // .npz entry names may have "arr_0" form; we stored named arrays.
    for key in [k, &format!("{k}.npy")] {
        if let Some(v) = m.get(key) {
            return v.clone();
        }
    }
    panic!("missing fixture key: {k}")
}

#[test]
fn frontend_stft_matches_python() {
    let f = match fixtures() {
        Some(f) => f,
        None => return,
    };
    // Python stored `lr_audio` (the 48k bandwidth-limited input) and `window`.
    let (_lr_shape, lr) = get(&f, "lr_audio");
    let n_fft = get(&f, "n_fft").1[0] as usize;
    let hop = get(&f, "hop_length").1[0] as usize;
    let alpha = get(&f, "alpha").1[0];
    let beta = get(&f, "beta").1[0];
    let eps = get(&f, "comp_eps").1[0];

    let fe = Frontend::new(n_fft, hop, alpha, beta, eps);
    let (spec, n_bins, t_frames) = fe.preprocess(&lr);
    assert_eq!(n_bins, 512, "frontend should drop Nyquist -> 512 bins");
    let total = get(&f, "total_freq_bins").1[0] as usize;
    assert_eq!(n_bins, total);
    let (reference_shape, reference) = get(&f, "Y_lr");
    let condition_bins = reference_shape[2];
    assert_eq!(t_frames, reference_shape[3]);
    let mut max_error = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut compared = 0usize;
    let mut max_location = (0usize, 0usize, 0usize, 0.0f32, 0.0f32);
    for channel in 0..2 {
        for bin in 0..condition_bins {
            for time in 0..t_frames {
                let rust_index = (channel * n_bins + bin) * t_frames + time;
                let reference_index = (channel * condition_bins + bin) * t_frames + time;
                let error = (spec[rust_index] - reference[reference_index]).abs();
                squared_error += error as f64 * error as f64;
                compared += 1;
                if error > max_error {
                    max_error = error;
                    max_location = (
                        channel,
                        bin,
                        time,
                        spec[rust_index],
                        reference[reference_index],
                    );
                }
            }
        }
    }
    let rmse = (squared_error / compared as f64).sqrt() as f32;
    assert!(
        max_error < 0.005 && rmse < 0.00035,
        "frontend max abs error: {max_error}, rmse={rmse}, location/value={max_location:?}, rust_complex=({},{}) ref_complex=({},{})",
        spec[(max_location.1) * t_frames + max_location.2],
        spec[(n_bins + max_location.1) * t_frames + max_location.2],
        reference[(max_location.1) * t_frames + max_location.2],
        reference[(condition_bins + max_location.1) * t_frames + max_location.2],
    );

    let (_, reference_window) = get(&f, "window");
    let window_error = fe
        .window
        .iter()
        .zip(reference_window)
        .map(|(rust, python)| (*rust - python).abs())
        .fold(0.0f32, f32::max);
    assert!(
        window_error < 1e-6,
        "Hann window max abs error: {window_error}"
    );
}

#[test]
fn onnx_single_step_matches_python() {
    let f = match fixtures() {
        Some(f) => f,
        None => return,
    };
    use sidespread::repair::universr::onnx_session::Sessions;
    use std::path::Path;

    let models_dir = Path::new("models");
    let model = models_dir.join("universr_backbone.onnx");
    if !model.exists() {
        eprintln!("[onnx test] ONNX models not found; skipping");
        return;
    }

    let (x_shape, x) = get(&f, "x_probe");
    let (_, t) = get(&f, "t_probe");
    let (y_shape, y) = get(&f, "y_probe");
    let (_, out_g_ref) = get(&f, "out_guided");
    let (_, out_u_ref) = get(&f, "out_unguided");

    // Shapes: x=[1,2,432,T], t=[1], y=[1,2,Fy,T], out=[1,2,432,T]
    let xs = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    let ys = (y_shape[0], y_shape[1], y_shape[2], y_shape[3]);
    let mut sess = Sessions::load(&model).expect("load onnx");

    let (out_g, out_u) = sess.run_both(&x, xs, &t, &y, ys).expect("ONNX run");

    assert_eq!(out_g.len(), out_g_ref.len(), "guided output length");
    assert_eq!(out_u.len(), out_u_ref.len(), "uncond output length");

    let err_g: f32 = out_g
        .iter()
        .zip(out_g_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let err_u: f32 = out_u
        .iter()
        .zip(out_u_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        err_g < 5e-4,
        "guided ONNX vs PyTorch mismatch too large: {err_g}"
    );
    assert!(
        err_u < 5e-4,
        "uncond ONNX vs PyTorch mismatch too large: {err_u}"
    );
}

#[test]
fn full_ode_and_istft_match_python() {
    let f = match fixtures() {
        Some(f) => f,
        None => return,
    };
    use sidespread::repair::universr::ode::OdeSolver;
    use sidespread::repair::universr::onnx_session::Sessions;
    use std::path::Path;

    let model = Path::new("models/universr_backbone.onnx");
    if !model.exists() {
        return;
    }

    let (x_shape, x0) = get(&f, "x0");
    let (y_shape, y_lr) = get(&f, "Y_lr");
    let (reference_shape, reference_spec) = get(&f, "full_spec_out");
    let guidance_scale = get(&f, "guidance_scale").1[0];
    let lr_bin_count = get(&f, "lr_bin_count").1[0] as usize;
    let hf_start = get(&f, "hf_start_bin").1[0] as usize;
    let mut sessions = Sessions::load(model).expect("load ONNX model");
    let solver = OdeSolver::new(4, guidance_scale);
    let generated = solver
        .solve(
            &mut sessions,
            &x0,
            (x_shape[0], x_shape[1], x_shape[2], x_shape[3]),
            &y_lr,
            (y_shape[0], y_shape[1], y_shape[2], y_shape[3]),
        )
        .expect("solve ODE");

    let time_frames = x_shape[3];
    let total_bins = reference_shape[2];
    let generated_bins = x_shape[2];
    let slice_start = lr_bin_count - hf_start;
    let mut full_spec = vec![0.0f32; 2 * total_bins * time_frames];
    for channel in 0..2 {
        for bin in 0..lr_bin_count {
            for time in 0..time_frames {
                full_spec[(channel * total_bins + bin) * time_frames + time] =
                    y_lr[(channel * lr_bin_count + bin) * time_frames + time];
            }
        }
        for bin in lr_bin_count..total_bins {
            let generated_bin = slice_start + bin - lr_bin_count;
            for time in 0..time_frames {
                full_spec[(channel * total_bins + bin) * time_frames + time] =
                    generated[(channel * generated_bins + generated_bin) * time_frames + time];
            }
        }
    }
    assert!(snr(&reference_spec, &full_spec) > 40.0);

    let n_fft = get(&f, "n_fft").1[0] as usize;
    let hop = get(&f, "hop_length").1[0] as usize;
    let alpha = get(&f, "alpha").1[0];
    let beta = get(&f, "beta").1[0];
    let mut padded_spec = vec![0.0f32; 2 * (total_bins + 1) * time_frames];
    for channel in 0..2 {
        for bin in 0..total_bins {
            for time in 0..time_frames {
                padded_spec[(channel * (total_bins + 1) + bin) * time_frames + time] =
                    full_spec[(channel * total_bins + bin) * time_frames + time];
            }
        }
    }
    let original_length = get(&f, "lr_audio").0.last().copied().unwrap();
    let rust_waveform = Backend::new(n_fft, hop, alpha, beta).postprocess(
        &padded_spec,
        total_bins + 1,
        time_frames,
        original_length,
    );
    let (_, python_waveform) = get(&f, "out_wav");
    assert!(snr(&python_waveform, &rust_waveform) > 35.0);
    let max_waveform_error = python_waveform
        .iter()
        .zip(&rust_waveform)
        .map(|(python, rust)| (python - rust).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_waveform_error < 1e-3,
        "end-to-end waveform max abs error: {max_waveform_error}"
    );
}

fn snr(reference: &[f32], candidate: &[f32]) -> f32 {
    let signal = reference
        .iter()
        .map(|sample| *sample as f64 * *sample as f64)
        .sum::<f64>();
    let noise = reference
        .iter()
        .zip(candidate)
        .map(|(reference, candidate)| (*reference as f64 - *candidate as f64).powi(2))
        .sum::<f64>();
    (10.0 * (signal / noise.max(1e-20)).log10()) as f32
}

#[test]
fn istft_roundtrip_stability() {
    // Basic smoke test for the iSTFT backend.
    let be = Backend::new(1024, 512, 0.2, 1.0);
    // Synthetic flat spectrum → some finite output.
    let t_frames = 64;
    let n_bins = 513;
    let spec = vec![0.1f32; 2 * n_bins * t_frames];
    let out = be.postprocess(&spec, n_bins, t_frames, 32768);
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(!out.is_empty());
}

#[test]
#[ignore = "manual real-audio evaluation; set SIDESPREAD_UNIVERSR_WAV"]
fn universr_real_audio_synthetic_evaluation() {
    use sidespread::config::{Config, Mode};
    use sidespread::io::{read_wav, write_wav, AudioBuffer};
    use sidespread::pipeline;
    use std::path::PathBuf;

    let input = std::env::var("SIDESPREAD_UNIVERSR_WAV")
        .expect("SIDESPREAD_UNIVERSR_WAV must point to a stereo WAV");
    let model = std::env::var_os("SIDESPREAD_UNIVERSR_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/universr_backbone.onnx"));
    let buffer = read_wav(input).unwrap();
    let excerpt_len = buffer.sample_rate as usize;
    assert!(buffer.frames() >= excerpt_len);
    let start = (buffer.frames() - excerpt_len) / 2;
    let excerpt = AudioBuffer {
        samples: buffer
            .samples
            .iter()
            .map(|channel| channel[start..start + excerpt_len].to_vec())
            .collect(),
        sample_rate: buffer.sample_rate,
        bits_per_sample: buffer.bits_per_sample,
        sample_format: buffer.sample_format,
    };
    let stem = format!("sidespread-universr-real-{}", std::process::id());
    let excerpt_path = std::env::temp_dir().join(format!("{stem}-input.wav"));
    let output_path = std::env::temp_dir().join(format!("{stem}-output.wav"));
    let report_path = std::env::temp_dir().join(format!("{stem}-report.json"));
    for path in [&excerpt_path, &output_path, &report_path] {
        let _ = std::fs::remove_file(path);
    }
    write_wav(&excerpt_path, &excerpt).unwrap();

    let mut config = Config::from_evaluation(8_000, 0.3, (0.35, 0.40), Mode::Nn, 4, Some(model));
    config.overwrite_existing = true;
    pipeline::eval(&excerpt_path, &output_path, &config, Some(&report_path)).unwrap();

    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap()).unwrap();
    let degraded = &report["evaluation"]["degraded"];
    let repaired = &report["evaluation"]["repaired"];
    println!(
        "UNIVERSR_REAL lsd_hf={:.4}->{:.4} snr_hf={:.4}->{:.4} snr_preserved={:.4} projection_db={:.4}",
        degraded["lsd_hf"].as_f64().unwrap(),
        repaired["lsd_hf"].as_f64().unwrap(),
        degraded["snr_hf_db"].as_f64().unwrap(),
        repaired["snr_hf_db"].as_f64().unwrap(),
        repaired["snr_preserved_db"].as_f64().unwrap(),
        report["evaluation"]["existing_hf_projection_db"]
            .as_f64()
            .unwrap(),
    );

    for path in [&excerpt_path, &output_path, &report_path] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
#[ignore = "manual full-track neural processing; set SIDESPREAD_UNIVERSR_WAV and SIDESPREAD_UNIVERSR_OUTPUT"]
fn universr_full_real_audio_process() {
    use sidespread::config::{Config, Mode};
    use sidespread::pipeline;
    use std::path::PathBuf;

    let input = PathBuf::from(
        std::env::var_os("SIDESPREAD_UNIVERSR_WAV")
            .expect("SIDESPREAD_UNIVERSR_WAV must point to a stereo WAV"),
    );
    let output = PathBuf::from(
        std::env::var_os("SIDESPREAD_UNIVERSR_OUTPUT")
            .expect("SIDESPREAD_UNIVERSR_OUTPUT must specify the output WAV"),
    );
    let model = std::env::var_os("SIDESPREAD_UNIVERSR_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/universr_backbone.onnx"));
    let mut config = Config::from_evaluation(8_000, 0.3, (0.35, 0.40), Mode::Nn, 4, Some(model));
    config.overwrite_existing = true;
    pipeline::process(input, output, &config, None).unwrap();
}
