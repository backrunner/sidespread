//! STFT / iSTFT via realfft with Hann window and overlap-add.

use realfft::RealFftPlanner;

/// Configuration for STFT analysis/synthesis.
#[derive(Debug, Clone, Copy)]
pub struct StftConfig {
    pub n_fft: usize,
    pub hop: usize,
}

impl StftConfig {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        assert!(n_fft > 0 && hop > 0 && hop <= n_fft, "invalid STFT config");
        Self { n_fft, hop }
    }

    /// Hann window of length `n_fft` with periodic normalization (numpy/librosa `ones`).
    pub fn window(&self) -> Vec<f32> {
        let n = self.n_fft;
        (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
            .collect()
    }

    /// Number of frames for `len` samples with center padding (librosa `center=True`).
    pub fn num_frames(&self, len: usize) -> usize {
        // center=True pads n_fft//2 on each side → effective len = len + n_fft.
        // frames = 1 + (eff_len - n_fft) / hop = 1 + len / hop.
        1 + len / self.hop
    }
}

/// One complex spectrum frame: interleaved real/imag pairs per bin, n_fft/2 + 1 bins.
#[derive(Debug, Clone)]
pub struct SpectrumFrame {
    /// Complex values as [re, im, re, im, ...] of length `2 * (n_fft/2 + 1)`.
    pub cplx: Vec<f32>,
    pub n_bins: usize,
}

impl SpectrumFrame {
    pub fn new(n_bins: usize) -> Self {
        Self {
            cplx: vec![0.0; 2 * n_bins],
            n_bins,
        }
    }

    pub fn re(&self, b: usize) -> f32 {
        self.cplx[2 * b]
    }
    pub fn im(&self, b: usize) -> f32 {
        self.cplx[2 * b + 1]
    }
    pub fn mag(&self, b: usize) -> f32 {
        (self.re(b).powi(2) + self.im(b).powi(2)).sqrt()
    }
    pub fn phase(&self, b: usize) -> f32 {
        self.im(b).atan2(self.re(b))
    }
}

/// Real-valued STFT. Returns frames of complex spectra (n_fft/2 + 1 bins each).
/// Uses center padding (reflect) and Hann window, librosa-compatible layout.
pub fn stft(signal: &[f32], cfg: &StftConfig) -> Vec<SpectrumFrame> {
    let n = cfg.n_fft;
    let hop = cfg.hop;
    let win = cfg.window();
    let n_bins = n / 2 + 1;

    // Center pad using reflection that remains well-defined for short tail segments.
    let pad = n / 2;
    let mut padded = vec![0.0f32; signal.len() + 2 * pad];
    for (dst, sample) in padded.iter_mut().enumerate() {
        let source = reflect_index(dst as isize - pad as isize, signal.len());
        *sample = source.map(|index| signal[index]).unwrap_or(0.0);
    }

    let num_frames = if padded.len() >= n {
        1 + (padded.len() - n) / hop
    } else {
        1
    };

    let mut planner: RealFftPlanner<f32> = RealFftPlanner::new();
    let r2c = planner.plan_fft_forward(n);

    let mut frames = Vec::with_capacity(num_frames);
    let mut frame_buf = vec![0.0f32; n];
    let mut spectrum = r2c.make_output_vec();

    for f in 0..num_frames {
        let start = f * hop;
        let end = (start + n).min(padded.len());
        for i in 0..n {
            frame_buf[i] = if start + i < end {
                padded[start + i] * win[i]
            } else {
                0.0
            };
        }
        spectrum.iter_mut().for_each(|c| {
            c.re = 0.0;
            c.im = 0.0;
        });
        r2c.process(&mut frame_buf, &mut spectrum)
            .expect("realfft forward buffer sizes are valid");
        let mut sf = SpectrumFrame::new(n_bins);
        for (bin, value) in spectrum.iter().enumerate().take(n_bins) {
            sf.cplx[2 * bin] = value.re;
            sf.cplx[2 * bin + 1] = value.im;
        }
        frames.push(sf);
    }
    frames
}

/// Inverse STFT with overlap-add and Hann synthesis window (librosa-compatible).
/// `expected_len` is the original signal length (before center padding).
pub fn istft(frames: &[SpectrumFrame], cfg: &StftConfig, expected_len: usize) -> Vec<f32> {
    let n = cfg.n_fft;
    let hop = cfg.hop;
    let win = cfg.window();
    let n_bins = n / 2 + 1;

    let mut planner: RealFftPlanner<f32> = RealFftPlanner::new();
    let c2r = planner.plan_fft_inverse(n);

    let num_frames = frames.len();
    let out_len = if num_frames > 0 {
        (num_frames - 1) * hop + n
    } else {
        return vec![];
    };

    let mut out = vec![0.0f32; out_len];
    let mut norm = vec![0.0f32; out_len];
    let mut spectrum_in = c2r.make_input_vec();
    let mut frame_out = vec![0.0f32; n];

    for (frame_index, frame) in frames.iter().enumerate().take(num_frames) {
        for (bin, value) in spectrum_in
            .iter_mut()
            .enumerate()
            .take(n_bins.min(frame.n_bins))
        {
            value.re = frame.re(bin);
            value.im = frame.im(bin);
        }
        spectrum_in[0].im = 0.0;
        spectrum_in[n_bins - 1].im = 0.0;
        c2r.process(&mut spectrum_in, &mut frame_out)
            .expect("realfft inverse buffer sizes are valid");
        // realfft's inverse does NOT normalize by 1/n; numpy's irfft does. Match numpy.
        let scale = 1.0 / n as f32;
        let start = frame_index * hop;
        for i in 0..n {
            if start + i < out_len {
                out[start + i] += frame_out[i] * win[i] * scale;
                norm[start + i] += win[i] * win[i];
            }
        }
    }

    // Normalize by window overlap-sum, then trim center padding.
    for i in 0..out_len {
        if norm[i] > 1e-8 {
            out[i] /= norm[i];
        }
    }

    // Trim: remove n//2 from each side (center pad).
    let pad = n / 2;
    let trimmed_start = pad;
    let trimmed_end = (pad + expected_len).min(out_len);
    if trimmed_start >= trimmed_end {
        return vec![];
    }
    out[trimmed_start..trimmed_end].to_vec()
}

fn reflect_index(mut index: isize, len: usize) -> Option<usize> {
    if len == 0 {
        return None;
    }
    if len == 1 {
        return Some(0);
    }
    let max = len as isize - 1;
    while index < 0 || index > max {
        if index < 0 {
            index = -index;
        }
        if index > max {
            index = 2 * max - index;
        }
    }
    Some(index as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn stft_istft_roundtrip_sine() {
        let sr = 48000;
        let freq = 1000.0;
        let n = sr; // 1 second
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        let cfg = StftConfig::new(4096, 1024);
        let frames = stft(&signal, &cfg);
        let recon = istft(&frames, &cfg, signal.len());
        // Compare the valid (non-edge) region.
        let edge = cfg.n_fft;
        let lo = edge;
        let hi = recon.len().saturating_sub(edge).max(lo);
        let mut max_err = 0.0f32;
        for i in lo..hi.min(signal.len()) {
            let e = (recon[i] - signal[i]).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(max_err < 1e-3, "STFT roundtrip max error {max_err}");
    }

    #[test]
    fn hann_window_symmetry() {
        let cfg = StftConfig::new(1024, 256);
        let w = cfg.window();
        let n = w.len();
        // Periodic Hann w[i] = 0.5 - 0.5*cos(2πi/n). Symmetry: w[i] == w[n-i] (mod n).
        for i in 1..n {
            let mirror = n - i;
            assert_relative_eq!(w[i], w[mirror], epsilon = 1e-5);
        }
    }

    #[test]
    fn short_tail_does_not_panic() {
        let signal = vec![0.25f32; 1921];
        let cfg = StftConfig::new(4096, 1024);
        let frames = stft(&signal, &cfg);
        assert!(!frames.is_empty());
        let reconstructed = istft(&frames, &cfg, signal.len());
        assert_eq!(reconstructed.len(), signal.len());
        assert!(reconstructed.iter().all(|sample| sample.is_finite()));
    }
}
