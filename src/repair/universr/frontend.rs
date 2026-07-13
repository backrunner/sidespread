//! UniverSR frontend: STFT + amplitude compression, matching `torch.stft` exactly.
//!
//! PyTorch reference (ComplexSTFT + CompressAmplitudesAndScale):
//!   X = torch.stft(x, n_fft=1024, hop=512, window=hann(1024), center=True,
//!                  onesided=True, return_complex=True)         # [F=513, T_frames]
//!   Xc = (|X + eps|^alpha) * exp(1j * angle(X + eps)) * beta     # amplitude-compressed
//!   real = view_as_real(Xc).permute(0,3,1,2)[:,:, :-1,:]         # [2, F-1=512, T_frames]
//!
//! `center=True` uses symmetric reflect padding of n_fft//2 on each side.
//! `onesided=True` returns n_fft/2 + 1 = 513 bins (rfft).
//!
//! We produce a [2, 512, T_frames] f32 tensor (real/imag interleaved as 2 channels).

use realfft::{RealFftPlanner, RealToComplex};

pub struct Frontend {
    pub n_fft: usize,
    pub hop: usize,
    pub window: Vec<f32>,
    pub alpha: f32,
    pub beta: f32,
    pub comp_eps: f32,
    r2c: std::sync::Arc<dyn RealToComplex<f64>>,
}

impl Frontend {
    pub fn new(n_fft: usize, hop: usize, alpha: f32, beta: f32, comp_eps: f32) -> Self {
        let window: Vec<f32> = (0..n_fft)
            .map(|i| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n_fft - 1) as f64).cos())
                    as f32
            })
            .collect();
        let mut planner: RealFftPlanner<f64> = RealFftPlanner::new();
        let r2c = planner.plan_fft_forward(n_fft);
        Self {
            n_fft,
            hop,
            window,
            alpha,
            beta,
            comp_eps,
            r2c,
        }
    }

    /// STFT + amplitude compression → `[2, F-1=512, T_frames]` interleaved real/imag channels.
    /// Input: mono f32 samples. Output: Vec<f32> of length 2 * (n_fft/2) * T_frames, row-major
    /// with layout [channel=2][freq=512][time=T_frames].
    pub fn preprocess(&self, waveform: &[f32]) -> (Vec<f32>, usize, usize) {
        let n = self.n_fft;
        let hop = self.hop;
        let n_bins_keep = n / 2; // 512 (drop Nyquist)

        // center=True reflect pad: pad n//2 on each side with reflect mode.
        let pad = n / 2;
        let total_len = waveform.len() + 2 * pad;
        let mut padded = vec![0.0f32; total_len];
        for (dst, sample) in padded.iter_mut().enumerate() {
            let source = reflect_index(dst as isize - pad as isize, waveform.len());
            *sample = source.map(|index| waveform[index]).unwrap_or(0.0);
        }

        // Number of frames: 1 + (total_len - n) / hop (torch center=True formula).
        let t_frames = if total_len >= n {
            1 + (total_len - n) / hop
        } else {
            1
        };

        let mut out = vec![0.0f32; 2 * n_bins_keep * t_frames];
        let mut frame_buf = vec![0.0f64; n];
        let mut spectrum = self.r2c.make_output_vec();

        for f in 0..t_frames {
            let start = f * hop;
            for i in 0..n {
                frame_buf[i] = if start + i < total_len {
                    (padded[start + i] * self.window[i]) as f64
                } else {
                    0.0
                };
            }
            for s in spectrum.iter_mut() {
                s.re = 0.0;
                s.im = 0.0;
            }
            self.r2c
                .process(&mut frame_buf, &mut spectrum)
                .expect("realfft forward buffer sizes are valid");

            // Amplitude compression per bin (complex eps addition).
            for b in 0..n_bins_keep {
                let re = spectrum[b].re + self.comp_eps as f64;
                let im = spectrum[b].im;
                let mag = (re * re + im * im).sqrt();
                let angle = im.atan2(re);
                let compressed_mag = mag.powf(self.alpha as f64) * self.beta as f64;
                let cre = compressed_mag * angle.cos();
                let cim = compressed_mag * angle.sin();
                // layout: [channel=2][freq=n_bins_keep][time=t_frames]
                out[b * t_frames + f] = cre as f32;
                out[(n_bins_keep + b) * t_frames + f] = cim as f32;
            }
        }

        (out, n_bins_keep, t_frames)
    }

    /// Number of frames torch would produce for `len` samples with center=True.
    pub fn num_frames(&self, len: usize) -> usize {
        let total = len + 2 * (self.n_fft / 2);
        if total >= self.n_fft {
            1 + (total - self.n_fft) / self.hop
        } else {
            1
        }
    }
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
