//! UniverSR backend: amplitude-compression invert + iSTFT, matching `torch.istft`.
//!
//! PyTorch reference (AmplitudeCompressedComplexSTFT.invert):
//!   X = |Xc / beta|^(1/alpha) * exp(1j * angle(Xc / beta))   # NOTE: no eps re-added
//!   x = torch.istft(X, n_fft=1024, hop=512, window=hann, center=True,
//!                   onesided=True, length=orig_len)
//!
//! `torch.istft` with `center=True` uses the overlap-add algorithm with the
//! analysis window and divides by the window overlap-sum normalization
//! (matching numpy/librosa conventions).

use realfft::{ComplexToReal, RealFftPlanner};

pub struct Backend {
    pub n_fft: usize,
    pub hop: usize,
    pub window: Vec<f32>,
    pub alpha: f32,
    pub beta: f32,
    c2r: std::sync::Arc<dyn ComplexToReal<f32>>,
}

impl Backend {
    pub fn new(n_fft: usize, hop: usize, alpha: f32, beta: f32) -> Self {
        let window: Vec<f32> = (0..n_fft)
            .map(|i| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n_fft - 1) as f64).cos())
                    as f32
            })
            .collect();
        let mut planner: RealFftPlanner<f32> = RealFftPlanner::new();
        let c2r = planner.plan_fft_inverse(n_fft);
        Self {
            n_fft,
            hop,
            window,
            alpha,
            beta,
            c2r,
        }
    }

    /// Inverse: compressed spec `[2, 513, T_frames]` → waveform.
    /// `spec` layout: [channel=2][freq=513][time=T_frames], f32 row-major.
    /// `orig_len` is the desired output length (matching torch istft `length`).
    pub fn postprocess(
        &self,
        spec: &[f32],
        n_bins: usize,
        t_frames: usize,
        orig_len: usize,
    ) -> Vec<f32> {
        let n = self.n_fft;
        let hop = self.hop;
        let n_bins_full = n / 2 + 1; // 513
        debug_assert_eq!(n_bins, n_bins_full);

        // Reconstruct full complex spectrum per frame, invert compression.
        let mut spectrum_in = self.c2r.make_input_vec();
        let mut frame_out = vec![0.0f32; n];

        // Output length with center padding: 1 + (T-1)*hop + ... ; we'll trim later.
        let out_len = (t_frames - 1) * hop + n;
        let mut out = vec![0.0f32; out_len];
        let mut norm = vec![0.0f32; out_len];

        for f in 0..t_frames {
            for b in 0..n_bins_full {
                let cre = spec[b * t_frames + f];
                let cim = spec[(n_bins_full + b) * t_frames + f];
                // Invert: Xc/beta, then |·|^(1/alpha) * exp(1j*angle(·)). No eps.
                let mag = (cre * cre + cim * cim).sqrt() / self.beta;
                let angle = cim.atan2(cre);
                let restored_mag = mag.powf(1.0 / self.alpha);
                spectrum_in[b].re = restored_mag * angle.cos();
                spectrum_in[b].im = restored_mag * angle.sin();
            }
            spectrum_in[0].im = 0.0;
            spectrum_in[n_bins_full - 1].im = 0.0;
            self.c2r
                .process(&mut spectrum_in, &mut frame_out)
                .expect("realfft inverse buffer sizes are valid");
            // realfft inverse: unnormalized. torch.istft normalizes by 1/n.
            let scale = 1.0 / n as f32;
            let start = f * hop;
            for i in 0..n {
                if start + i < out_len {
                    out[start + i] += frame_out[i] * self.window[i] * scale;
                    norm[start + i] += self.window[i] * self.window[i];
                }
            }
        }

        // Normalize by window overlap-sum.
        for i in 0..out_len {
            if norm[i] > 1e-10 {
                out[i] /= norm[i];
            }
        }

        // Trim center padding: torch center=True pads n//2 each side; output length = orig_len.
        // The overlap-add reconstruction gives us out_len = (T-1)*hop + n.
        // The valid region is [n//2, n//2 + orig_len).
        let pad = n / 2;
        let start = pad;
        let end = (pad + orig_len).min(out_len);
        if start >= end {
            return vec![0.0; orig_len];
        }
        let mut trimmed = vec![0.0f32; orig_len];
        let copy_len = end - start;
        trimmed[..copy_len].copy_from_slice(&out[start..end]);
        trimmed
    }
}
