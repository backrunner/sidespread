//! Mid/Side encoding and decoding.
//!
//! M = (L + R) / 2  — mid, contains centrally-panned content (vocals, bass, kick).
//! S = (L - R) / 2  — side, contains stereo width information.
//!
//! Sidespread targets high-frequency loss in the S channel specifically.

use crate::io::AudioBuffer;
use anyhow::{bail, Result};

/// Convert a stereo `AudioBuffer` into (M, S) mono-channel vectors.
/// Errors if the input is not stereo.
pub fn lr_to_ms(buf: &AudioBuffer) -> Result<(Vec<f32>, Vec<f32>)> {
    let (l, r) = buf
        .stereo()
        .ok_or_else(|| anyhow::anyhow!("lr_to_ms requires stereo input"))?;
    let n = l.len().min(r.len());
    let mut m = vec![0.0f32; n];
    let mut s = vec![0.0f32; n];
    for i in 0..n {
        m[i] = 0.5 * (l[i] + r[i]);
        s[i] = 0.5 * (l[i] - r[i]);
    }
    Ok((m, s))
}

/// Reconstruct a stereo `AudioBuffer` from M and S channels, with soft-clip limiting.
/// `sample_rate` and `bits_per_sample` are taken from `template` (typically the original input).
pub fn ms_to_lr(m: &[f32], s: &[f32], template: &AudioBuffer) -> AudioBuffer {
    let n = m.len().min(s.len());
    let mut l = vec![0.0f32; n];
    let mut r = vec![0.0f32; n];
    for i in 0..n {
        let li = m[i] + s[i];
        let ri = m[i] - s[i];
        // Soft-clip to prevent intersample peaks after M/S reconstruction.
        l[i] = soft_clip(li);
        r[i] = soft_clip(ri);
    }
    AudioBuffer {
        samples: vec![l, r],
        sample_rate: template.sample_rate,
        bits_per_sample: template.bits_per_sample,
        sample_format: template.sample_format,
    }
}

/// Soft clip with a narrow knee below full scale, bounded to [-1, 1].
#[inline]
pub fn soft_clip(x: f32) -> f32 {
    const THRESH: f32 = 0.95;
    let ax = x.abs();
    if ax <= THRESH {
        return x;
    }
    x.signum() * (THRESH + (1.0 - THRESH) * ((ax - THRESH) / (1.0 - THRESH)).tanh())
}

/// Sanity check: refuse to process a mono file with a helpful message.
pub fn require_stereo(buf: &AudioBuffer) -> Result<()> {
    if buf.channels() != 2 {
        bail!(
            "input must be stereo (got {} channel(s)). \
             Sidespread repairs the side (L-R) channel, which requires a stereo signal. \
             Mono audio has no side channel to repair.",
            buf.channels()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_buf(l: Vec<f32>, r: Vec<f32>, sr: u32) -> AudioBuffer {
        AudioBuffer {
            samples: vec![l, r],
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        }
    }

    #[test]
    fn lr_ms_roundtrip() {
        // Use signals with peaks well below 1.0 so soft_clip stays in its linear region.
        let l: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0).sin() * 0.3).collect();
        let r: Vec<f32> = (0..1000).map(|i| (i as f32 / 500.0).cos() * 0.2).collect();
        let buf = make_buf(l.clone(), r.clone(), 48000);
        let (m, s) = lr_to_ms(&buf).unwrap();
        let recon = ms_to_lr(&m, &s, &buf);
        let (rl, rr) = recon.stereo().unwrap();
        for i in 0..l.len() {
            assert!(
                (rl[i] - l[i]).abs() < 1e-4,
                "L mismatch at {i}: {} vs {}",
                rl[i],
                l[i]
            );
            assert!(
                (rr[i] - r[i]).abs() < 1e-4,
                "R mismatch at {i}: {} vs {}",
                rr[i],
                r[i]
            );
        }
    }

    #[test]
    fn mono_rejected() {
        let buf = AudioBuffer {
            samples: vec![vec![0.0; 100]],
            sample_rate: 48000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        assert!(require_stereo(&buf).is_err());
    }

    #[test]
    fn pure_mono_signal_has_zero_side() {
        // L == R → S == 0.
        let l: Vec<f32> = (0..256).map(|i| (i as f32 / 40.0).sin()).collect();
        let buf = make_buf(l.clone(), l.clone(), 48000);
        let (_m, s) = lr_to_ms(&buf).unwrap();
        assert!(s.iter().all(|&v| v.abs() < 1e-7));
    }
}
