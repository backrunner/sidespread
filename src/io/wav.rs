//! WAV reading/writing via hound. Supports 16/24/32-bit PCM and float.

use anyhow::{bail, Context, Result};
use std::path::Path;

/// Multi-channel audio buffer. We only handle stereo (channels==2) for sidespread,
/// but the type is general enough for the analysis pipeline.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Interleaved-per-channel: `samples[ch][frame]`.
    pub samples: Vec<Vec<f32>>,
    pub sample_rate: u32,
    pub bits_per_sample: u16,
    pub sample_format: hound::SampleFormat,
}

impl AudioBuffer {
    pub fn channels(&self) -> usize {
        self.samples.len()
    }

    pub fn frames(&self) -> usize {
        self.samples.first().map(|c| c.len()).unwrap_or(0)
    }

    pub fn duration_secs(&self) -> f64 {
        self.frames() as f64 / self.sample_rate as f64
    }

    pub fn stereo(&self) -> Option<(&[f32], &[f32])> {
        if self.channels() == 2 {
            Some((&self.samples[0], &self.samples[1]))
        } else {
            None
        }
    }
}

/// Read a WAV file into an `AudioBuffer` with f32 samples normalized to [-1, 1].
pub fn read_wav<P: AsRef<Path>>(path: P) -> Result<AudioBuffer> {
    let path = path.as_ref();
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open wav for reading: {}", path.display()))?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let channels = spec.channels as usize;
    let bps = spec.bits_per_sample;
    let sample_format = spec.sample_format;

    if !matches!(sr, 44_100 | 48_000) {
        bail!(
            "sidespread supports 44.1 kHz or 48 kHz WAV input, but {} is {} Hz",
            path.display(),
            sr
        );
    }
    let valid_format = matches!(
        (sample_format, bps),
        (hound::SampleFormat::Int, 16 | 24 | 32) | (hound::SampleFormat::Float, 32)
    );
    if !valid_format {
        bail!(
            "unsupported WAV encoding: {:?} {}-bit; expected 16/24/32-bit PCM or 32-bit float",
            sample_format,
            bps
        );
    }

    if channels != 2 {
        bail!(
            "sidespread requires stereo input, but {} has {} channel(s)",
            path.display(),
            channels
        );
    }

    let total_frames = reader.len() / channels as u32;
    let mut samples: Vec<Vec<f32>> = vec![Vec::with_capacity(total_frames as usize); channels];

    match sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1u64 << (bps - 1)) as f32;
            for (i, r) in reader.samples::<i32>().enumerate() {
                let v = r.context("decode error while reading samples")?;
                let f = (v as f32) / max_val;
                samples[i % channels].push(f);
            }
        }
        hound::SampleFormat::Float => {
            for (i, r) in reader.samples::<f32>().enumerate() {
                let v = r.context("decode error while reading float samples")?;
                samples[i % channels].push(v);
            }
        }
    }

    Ok(AudioBuffer {
        samples,
        sample_rate: sr,
        bits_per_sample: bps,
        sample_format,
    })
}

/// Write an `AudioBuffer` to a WAV file. Uses the same bits_per_sample as the input
/// (16/24/32-bit PCM or 32-bit float).
pub fn write_wav<P: AsRef<Path>>(path: P, buf: &AudioBuffer) -> Result<()> {
    let path = path.as_ref();
    let channels = buf.channels() as u16;
    let sr = buf.sample_rate;
    let bps = buf.bits_per_sample;
    let sample_format = buf.sample_format;

    let spec = hound::WavSpec {
        channels,
        sample_rate: sr,
        bits_per_sample: bps,
        sample_format,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create wav: {}", path.display()))?;

    let frames = buf.frames();
    match sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (bps - 1)) - 1;
            for f in 0..frames {
                for ch in 0..channels as usize {
                    let v = buf.samples[ch][f].clamp(-1.0, 1.0);
                    let iv = (v * max_val as f32).round() as i32;
                    writer.write_sample(iv)?;
                }
            }
        }
        hound::SampleFormat::Float => {
            for f in 0..frames {
                for ch in 0..channels as usize {
                    writer.write_sample(buf.samples[ch][f].clamp(-1.0, 1.0))?;
                }
            }
        }
    }
    writer.finalize()?;
    Ok(())
}
