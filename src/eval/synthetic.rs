//! Synthetic side-channel degradation for the `eval` command.

use crate::analysis::spectrum::freq_of;
use crate::analysis::stft::{istft, stft, StftConfig};

pub fn degrade_side(signal: &[f32], sample_rate: u32, cutoff_hz: f32) -> Vec<f32> {
    let config = StftConfig::new(4096, 1024);
    let mut spectrum = stft(signal, &config);
    let transition_hz = 500.0f32;
    let passband_end = (cutoff_hz - transition_hz).max(0.0);
    let stopband_start = cutoff_hz + transition_hz;

    for frame in &mut spectrum {
        for bin in 0..frame.n_bins {
            let frequency = freq_of(bin, config.n_fft, sample_rate);
            let gain = if frequency <= passband_end {
                1.0
            } else if frequency >= stopband_start {
                0.0
            } else {
                let position = (frequency - passband_end) / (stopband_start - passband_end);
                0.5 + 0.5 * (std::f32::consts::PI * position).cos()
            };
            frame.cplx[2 * bin] *= gain;
            frame.cplx[2 * bin + 1] *= gain;
        }
    }
    istft(&spectrum, &config, signal.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_high_frequency_tone() {
        let sample_rate = 48_000;
        let signal = (0..sample_rate)
            .map(|index| {
                (2.0 * std::f32::consts::PI * 12_000.0 * index as f32 / sample_rate as f32).sin()
                    * 0.5
            })
            .collect::<Vec<_>>();
        let degraded = degrade_side(&signal, sample_rate, 8000.0);
        let input_rms = rms(&signal);
        let output_rms = rms(&degraded);
        assert!(output_rms < input_rms * 0.05);
    }

    fn rms(signal: &[f32]) -> f32 {
        (signal.iter().map(|sample| sample * sample).sum::<f32>() / signal.len() as f32).sqrt()
    }
}
