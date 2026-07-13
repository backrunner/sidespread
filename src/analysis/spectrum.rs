//! Spectral feature helpers: power, log power, normalization, mel basis.

/// Power spectrum (|X|^2) per bin for a frame.
pub fn power(spec: &super::stft::SpectrumFrame) -> Vec<f32> {
    (0..spec.n_bins)
        .map(|b| {
            let re = spec.re(b);
            let im = spec.im(b);
            re * re + im * im
        })
        .collect()
}

/// Log power spectrum (10*log10 with floor).
pub fn log_power(spec: &super::stft::SpectrumFrame, floor: f32) -> Vec<f32> {
    power(spec)
        .iter()
        .map(|&p| 10.0 * (p.max(floor)).log10())
        .collect()
}

/// Bin index for a given frequency at a given sample rate.
pub fn bin_of(hz: f32, n_fft: usize, sample_rate: u32) -> usize {
    ((hz * n_fft as f32) / sample_rate as f32).round() as usize
}

/// Frequency of a bin.
pub fn freq_of(bin: usize, n_fft: usize, sample_rate: u32) -> f32 {
    (bin as f32 * sample_rate as f32) / n_fft as f32
}
