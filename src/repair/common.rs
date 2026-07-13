//! Shared helpers: crossfade, phase jitter, smoothstep, band-merge mask.

/// 3rd-order smoothstep: 3t^2 - 2t^3, t in [0,1] clamped.
#[inline]
pub fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Linear crossfade between `a` and `b` over `n` samples, writing into `out`.
pub fn crossfade(a: &[f32], b: &[f32], n: usize, out: &mut [f32]) {
    let n = n.min(a.len()).min(b.len()).min(out.len());
    for i in 0..n {
        let t = i as f32 / n as f32;
        out[i] = a[i] * (1.0 - t) + b[i] * t;
    }
}

/// Smoothstep band-merge mask value for bin `b`: 0 below `lo`, 1 above `hi`, smoothstep between.
pub fn band_mask(b: usize, lo: usize, hi: usize) -> f32 {
    if b <= lo {
        0.0
    } else if b >= hi {
        1.0
    } else {
        smoothstep((b - lo) as f32 / (hi - lo) as f32)
    }
}

/// Deterministic per-bin phase jitter in radians, smoothly varying across bins.
/// `max_rad` is the amplitude (e.g. 30° = 0.524 rad).
pub fn phase_jitter(b: usize, max_rad: f32) -> f32 {
    // Smooth low-rate sinusoid across bins to avoid per-bin discontinuities.
    let s = (b as f32 * 0.7).sin() + (b as f32 * 0.13).sin();
    (s * 0.5) * max_rad
}
