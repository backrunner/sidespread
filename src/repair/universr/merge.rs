//! Band-merge: combine original side midband with UniverSR-generated highband.
//!
//! After UniverSR produces the full repaired side spectrum `[2, 512, T]`,
//! we keep the original side's bins below `fc_bin` and take only the
//! UniverSR bins above `fc_bin`, crossfading in a transition band.

use crate::repair::common::band_mask;

/// Merge original and repaired spectra in the `[2, n_bins, T_frames]` layout.
/// `orig` and `repaired` must have the same shape. `fc_bin` is the cutoff bin;
/// `transition` is the half-width of the crossfade band (in bins).
///
/// Layout assumption: [channel=2][freq=n_bins][time=T_frames] row-major f32,
/// matching `frontend::preprocess` output.
pub fn band_merge(
    out: &mut [f32],
    orig: &[f32],
    repaired: &[f32],
    n_bins: usize,
    t_frames: usize,
    fc_bin: usize,
    transition: usize,
) {
    let lo = fc_bin.saturating_sub(transition);
    let hi = (fc_bin + transition).min(n_bins - 1);
    for ch in 0..2 {
        for b in 0..n_bins {
            let m = band_mask(b, lo, hi);
            for t in 0..t_frames {
                let idx = (ch * n_bins + b) * t_frames + t;
                out[idx] = orig[idx] * (1.0 - m) + repaired[idx] * m;
            }
        }
    }
}
