//! Sliding analysis windows and content-aware repair boundary selection.

#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct AdaptiveBoundary {
    pub index: usize,
    pub fade_samples: usize,
    pub score: f32,
}

/// Produce a list of segments covering `len` samples, each `segment_ms` long with `overlap` fractional overlap.
pub fn segments(len: usize, sample_rate: u32, segment_ms: usize, overlap: f32) -> Vec<Segment> {
    let seg_len = (segment_ms * sample_rate as usize) / 1000;
    if seg_len == 0 || len == 0 {
        return Vec::new();
    }
    let hop = ((seg_len as f32) * (1.0 - overlap)).round().max(1.0) as usize;
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < len {
        let end = (start + seg_len).min(len);
        out.push(Segment { start, end });
        if end >= len {
            break;
        }
        start += hop;
    }
    out
}

/// Select the safest repair boundary inside an inclusive search range.
///
/// Low local energy, low derivative energy, stable L/R correlation, and a small
/// instantaneous amplitude are preferred. The returned fade length grows when
/// even the best available boundary remains active or unstable.
pub fn select_safe_boundary(
    mid: &[f32],
    side: &[f32],
    search_start: usize,
    search_end: usize,
    sample_rate: u32,
) -> AdaptiveBoundary {
    let length = mid.len().min(side.len());
    if length == 0 {
        return AdaptiveBoundary {
            index: 0,
            fade_samples: 0,
            score: 0.0,
        };
    }
    let start = search_start.min(length - 1);
    let end = search_end.max(start).min(length - 1);
    if start == 0 && end == 0 {
        return edge_boundary(0);
    }
    if start == length - 1 && end == length - 1 {
        return edge_boundary(length);
    }

    let radius = (sample_rate as usize / 500).max(16);
    let step = (sample_rate as usize / 2_000).max(1);
    let mut candidates = Vec::new();
    let mut index = start;
    loop {
        candidates.push(boundary_features(mid, side, index, radius));
        if index >= end {
            break;
        }
        index = (index + step).min(end);
    }

    let ranges = [
        feature_range(&candidates, |feature| feature.energy),
        feature_range(&candidates, |feature| feature.derivative),
        feature_range(&candidates, |feature| feature.correlation_jump),
        feature_range(&candidates, |feature| feature.instantaneous),
    ];
    let mut best_index = start;
    let mut best_score = f32::INFINITY;
    for feature in candidates {
        let score = 0.40 * normalize(feature.energy, ranges[0])
            + 0.30 * normalize(feature.derivative, ranges[1])
            + 0.20 * normalize(feature.correlation_jump, ranges[2])
            + 0.10 * normalize(feature.instantaneous, ranges[3]);
        if score < best_score {
            best_score = score;
            best_index = feature.index;
        }
    }

    let minimum_fade = sample_rate as usize * 15 / 1_000;
    let maximum_fade = sample_rate as usize * 60 / 1_000;
    let fade_samples = minimum_fade
        + ((maximum_fade - minimum_fade) as f32 * best_score.clamp(0.0, 1.0)).round() as usize;
    AdaptiveBoundary {
        index: best_index,
        fade_samples,
        score: best_score,
    }
}

#[derive(Debug, Clone, Copy)]
struct BoundaryFeatures {
    index: usize,
    energy: f32,
    derivative: f32,
    correlation_jump: f32,
    instantaneous: f32,
}

fn boundary_features(mid: &[f32], side: &[f32], index: usize, radius: usize) -> BoundaryFeatures {
    let length = mid.len().min(side.len());
    let start = index.saturating_sub(radius);
    let end = (index + radius + 1).min(length);
    let mut energy = 0.0f64;
    let mut derivative = 0.0f64;
    for sample in start..end {
        energy += (mid[sample] * mid[sample] + side[sample] * side[sample]) as f64;
        if sample > start {
            derivative += ((mid[sample] - mid[sample - 1]).powi(2)
                + (side[sample] - side[sample - 1]).powi(2)) as f64;
        }
    }
    let sample_count = (end - start).max(1) as f64;
    let derivative_count = (end - start).saturating_sub(1).max(1) as f64;
    let before = stereo_correlation(mid, side, start, index.max(start + 1));
    let after = stereo_correlation(mid, side, index.min(end - 1), end);
    BoundaryFeatures {
        index,
        energy: (energy / sample_count) as f32,
        derivative: (derivative / derivative_count) as f32,
        correlation_jump: (before - after).abs(),
        instantaneous: mid[index].powi(2) + side[index].powi(2),
    }
}

fn stereo_correlation(mid: &[f32], side: &[f32], start: usize, end: usize) -> f32 {
    if end <= start {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut left_energy = 0.0f64;
    let mut right_energy = 0.0f64;
    for index in start..end {
        let left = (mid[index] + side[index]) as f64;
        let right = (mid[index] - side[index]) as f64;
        dot += left * right;
        left_energy += left * left;
        right_energy += right * right;
    }
    let denominator = (left_energy * right_energy).sqrt();
    if denominator <= 1e-12 {
        0.0
    } else {
        (dot / denominator) as f32
    }
}

fn feature_range(
    candidates: &[BoundaryFeatures],
    value: impl Fn(&BoundaryFeatures) -> f32,
) -> (f32, f32) {
    candidates.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(minimum, maximum), candidate| {
            let value = value(candidate);
            (minimum.min(value), maximum.max(value))
        },
    )
}

fn normalize(value: f32, (minimum, maximum): (f32, f32)) -> f32 {
    if !value.is_finite() || maximum - minimum <= 1e-12 {
        0.0
    } else {
        ((value - minimum) / (maximum - minimum)).clamp(0.0, 1.0)
    }
}

fn edge_boundary(index: usize) -> AdaptiveBoundary {
    AdaptiveBoundary {
        index,
        fade_samples: 0,
        score: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_full_signal() {
        let segs = segments(48000, 48000, 80, 0.5);
        assert!(!segs.is_empty());
        assert_eq!(segs.last().unwrap().end, 48000);
        // 80ms @ 48k = 3840 samples, 50% overlap → hop 1920.
        assert_eq!(segs[0].start, 0);
        assert_eq!(segs[0].end, 3840);
        assert_eq!(segs[1].start, 1920);
    }

    #[test]
    fn empty_input() {
        assert!(segments(0, 48000, 80, 0.5).is_empty());
    }

    #[test]
    fn safe_boundary_avoids_an_active_transient() {
        let sample_rate = 48_000;
        let length = 4_000;
        let mut mid = vec![0.0f32; length];
        let mut side = vec![0.0f32; length];
        for index in 1_900..2_100 {
            let phase = 2.0 * std::f32::consts::PI * 10_000.0 * index as f32 / sample_rate as f32;
            mid[index] = phase.sin() * 0.8;
            side[index] = phase.cos() * 0.4;
        }

        let boundary = select_safe_boundary(&mid, &side, 1_700, 2_300, sample_rate);
        assert!(boundary.index < 1_900 || boundary.index >= 2_100);
        assert!((720..=2_880).contains(&boundary.fade_samples));
    }

    #[test]
    fn signal_edges_need_no_crossfade() {
        let signal = vec![0.1f32; 1024];
        assert_eq!(
            select_safe_boundary(&signal, &signal, 0, 0, 48_000).fade_samples,
            0
        );
        assert_eq!(
            select_safe_boundary(&signal, &signal, 1023, 1023, 48_000).index,
            1024
        );
    }
}
