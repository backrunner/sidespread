//! Segmentation: split a signal into overlapping fixed-length segments for per-segment analysis.

#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub start: usize,
    pub end: usize,
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
}
