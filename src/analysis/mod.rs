//! Analysis: STFT, spectrum, segmentation, and deficiency detection.

pub mod bandwidth;
pub mod defects;
pub mod detector;
pub mod segment;
pub mod spectrum;
pub mod stft;

pub use detector::{analyze, SegmentMetrics, SegmentReport};
