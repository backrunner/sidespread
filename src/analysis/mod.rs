//! Analysis: STFT, spectrum, segmentation, and deficiency detection.

pub mod detector;
pub mod segment;
pub mod spectrum;
pub mod stft;

pub use detector::{analyze, SegmentMetrics, SegmentReport};
