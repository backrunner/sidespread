//! Evaluation modules: metrics, synthetic degradation, report generation.

pub mod metrics;
pub mod report;
pub mod synthetic;

pub use metrics::{
    EvaluationMetrics, MetricConfig, ProcessingMetrics, QualityMetrics, ReferenceMetrics,
};
