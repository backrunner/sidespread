//! Repair modules: DSP (A route) and neural (B route, UniverSR via ONNX).

pub mod artifacts;
pub mod bandwidth;
pub mod common;
pub mod dsp;
pub mod neural;
pub(crate) mod safety;
pub mod universr;

pub use dsp::repair as dsp_repair;
