//! Repair modules: DSP (A route) and neural (B route, UniverSR via ONNX).

pub mod common;
pub mod dsp;
pub mod neural;
pub mod universr;

pub use dsp::repair as dsp_repair;
