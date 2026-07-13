//! ONNX session management and model-shape validation.

use anyhow::{bail, Context, Result};
use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionOutputs};
use ort::value::{Value, ValueType};
use std::path::Path;

pub struct Sessions {
    model: Session,
    pub hr_bins: usize,
    pub condition_bins: usize,
    pub time_frames: usize,
}

impl Sessions {
    pub fn load(model_path: &Path) -> Result<Self> {
        let model = Session::builder()
            .map_err(|error| anyhow::anyhow!("creating ONNX session: {error:?}"))?
            .with_execution_providers([ort::ep::CPUExecutionProvider::default()
                .with_arena_allocator(false)
                .build()])
            .map_err(|error| anyhow::anyhow!("configuring ONNX CPU allocator: {error:?}"))?
            .with_intra_threads(1)
            .map_err(|error| anyhow::anyhow!("configuring ONNX intra-op threads: {error:?}"))?
            .with_inter_threads(1)
            .map_err(|error| anyhow::anyhow!("configuring ONNX inter-op threads: {error:?}"))?
            .with_memory_pattern(false)
            .map_err(|error| anyhow::anyhow!("configuring ONNX memory pattern: {error:?}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| anyhow::anyhow!("configuring ONNX session: {error:?}"))?
            .commit_from_file(model_path)
            .with_context(|| format!("loading UniverSR ONNX model: {}", model_path.display()))?;
        let x_shape = input_shape(&model, "x")?;
        let y_shape = input_shape(&model, "y")?;
        if x_shape.len() != 4 || y_shape.len() != 4 {
            bail!("UniverSR ONNX inputs must be four-dimensional");
        }
        if x_shape[0] != 1 || x_shape[1] != 2 || y_shape[0] != 1 || y_shape[1] != 2 {
            bail!("UniverSR ONNX model must use [1,2,F,T] tensors");
        }
        if x_shape[3] != y_shape[3] {
            bail!("UniverSR x/y time dimensions do not match");
        }
        for output in ["guided", "unconditioned"] {
            if !model
                .outputs()
                .iter()
                .any(|candidate| candidate.name() == output)
            {
                bail!("UniverSR ONNX model has no `{output}` output");
            }
        }

        Ok(Self {
            model,
            hr_bins: x_shape[2],
            condition_bins: y_shape[2],
            time_frames: x_shape[3],
        })
    }

    pub fn run_both(
        &mut self,
        x: &[f32],
        x_shape: (usize, usize, usize, usize),
        t: &[f32],
        y: &[f32],
        y_shape: (usize, usize, usize, usize),
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.validate_shapes(x_shape, y_shape)?;
        let x_array = Array4::from_shape_vec(x_shape, x.to_vec()).context("x shape")?;
        let t_array = ndarray::Array1::from_vec(t.to_vec());
        let y_array = Array4::from_shape_vec(y_shape, y.to_vec()).context("condition shape")?;
        let outputs: SessionOutputs = self.model.run(ort::inputs![
            "x" => Value::from_array(x_array)?,
            "t" => Value::from_array(t_array)?,
            "y" => Value::from_array(y_array)?
        ])?;
        let guided = outputs["guided"].try_extract_tensor::<f32>()?.1.to_vec();
        let unconditioned = outputs["unconditioned"]
            .try_extract_tensor::<f32>()?
            .1
            .to_vec();
        Ok((guided, unconditioned))
    }

    fn validate_shapes(
        &self,
        x_shape: (usize, usize, usize, usize),
        y_shape: (usize, usize, usize, usize),
    ) -> Result<()> {
        if x_shape != (1, 2, self.hr_bins, self.time_frames) {
            bail!("unexpected UniverSR x shape {x_shape:?}");
        }
        if y_shape != (1, 2, self.condition_bins, self.time_frames) {
            bail!("unexpected UniverSR condition shape {y_shape:?}");
        }
        Ok(())
    }
}

fn input_shape(session: &Session, name: &str) -> Result<Vec<usize>> {
    let input = session
        .inputs()
        .iter()
        .find(|input| input.name() == name)
        .with_context(|| format!("ONNX model has no `{name}` input"))?;
    let ValueType::Tensor { shape, .. } = input.dtype() else {
        bail!("ONNX input `{name}` is not a tensor");
    };
    shape
        .iter()
        .map(|dimension| {
            if *dimension <= 0 {
                bail!("dynamic ONNX input dimensions are not supported for `{name}`")
            } else {
                Ok(*dimension as usize)
            }
        })
        .collect()
}
