//! ODE solver: 4-step midpoint with CFG (classifier-free guidance).
//!
//! Reference (TorchDiffeqSolver, method="midpoint", CFGVectorFieldODE):
//!   drift(x, t, y) = (1 - guidance_scale) * uncond(x, t) + guidance_scale * guided(x, t, y)
//!   midpoint: x_{n+1} = x_n + h * drift(x_n + 0.5*h*drift(x_n), t + 0.5*h)
//!
//! ts = linspace(0, 1, ode_steps+1) = [0, 0.25, 0.5, 0.75, 1.0] for 4 steps.

use crate::repair::universr::onnx_session::Sessions;
use anyhow::Result;

pub struct OdeSolver {
    pub steps: usize,
    pub guidance_scale: f32,
}

impl OdeSolver {
    pub fn new(steps: usize, guidance_scale: f32) -> Self {
        Self {
            steps,
            guidance_scale,
        }
    }

    /// Run the full ODE. `x0` is the initial noise [1,2,F=432,T].
    /// `y_lr` is the LR condition [1,2,Fy,T]. Returns final x [1,2,F,T].
    pub fn solve(
        &self,
        sess: &mut Sessions,
        x0: &[f32],
        x_shape: (usize, usize, usize, usize),
        y_lr: &[f32],
        y_shape: (usize, usize, usize, usize),
    ) -> Result<Vec<f32>> {
        let n = self.steps;
        let mut x = x0.to_vec();
        let (xb, xc, xf, xt) = x_shape;

        for i in 0..n {
            let t_i = i as f32 / n as f32;
            let t_next = (i + 1) as f32 / n as f32;
            let h = t_next - t_i;
            let t_mid = t_i + 0.5 * h;

            // k1 = drift(x, t_i)
            let k1 = self.drift(sess, &x, x_shape, t_i, y_lr, y_shape)?;
            // x_mid = x + 0.5*h*k1
            let x_mid: Vec<f32> = x
                .iter()
                .zip(k1.iter())
                .map(|(a, b)| a + 0.5 * h * b)
                .collect();
            // k2 = drift(x_mid, t_mid)
            let k2 = self.drift(sess, &x_mid, x_shape, t_mid, y_lr, y_shape)?;
            // x = x + h*k2
            for j in 0..x.len() {
                x[j] += h * k2[j];
            }
        }
        let _ = (xb, xc, xf, xt);
        Ok(x)
    }

    fn drift(
        &self,
        sess: &mut Sessions,
        x: &[f32],
        x_shape: (usize, usize, usize, usize),
        t: f32,
        y_lr: &[f32],
        y_shape: (usize, usize, usize, usize),
    ) -> Result<Vec<f32>> {
        let t_arr = [t];
        let (g, u) = sess.run_both(x, x_shape, &t_arr, y_lr, y_shape)?;
        // drift = (1 - gs) * uncond + gs * guided
        Ok(u.iter()
            .zip(g.iter())
            .map(|(&uu, &gg)| (1.0 - self.guidance_scale) * uu + self.guidance_scale * gg)
            .collect())
    }
}
