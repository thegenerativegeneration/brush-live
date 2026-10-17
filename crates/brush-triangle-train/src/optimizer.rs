use brush_recon::ReconError;
use burn::tensor::Tensor;

pub struct DeviceAdam<const D: usize> {
    m: Tensor<D>,
    v: Tensor<D>,
    step: u64,
    readbacks: u64,
}
impl<const D: usize> DeviceAdam<D> {
    pub fn new(value: &Tensor<D>) -> Self {
        let d = value.device().inner();
        Self {
            m: Tensor::zeros(value.dims(), &d),
            v: Tensor::zeros(value.dims(), &d),
            step: 0,
            readbacks: 0,
        }
    }
    pub fn step_count(&self) -> u64 {
        self.step
    }
    pub fn scalar_readbacks(&self) -> u64 {
        self.readbacks
    }
    pub async fn update(
        &mut self,
        value: &Tensor<D>,
        gradient: Tensor<D>,
        lr: f32,
    ) -> Result<Tensor<D>, ReconError> {
        if !lr.is_finite()
            || lr <= 0.
            || value.dims() != self.m.dims()
            || gradient.dims() != value.dims()
        {
            return Err(ReconError::input(
                "invalid Adam rate or parameter/gradient shape",
            ));
        }
        let step = self
            .step
            .checked_add(1)
            .ok_or_else(|| ReconError::ModelState("Adam step overflow".into()))?;
        let gradient = gradient.detach().without_autodiff();
        let value = value.clone().detach().without_autodiff();
        let m = self.m.clone() * 0.9 + gradient.clone() * 0.1;
        let v = self.v.clone() * 0.999 + gradient.powi_scalar(2) * 0.001;
        let corrected_m = m.clone() / (1. - 0.9_f64.powf(step as f64)) as f32;
        let corrected_v = v.clone() / (1. - 0.999_f64.powf(step as f64)) as f32;
        let next = value - corrected_m / (corrected_v.sqrt() + 1e-8) * lr;
        let invalid = |t: Tensor<D>| t.clone().is_nan().bool_or(t.is_inf()).float().sum();
        let flag = invalid(m.clone()) + invalid(v.clone()) + invalid(next.clone());
        self.readbacks += 1;
        let flag = flag
            .into_data_async()
            .await
            .map_err(|e| ReconError::Gpu(format!("Adam finite-state readback: {e:?}")))?
            .try_to_vec::<f32>()
            .map_err(|e| ReconError::Gpu(format!("Adam finite-state scalar: {e:?}")))?[0];
        if flag != 0. {
            return Err(ReconError::ModelState(format!(
                "nonfinite Adam state at proposed step {step}"
            )));
        }
        self.m = m.detach();
        self.v = v.detach();
        self.step = step;
        Ok(next.detach())
    }
}

pub fn cap_vertex_displacement(
    old: Tensor<2>,
    proposal: Tensor<2>,
    max_m: f32,
) -> Result<Tensor<2>, ReconError> {
    if !max_m.is_finite() || max_m <= 0. || old.dims() != proposal.dims() || old.dims()[1] != 3 {
        return Err(ReconError::input(
            "invalid vertex displacement limit or shape",
        ));
    }
    let old = old.without_autodiff();
    let delta = proposal.without_autodiff() - old.clone();
    let length = delta
        .clone()
        .powi_scalar(2)
        .sum_dim(1)
        .sqrt()
        .clamp_min(1e-12);
    Ok(old + delta * (length.recip() * max_m).clamp_max(1.))
}
