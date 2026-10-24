use burn::tensor::{Bool, Device, Int, Tensor, TensorData, s};

/// Per-splat age (refines survived), importance and whether a step ever trained it, row-aligned with the splats.
#[cfg_attr(test, derive(Clone))]
pub(crate) struct SplatLife {
    pub age: Tensor<1>,
    /// `+inf` until scored.
    pub importance: Tensor<1>,
    /// 1 once a step updated the row (it was not frozen), else 0; rows appended back from the host take the flag they
    /// carry (trained without one), split children take their parent's.
    pub trained: Tensor<1>,
}

impl SplatLife {
    pub(crate) fn new(n: usize, device: &Device) -> Self {
        Self {
            age: Tensor::zeros([n], device),
            importance: Tensor::full([n], f32::INFINITY, device),
            trained: Tensor::zeros([n], device),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.age.dims()[0]
    }

    pub(crate) fn pad(&mut self, n: usize) {
        let device = self.age.device();
        self.age = Tensor::cat(vec![self.age.clone(), Tensor::zeros([n], &device)], 0);
        self.importance = Tensor::cat(
            vec![
                self.importance.clone(),
                Tensor::full([n], f32::INFINITY, &device),
            ],
            0,
        );
        self.trained = Tensor::cat(vec![self.trained.clone(), Tensor::zeros([n], &device)], 0);
    }

    pub(crate) fn keep(&mut self, indices: Tensor<1, Int>) {
        self.age = self.age.clone().select(0, indices.clone());
        self.importance = self.importance.clone().select(0, indices.clone());
        self.trained = self.trained.clone().select(0, indices);
    }

    /// Sets the age, importance and trained flag of the last `age.len()` rows (they come back from the host); empty
    /// `trained` marks them all trained.
    pub(crate) fn set_tail(&mut self, age: &[f32], importance: &[f32], trained: &[bool]) {
        let (k, len) = (age.len(), self.len());
        assert_eq!(importance.len(), k, "one importance per age");
        assert!(trained.is_empty() || trained.len() == k, "one trained flag per age, or none");
        assert!(k <= len, "{k} tail rows for {len} rows");
        if k == 0 {
            return;
        }
        let device = self.age.device();
        let tail = |v: &[f32]| Tensor::<1>::from_data(TensorData::new(v.to_vec(), [k]), &device);
        self.age = self.age.clone().slice_assign(s![len - k..len], tail(age));
        self.importance = self
            .importance
            .clone()
            .slice_assign(s![len - k..len], tail(importance));
        let flags = match trained {
            [] => Tensor::ones([k], &device),
            flags => tail(&flags.iter().map(|&t| if t { 1.0 } else { 0.0 }).collect::<Vec<_>>()),
        };
        self.trained = self.trained.clone().slice_assign(s![len - k..len], flags);
    }

    /// Appends one child per parent index: age 0, the parent's importance and trained flag.
    pub(crate) fn split(&mut self, parents: Tensor<1, Int>) {
        let n = parents.dims()[0];
        let device = self.age.device();
        let trained = self.trained.clone().select(0, parents.clone());
        self.trained = Tensor::cat(vec![self.trained.clone(), trained], 0);
        let inherited = self.importance.clone().select(0, parents);
        self.age = Tensor::cat(vec![self.age.clone(), Tensor::zeros([n], &device)], 0);
        self.importance = Tensor::cat(vec![self.importance.clone(), inherited], 0);
    }

    /// Mean importance per seeing view over the refine window; splats no view saw keep theirs.
    pub(crate) fn update_importance(&mut self, sum: Tensor<1>, views: Tensor<1>) {
        let seen = views.clone().greater_elem(0.0);
        let mean = sum.div(views.clamp_min(1.0));
        self.importance = self.importance.clone().mask_where(seen, mean);
    }

    /// Marks the rows a step updated as trained: every row, or the rows not set in `frozen`.
    pub(crate) fn mark_trained(&mut self, frozen: Option<&Tensor<1, Bool>>) {
        self.trained = match frozen {
            Some(f) => self.trained.clone().mask_fill(f.clone().bool_not(), 1.0),
            None => self.trained.clone().ones_like(),
        };
    }

    pub(crate) fn tick(&mut self) {
        self.age = self.age.clone() + 1.0;
    }
}
