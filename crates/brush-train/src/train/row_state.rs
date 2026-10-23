//! The fork's per-row state beside the splats (eviction life, origin tags, held rows, row budget), kept row-aligned
//! through one keep, pad and split, so a row edit cannot update one column and miss another.

use super::{SplatTrainer, radius};
use crate::budget::RowBudget;
use crate::evict::SplatLife;
use crate::origin::SplatOrigin;
use burn::tensor::{Bool, Int, Tensor};

/// Per-row structures the trainer keeps aligned with the splats; each is `None` while off.
#[derive(Default)]
#[cfg_attr(test, derive(Clone))]
pub(super) struct RowState {
    /// Age, importance and trained flag per row. `Some` only while eviction is on: created under `evict.is_some()`
    /// (`mark_trained`, `append_rows`), reset by `enable_eviction`.
    pub life: Option<SplatLife>,
    /// Seeding view per row, when tracked (`track_origins`).
    pub origin: Option<SplatOrigin>,
    /// Rows the caller holds still until the next refine ([`SplatTrainer::set_frozen_rows`]).
    pub held: Option<Tensor<1, Bool>>,
    /// Groups and their budgets for the next refine ([`SplatTrainer::set_row_budget`]).
    pub budget: Option<RowBudget>,
}

impl RowState {
    /// Keeps the rows at `indices` (inner device), in that order. `taken` flags, one per row before the keep, the rows
    /// that go; the budget needs it, so a budget without `taken` panics.
    pub fn keep(&mut self, indices: Tensor<1, Int>, taken: Option<&[bool]>) {
        if let Some(life) = self.life.as_mut() {
            life.keep(indices.clone());
        }
        if let Some(origin) = self.origin.as_mut() {
            origin.keep(indices.clone());
        }
        self.held = self.held.take().map(|h| h.select(0, indices));
        if let Some(budget) = self.budget.as_mut() {
            budget.remove(taken.expect("the row budget keeps rows by their take flags"));
        }
    }

    /// Appends `n` rows: new life, origin `slot`, not held, of no group.
    pub fn pad(&mut self, n: usize, slot: Option<u32>) {
        if let Some(life) = self.life.as_mut() {
            life.pad(n);
        }
        if let Some(origin) = self.origin.as_mut() {
            origin.pad(n, slot);
        }
        self.held = self.held.take().map(|h| radius::pad_unset(h, n));
        if let Some(budget) = self.budget.as_mut() {
            budget.pad(n);
        }
    }

    /// Appends one split child per index in `parents` (inner device): life and origin from the parent; the children
    /// are not held and of no group. Refine takes the held rows and the budget before it splits,
    /// so both are `None` here.
    pub fn split(&mut self, parents: Tensor<1, Int>) {
        let n = parents.dims()[0];
        if let Some(origin) = self.origin.as_mut() {
            origin.split(parents.clone());
        }
        if let Some(life) = self.life.as_mut() {
            life.split(parents);
        }
        self.held = self.held.take().map(|h| radius::pad_unset(h, n));
        if let Some(budget) = self.budget.as_mut() {
            budget.pad(n);
        }
    }
}

impl SplatTrainer {
    /// Keeps the rows at `indices` (inner device) in the per-row state and the refine record ([`RowState::keep`]).
    pub(super) fn keep_rows(&mut self, indices: Tensor<1, Int>, taken: Option<&[bool]>) {
        self.refine_record = self.refine_record.take().map(|r| r.keep(indices.clone()));
        self.rows.keep(indices, taken);
    }

    /// Appends `n` rows with origin `slot` to the per-row state and the refine record ([`RowState::pad`]).
    pub(super) fn pad_rows(&mut self, n: usize, slot: Option<u32>) {
        self.refine_record = self.refine_record.take().map(|r| r.pad(n));
        self.rows.pad(n, slot);
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use crate::budget::{GroupBudget, NO_GROUP};
    use burn::tensor::{Device, TensorData};

    async fn floats(t: Tensor<1>) -> Vec<f32> {
        t.into_data_async().await.expect("readback").try_to_vec::<f32>().expect("f32")
    }

    async fn flags(t: Tensor<1, Bool>) -> Vec<bool> {
        let data = t.int().into_data_async().await.expect("readback");
        data.convert::<i32>().try_to_vec::<i32>().expect("int").into_iter().map(|v| v != 0).collect()
    }

    fn ints(v: &[i32], device: &Device) -> Tensor<1, Int> {
        Tensor::from_data(TensorData::new(v.to_vec(), [v.len()]), device)
    }

    #[tokio::test]
    async fn row_state_ops_keep_every_column_aligned() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let mut life = SplatLife::new(6, &device);
        life.importance = Tensor::from_data(TensorData::new(vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], [6]), &device);
        let held = [true, false, true, false, true, false];
        let budget = RowBudget {
            group: vec![0, 1, 0, 1, 0, 1],
            groups: vec![GroupBudget { max_rows: u32::MAX, full: false, evictable: 0 }; 2],
            ..RowBudget::default()
        };
        let mut rows = RowState {
            life: Some(life),
            origin: Some(SplatOrigin::new(6, 1, &device)),
            held: Some(Tensor::from_data(TensorData::new(held.to_vec(), [6]), &device)),
            budget: Some(budget),
        };

        rows.pad(2, Some(3));
        let life = rows.life.as_ref().unwrap();
        assert_eq!((life.len(), life.importance.dims()[0], life.trained.dims()[0]), (8, 8, 8));
        let origin = rows.origin.as_ref().unwrap().slots().await;
        assert_eq!(origin, [vec![Some(1); 6], vec![Some(3); 2]].concat());
        let held_now = flags(rows.held.clone().unwrap()).await;
        assert_eq!(held_now, [&held[..], &[false, false][..]].concat());
        assert_eq!(rows.budget.as_ref().unwrap().group, vec![0, 1, 0, 1, 0, 1, NO_GROUP, NO_GROUP]);

        let mut taken = vec![true; 8];
        for i in [0, 2, 7] {
            taken[i] = false;
        }
        rows.keep(ints(&[0, 2, 7], &device), Some(&taken));
        let life = rows.life.as_ref().unwrap();
        assert_eq!(life.len(), 3);
        assert_eq!(floats(life.importance.clone()).await, vec![0.0, 2.0, f32::INFINITY]);
        assert_eq!(rows.origin.as_ref().unwrap().slots().await, vec![Some(1), Some(1), Some(3)]);
        assert_eq!(flags(rows.held.clone().unwrap()).await, vec![true, true, false]);
        assert_eq!(rows.budget.as_ref().unwrap().group, vec![0, 0, NO_GROUP]);

        rows.split(ints(&[1], &device));
        let life = rows.life.as_ref().unwrap();
        assert_eq!(floats(life.importance.clone()).await, vec![0.0, 2.0, f32::INFINITY, 2.0]);
        assert_eq!(floats(life.age.clone()).await.len(), 4);
        assert_eq!(rows.origin.as_ref().unwrap().slots().await, vec![Some(1), Some(1), Some(3), Some(1)]);
        assert_eq!(flags(rows.held.clone().unwrap()).await, vec![true, true, false, false]);
        assert_eq!(rows.budget.as_ref().unwrap().group, vec![0, 0, NO_GROUP, NO_GROUP]);
    }
}
