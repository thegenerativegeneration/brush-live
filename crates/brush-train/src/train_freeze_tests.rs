//! Rows the caller holds ([`SplatTrainer::set_frozen_rows`], area training's left tiles): no step updates them, and
//! refine treats them as rows no step trained (no decay, densification or floor bake), with or without the training
//! radius; the mask follows take and append, refine drops it.

use super::radius_tests::{
    FAR, FAR_ROWS, Hold, NEAR, assert_split_and_decayed, assert_untrained_rows_keep_raw_params_and_floor, batch,
    bits, device, far_rows_mask, floats, masked_count, refine_after_steps, sigmoid, splats, step, trainer,
    trainer_after_one_step,
};
use super::*;
use burn::module::Module;

/// Held rows keep their parameters, Adam moments and means (no position noise at opacity 0.05) through a step with
/// the radius off; the other rows update, and the radius count stays the radius'.
#[tokio::test]
async fn held_rows_keep_params_moments_and_means() {
    let (mut trainer, s) = trainer_after_one_step(0.0, 0.05).await;
    let device = s.device();
    trainer.set_frozen_rows(Some(far_rows_mask(NEAR + FAR + 1, &device)));
    let (before, after, _) = step(&mut trainer, s).await;
    assert_eq!(before.rows(FAR_ROWS), after.rows(FAR_ROWS), "held splats moved");
    assert_eq!(before.far_sh_rest(), after.far_sh_rest());
    for ((name, b), (_, a)) in before.rows(0..NEAR).iter().zip(after.rows(0..NEAR)) {
        assert_ne!(b, &a, "near splats' {name} did not update");
    }
    assert!(trainer.masked_by_radius().is_none());
}

/// Held rows and the radius together: rows either one flags keep their state.
#[tokio::test]
async fn held_rows_and_the_radius_both_freeze() {
    // Radius 5 m freezes the far rows; the held mask adds the first near row.
    let (mut trainer, s) = trainer_after_one_step(5.0, 0.9).await;
    let flags: Vec<bool> = (0..NEAR + FAR + 1).map(|i| i == 0).collect();
    let n = flags.len();
    trainer.set_frozen_rows(Some(Tensor::from_data(TensorData::new(flags, [n]), &s.device().inner())));
    let (before, after, _) = step(&mut trainer, s).await;
    assert_eq!(before.rows(FAR_ROWS), after.rows(FAR_ROWS), "far splats moved");
    assert_eq!(before.rows(0..1), after.rows(0..1), "the held near splat moved");
    assert_ne!(before.rows(1..2), after.rows(1..2), "an unheld near splat did not update");
    assert_eq!(masked_count(&trainer).await, Some(FAR as i64), "the radius count holds only the radius' rows");
}

/// Refine with the radius off: held rows are not decayed, split, grown or picked as replacement parents.
#[tokio::test]
async fn held_rows_are_not_decayed_or_densified_at_refine() {
    assert_held_rows_refine_untouched(Hold::FromStart).await;
}

/// Rows held after steps trained them refine as rows no step trained: the walker left mid-window.
#[tokio::test]
async fn rows_held_mid_window_refine_as_untrained() {
    assert_held_rows_refine_untouched(Hold::AfterSteps).await;
}

async fn assert_held_rows_refine_untouched(hold: Hold) {
    let r = refine_after_steps(0.0, hold).await;
    assert!(r.trained_column, "holding rows builds the trained column");
    assert_eq!(r.stats.num_pruned, 1);
    assert_eq!(r.opac_after.len(), 2 * NEAR + FAR, "only the near splats split");
    assert_eq!(bits(&r.opac[FAR_ROWS]), bits(&r.opac_after[FAR_ROWS]), "held opacity changed");
    let far_t = NEAR * 10..(NEAR + FAR) * 10;
    assert_eq!(bits(&r.transforms[far_t.clone()]), bits(&r.transforms_after[far_t]), "held splats split");
    assert_split_and_decayed(&r, 0..NEAR);
    assert!(sigmoid(r.opac[0]) > 0.5);
}

/// Held rows keep their raw params and Mip floor across a refine, the radius off.
#[tokio::test]
async fn held_rows_keep_raw_params_and_floor_across_refine() {
    assert_untrained_rows_keep_raw_params_and_floor(0.0, true).await;
}

/// Refine drops the mask (its rows change); without a mask and the radius off, no trained column is built.
#[tokio::test]
async fn refine_drops_the_mask_and_none_is_todays_path() {
    let device = device().await;
    let mut trainer = trainer(0.0, &device);
    trainer.set_frozen_rows(Some(far_rows_mask(NEAR + FAR + 1, &device)));
    let s = trainer.step(batch(), splats(0.9, &device).train()).await.0;
    let s = trainer.refine(1, s.valid()).await.0;
    assert!(trainer.frozen_rows.is_none(), "refine kept the mask");
    trainer.set_frozen_rows(None);
    trainer.step(batch(), s.train()).await;
    assert!(trainer.refine_record.as_ref().unwrap().trained.is_none(), "no mask, no trained column");
}

/// The mask follows take and append: taking two near rows shifts the held rows down, the rows appended back are not
/// held, and a step keeps the held rows' state and trains the appended ones.
#[tokio::test]
async fn the_mask_follows_take_and_append() {
    let (mut trainer, s) = trainer_after_one_step(0.0, 0.9).await;
    let device = s.device();
    trainer.set_frozen_rows(Some(far_rows_mask(NEAR + FAR + 1, &device)));
    let n = NEAR + FAR + 1;
    let take: Vec<bool> = (0..n).map(|i| i < 2).collect();
    let (s, taken) = trainer.take_rows(s.valid(), &take).await;
    let taken = taken.expect("two rows");
    let s = trainer.append_rows(s, &taken);
    let mask: Vec<bool> = trainer
        .frozen_rows
        .clone()
        .unwrap()
        .int()
        .into_data_async()
        .await
        .unwrap()
        .convert::<i32>()
        .try_into_vec::<i32>()
        .unwrap()
        .into_iter()
        .map(|v| v != 0)
        .collect();
    let want: Vec<bool> = (0..n).map(|i| (NEAR - 2..NEAR - 2 + FAR).contains(&i)).collect();
    assert_eq!(mask, want);
    let s = s.train();
    let before = floats(s.transforms.val().inner()).await;
    let (s, _) = trainer.step(batch(), s).await;
    let after = floats(s.transforms.val().inner()).await;
    let held = (NEAR - 2) * 10..(NEAR - 2 + FAR) * 10;
    assert_eq!(bits(&before[held.clone()]), bits(&after[held]), "held rows moved after take and append");
    let appended = (n - 2) * 10..n * 10;
    assert_ne!(bits(&before[appended.clone()]), bits(&after[appended]), "appended rows did not train");
}

/// Rows no longer held train again: after the mask is cleared, the next step marks every row trained.
#[tokio::test]
async fn cleared_rows_count_as_trained_again() {
    let device = device().await;
    let mut trainer = trainer(0.0, &device);
    trainer.set_frozen_rows(Some(far_rows_mask(NEAR + FAR + 1, &device)));
    let s = trainer.step(batch(), splats(0.9, &device).train()).await.0;
    trainer.set_frozen_rows(None);
    trainer.step(batch(), s).await;
    let trained = floats(trainer.refine_record.as_ref().unwrap().trained.clone().unwrap()).await;
    assert!(trained.iter().all(|&t| t == 1.0), "{trained:?}");
}

/// A mask of the wrong length fails open in release builds: logged, dropped, every row trains (no panic).
#[tokio::test]
async fn a_mask_of_the_wrong_length_is_dropped() {
    if cfg!(debug_assertions) {
        return;
    }
    let (mut trainer, s) = trainer_after_one_step(0.0, 0.9).await;
    let device = s.device();
    trainer.set_frozen_rows(Some(far_rows_mask(NEAR + FAR, &device)));
    assert!(trainer.frozen_rows().is_none(), "set_frozen_rows kept a short mask");
    // A mask that fits when set but not at the step (rows changed behind the trainer's back) is dropped there.
    trainer.frozen_rows = Some(far_rows_mask(NEAR + FAR, &device).bool_not());
    let (before, after, _) = step(&mut trainer, s).await;
    assert!(trainer.frozen_rows().is_none());
    assert_ne!(before.rows(FAR_ROWS), after.rows(FAR_ROWS), "far rows did not train");
}

/// What the next refine evicts at least: down to the target with a seed shortfall, nothing without one.
#[tokio::test]
async fn pending_evictions_follow_the_seed_shortfall() {
    let (mut trainer, _) = trainer_after_one_step(0.0, 0.9).await;
    assert_eq!(trainer.pending_evictions(100), 0, "no eviction state");
    trainer.enable_eviction(crate::evict::EvictConfig {
        headroom: 0.1,
        min_age: 0,
        max_cell_fraction: 1.0,
        recent_refines: 2,
    });
    trainer.config.max_splats = 100;
    assert_eq!(trainer.pending_evictions(100), 0);
    trainer.note_seed_shortfall(1);
    assert_eq!(trainer.pending_evictions(100), 10);
    assert_eq!(trainer.pending_evictions(95), 5);
}

/// The trainer tells how many held rows no step ever trained: rows held from the first step stay untrained through
/// steps, rows a step trained count as trained for good, and rows appended back from the host keep their flag (count
/// as trained without one).
/// Without eviction state (where the flag lives) there is no count.
#[tokio::test]
async fn held_rows_no_step_ever_trained_are_counted() {
    let device = device().await;
    let n = NEAR + FAR + 1;
    let mut trainer = trainer(0.0, &device);
    trainer.set_frozen_rows(Some(far_rows_mask(n, &device)));
    let s = trainer.step(batch(), splats(0.9, &device).train()).await.0;
    assert!(trainer.held_untrained_rows().is_none(), "no eviction state, no flag");

    let mut trainer = super::radius_tests::trainer(0.0, &device);
    trainer.enable_eviction(crate::evict::EvictConfig {
        headroom: 0.1,
        min_age: 0,
        max_cell_fraction: 1.0,
        recent_refines: 2,
    });
    trainer.set_frozen_rows(Some(far_rows_mask(n, &device)));
    let s = trainer.step(batch(), s).await.0;
    let s = trainer.step(batch(), s).await.0;
    assert_eq!(held_untrained(&trainer).await, Some(FAR as i64));
    // Taking two held rows out and appending them back: they keep their flag and stay untrained.
    let take: Vec<bool> = (0..n).map(|i| i == NEAR || i == NEAR + 1).collect();
    let (s, taken) = trainer.take_rows(s.valid(), &take).await;
    let taken = taken.unwrap();
    assert_eq!(taken.trained, vec![false; 2]);
    let s = trainer.append_rows(s, &taken);
    let hold_all = |device: &Device| Some(Tensor::from_data(TensorData::new(vec![true; n], [n]), &device.clone().inner()));
    trainer.set_frozen_rows(hold_all(&device));
    assert_eq!(held_untrained(&trainer).await, Some(FAR as i64), "appended rows keep their flag");
    // Without flags (a version 1 tile file) they count as trained.
    let (s, taken) = trainer.take_rows(s, &(0..n).map(|i| i >= n - 2).collect::<Vec<_>>()).await;
    let s = trainer.append_rows(s, &crate::rows::SplatRows { trained: Vec::new(), ..taken.unwrap() });
    trainer.set_frozen_rows(hold_all(&device));
    assert_eq!(held_untrained(&trainer).await, Some(FAR as i64 - 2), "rows without flags count as trained");
    // Released for one step, every row has trained.
    trainer.set_frozen_rows(None);
    let _ = trainer.step(batch(), s.train()).await;
    trainer.set_frozen_rows(Some(Tensor::from_data(TensorData::new(vec![true; n], [n]), &device.inner())));
    assert_eq!(held_untrained(&trainer).await, Some(0));
}

async fn held_untrained(trainer: &SplatTrainer) -> Option<i64> {
    let count = trainer.held_untrained_rows()?;
    Some(count.into_data_async().await.unwrap().convert::<i64>().try_to_vec::<i64>().unwrap()[0])
}
