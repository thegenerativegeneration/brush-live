use super::*;
use rand::{RngExt as _, SeedableRng};

/// A seeded scene: flat patches with jittered axes, round blobs, faint and
/// broken Gaussians, a spread of information so the percentiles matter.
/// Every Gaussian carries information: voxels without any are where the
/// legacy round and the split path differ on purpose (they stay uninformed
/// or keep earlier bytes instead of mapping to 255).
fn fixture(seed: u64, n: usize) -> Vec<GaussianScore> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|i| {
            let pos = Vec3::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-0.5..0.5),
                if i % 3 == 0 {
                    rng.random_range(-1.0..1.0)
                } else {
                    0.3 + rng.random_range(-0.01..0.01)
                },
            );
            let opacity = match i % 17 {
                0 => 0.02,
                1 => f32::NAN,
                _ => rng.random_range(0.05..1.0),
            };
            let s = 10f32.powf(rng.random_range(-3.0..-1.0));
            let mut fisher_pos = info(s);
            fisher_pos[1] = rng.random_range(-0.1..0.1) / (s * s);
            fisher_pos[3] = fisher_pos[1];
            let axis = if i % 3 == 0 {
                Vec3::new(rng.random(), rng.random(), rng.random())
            } else {
                Vec3::new(
                    rng.random_range(-0.1..0.1),
                    rng.random_range(-0.1..0.1),
                    1.0,
                )
            };
            GaussianScore {
                pos: if i % 41 == 0 { Vec3::NAN } else { pos },
                opacity,
                coverage: if i % 29 == 0 { f32::NAN } else { rng.random() },
                fisher_pos,
                axis,
                flatness: rng.random(),
            }
        })
        .collect()
}

fn cameras() -> Vec<ViewCone> {
    vec![
        cam([0.0, 0.0, 3.0], [0.0, 0.0, -1.0]),
        cam([2.0, 0.0, 2.0], [-1.0, 0.0, -1.0]),
        cam([0.0, 0.0, -3.0], [0.0, 0.0, 1.0]),
    ]
}

fn sorted(mut cells: Vec<Cell>) -> Vec<Cell> {
    cells.sort_by(|a, b| a.center.partial_cmp(&b.center).expect("finite centres"));
    cells
}

const PROD_SCALE: UncertaintyScale = UncertaintyScale {
    ridge: FisherRidge {
        abs: 1e-6,
        rel: 1e-3,
    },
    sigma_pix: 0.05,
};

#[test]
fn cells_between_fisher_passes_keep_the_last_bytes_and_mark_new_voxels_uninformed() {
    let cams = cameras();
    let mut agg = VoxelAggregator::new(0.1, 0.1, PROD_SCALE);
    let gs = fixture(5, 3000);
    let with_fisher = sorted(agg.aggregate(&gs, &cams, 0.0));
    // The splats move on: shift everything by one voxel along x, so most
    // voxels are the same keys and the last column is new.
    let moved: Vec<SplatGeom> = gs
        .iter()
        .map(|g| SplatGeom {
            pos: g.pos + Vec3::new(0.1, 0.0, 0.0),
            ..g.geom()
        })
        .collect();
    let later = agg.cells(&moved, &cams, 3.0);
    let by_key = |cells: &[Cell]| -> HashMap<IVec3, Cell> {
        cells
            .iter()
            .map(|c| ((Vec3::from(c.center) / 0.1).floor().as_ivec3(), *c))
            .collect()
    };
    let before = by_key(&with_fisher);
    let mut new_keys = 0;
    for (key, c) in by_key(&later) {
        match before.get(&key) {
            Some(b) => {
                assert_eq!(
                    (c.coverage, c.uncertainty),
                    (b.coverage, b.uncertainty),
                    "{key}"
                );
            }
            None => {
                new_keys += 1;
                assert_eq!(
                    (c.coverage, c.uncertainty, c.uninformed),
                    (UNINFORMED_COVERAGE, UNINFORMED_UNCERTAINTY, true),
                    "{key}"
                );
            }
        }
    }
    assert!(new_keys > 0);
}

#[test]
fn voxels_before_any_fisher_pass_are_uninformed() {
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    let cells = agg.cells(&[g([0.5; 3], 1.0, 1.0, 0.1).geom()], &[], 0.0);
    assert_eq!(cells.len(), 1);
    assert_eq!(cells[0].coverage, 0);
    assert_eq!(cells[0].uncertainty, 255);
    assert!(cells[0].uninformed);
}

#[test]
fn a_pass_that_does_not_observe_a_voxel_leaves_its_bytes_alone() {
    let mut agg = VoxelAggregator::new(1.0, 0.0, SCALE);
    let ladder = |unseen: Option<usize>| -> Vec<GaussianScore> {
        (0..20)
            .map(|i| {
                let s = if Some(i) == unseen {
                    f32::INFINITY
                } else {
                    0.01 * 1.2f32.powi(i as i32)
                };
                g([i as f32 + 0.5, 0.5, 0.5], 1.0, 0.5, s)
            })
            .collect()
    };
    let first = agg.aggregate(&ladder(None), &[], 0.0);
    // Voxel 18 is near the top of the ranking; the next pass sees none of it.
    let second = agg.aggregate(&ladder(Some(18)), &[], 1.0);
    let at = |cells: &[Cell], x: usize| *cells.iter().find(|c| c.center[0] as usize == x).unwrap();
    assert_eq!(
        at(&second, 18),
        Cell {
            age: 1,
            ..at(&first, 18)
        },
        "kept, not pulled to 255"
    );
    assert!(!at(&second, 18).uninformed);
    // A voxel unseen from its first pass on stays uninformed.
    let mut fresh = VoxelAggregator::new(1.0, 0.0, SCALE);
    let cells = fresh.aggregate(&ladder(Some(3)), &[], 0.0);
    assert!(at(&cells, 3).uninformed);
    assert!(cells.iter().filter(|c| c.uninformed).count() == 1);
}
