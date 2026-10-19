use super::{RowMoments, SplatRows};

#[test]
fn select_and_concat_keep_rows_together() {
    let rows = SplatRows {
        means: (0..9).map(|v| v as f32).collect(),
        rotations: (0..12).map(|v| v as f32).collect(),
        log_scales: (0..9).map(|v| -(v as f32)).collect(),
        sh_coeffs: (0..9).map(|v| v as f32 * 0.1).collect(),
        raw_opacities: vec![0.1, 0.2, 0.3],
        min_scale: vec![],
        origin: vec![Some(1), None, Some(3)],
        age: vec![1.0, 2.0, 3.0],
        importance: vec![0.5, f32::INFINITY, 0.7],
        sh_coeffs_per_channel: 1,
        ..SplatRows::default()
    };
    let picked = rows.select(&[2, 0]);
    assert_eq!(picked.means, vec![6.0, 7.0, 8.0, 0.0, 1.0, 2.0]);
    assert_eq!(picked.origin, vec![Some(3), Some(1)]);
    assert_eq!(picked.mean(1), glam::vec3(0.0, 1.0, 2.0));
    let joined = SplatRows::concat(&[rows.select(&[0]), rows.select(&[1, 2])]);
    assert_eq!(joined, rows);
    // A part with a floor and a part without: the missing floor is a zero one (no floor).
    let mut floored = rows.select(&[0]);
    floored.min_scale = vec![0.02];
    let mixed = SplatRows::concat(&[floored, rows.select(&[1, 2])]);
    assert_eq!(mixed.min_scale, vec![0.02, 0.0, 0.0]);
    assert!(SplatRows::concat(&[]).is_empty());
}

#[test]
fn check_names_the_first_column_out_of_length() {
    let good = SplatRows {
        means: vec![0.0; 6],
        rotations: vec![0.0; 8],
        log_scales: vec![0.0; 6],
        sh_coeffs: vec![0.0; 2 * 4 * 3],
        raw_opacities: vec![0.0; 2],
        min_scale: vec![],
        origin: vec![None; 2],
        age: vec![0.0; 2],
        importance: vec![0.0; 2],
        sh_coeffs_per_channel: 4,
        ..SplatRows::default()
    };
    assert_eq!(good.check(), Ok(()));
    let floored = SplatRows {
        min_scale: vec![0.1; 2],
        ..good.clone()
    };
    assert_eq!(floored.check(), Ok(()));
    let broken: [(&str, SplatRows); 8] = [
        (
            "means",
            SplatRows {
                means: vec![0.0; 5],
                ..good.clone()
            },
        ),
        (
            "rotations",
            SplatRows {
                rotations: vec![0.0; 4],
                ..good.clone()
            },
        ),
        (
            "log_scales",
            SplatRows {
                log_scales: vec![0.0; 3],
                ..good.clone()
            },
        ),
        (
            "sh_coeffs",
            SplatRows {
                sh_coeffs: vec![0.0; 2 * 3],
                ..good.clone()
            },
        ),
        (
            "min_scale",
            SplatRows {
                min_scale: vec![0.1],
                ..good.clone()
            },
        ),
        (
            "origin",
            SplatRows {
                origin: vec![None; 3],
                ..good.clone()
            },
        ),
        (
            "age",
            SplatRows {
                age: vec![],
                ..good.clone()
            },
        ),
        (
            "importance",
            SplatRows {
                importance: vec![0.0],
                ..good.clone()
            },
        ),
    ];
    for (column, rows) in broken {
        let err = rows.check().expect_err(column);
        assert!(err.contains(column), "{column}: {err}");
    }
    // Several columns off: the first one in field order is named.
    let both = SplatRows {
        age: vec![],
        origin: vec![],
        ..good
    };
    assert!(both.check().unwrap_err().contains("origin"));
}

/// `n` rows of moments for `coeffs` SH coefficients per channel, each value its index plus `offset`.
fn moments(n: usize, coeffs: usize, offset: f32) -> RowMoments {
    let ramp = |len: usize| (0..len).map(|v| v as f32 + offset).collect::<Vec<_>>();
    RowMoments {
        transforms_m1: ramp(n * 10),
        transforms_m2: ramp(n * 10),
        sh_m1: ramp(n * coeffs * 3),
        sh_m2: ramp(n),
        opacity_m1: ramp(n),
        opacity_m2: ramp(n),
    }
}

/// The trained flags and moments are rows like the parameters: [`SplatRows::check`] names an out-of-length one, and
/// select and concat keep them with their rows (parts without them get trained rows with zero moments).
#[test]
fn trained_flags_and_moments_are_row_columns() {
    let good = SplatRows {
        means: vec![0.0; 6],
        rotations: vec![0.0; 8],
        log_scales: vec![0.0; 6],
        sh_coeffs: vec![0.0; 2 * 4 * 3],
        raw_opacities: vec![0.0; 2],
        origin: vec![None; 2],
        age: vec![0.0; 2],
        importance: vec![0.0; 2],
        sh_coeffs_per_channel: 4,
        trained: vec![true, false],
        moments: Some(moments(2, 4, 0.0)),
        ..SplatRows::default()
    };
    assert_eq!(good.check(), Ok(()));
    assert_eq!(SplatRows { trained: vec![], moments: None, ..good.clone() }.check(), Ok(()), "both are optional");
    let with = |m: RowMoments| SplatRows { moments: Some(m), ..good.clone() };
    let m = moments(2, 4, 0.0);
    let broken: [(&str, SplatRows); 7] = [
        ("trained", SplatRows { trained: vec![true], ..good.clone() }),
        ("transforms_m1", with(RowMoments { transforms_m1: vec![0.0; 19], ..m.clone() })),
        ("transforms_m2", with(RowMoments { transforms_m2: vec![0.0; 21], ..m.clone() })),
        ("sh_m1", with(RowMoments { sh_m1: vec![0.0; 2 * 3], ..m.clone() })),
        ("sh_m2", with(RowMoments { sh_m2: vec![0.0; 2 * 4 * 3], ..m.clone() })),
        ("opacity_m1", with(RowMoments { opacity_m1: vec![], ..m.clone() })),
        ("opacity_m2", with(RowMoments { opacity_m2: vec![0.0; 3], ..m.clone() })),
    ];
    for (column, rows) in broken {
        let err = rows.check().expect_err(column);
        assert!(err.contains(column), "{column}: {err}");
    }

    let picked = good.select(&[1]);
    assert_eq!(picked.trained, vec![false]);
    let pm = picked.moments.as_ref().unwrap();
    assert_eq!(pm.transforms_m1, (10..20).map(|v| v as f32).collect::<Vec<_>>());
    assert_eq!(pm.sh_m1, (12..24).map(|v| v as f32).collect::<Vec<_>>());
    assert_eq!((pm.sh_m2[0], pm.opacity_m1[0], pm.opacity_m2[0]), (1.0, 1.0, 1.0));
    assert_eq!(SplatRows::concat(&[good.select(&[0]), picked]), good);

    let bare = SplatRows { trained: vec![], moments: None, ..good.select(&[0]) };
    let mixed = SplatRows::concat(&[good.clone(), bare]);
    assert_eq!(mixed.check(), Ok(()));
    assert_eq!(mixed.trained, vec![true, false, true], "rows without flags count as trained");
    let mm = mixed.moments.unwrap();
    assert_eq!(&mm.transforms_m1[..20], &good.moments.as_ref().unwrap().transforms_m1[..]);
    assert!(mm.transforms_m1[20..].iter().chain(&mm.sh_m1[24..]).chain(&mm.sh_m2[2..]).all(|&v| v == 0.0));
}

/// A row parked for k steps comes back with its first moments scaled by β1^k, as k steps without a gradient would leave
/// them; second moments stay.
#[test]
fn decay_first_scales_only_the_first_moments() {
    let m = moments(2, 4, 1.0);
    let mut d = m.clone();
    d.decay_first(3);
    let f = 0.9f32.powi(3);
    for ((name, got, _), (_, was, _)) in d.columns(4).iter().zip(m.columns(4)) {
        let scale = if name.ends_with("m1") { f } else { 1.0 };
        assert!(got.iter().zip(was).all(|(g, w)| (g - w * scale).abs() <= 1e-6 * w.abs()), "{name}");
    }
    let mut z = m.clone();
    z.decay_first(0);
    assert_eq!(z, m);
}
