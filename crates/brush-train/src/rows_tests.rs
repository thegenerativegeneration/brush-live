use super::SplatRows;

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
