use super::*;
use brush_train::rows::RowMoments;

/// A deterministic pseudo-random stream in [0, 1) (xorshift), so the test needs no RNG crate API.
fn stream(mut s: u32) -> impl FnMut() -> f32 {
    move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s >> 8) as f32 / (1u32 << 24) as f32
    }
}

fn rows(n: usize, offset: Vec3) -> SplatRows {
    let mut r = stream(7);
    let mut f = |lo: f32, hi: f32| lo + (hi - lo) * r();
    SplatRows {
        means: (0..n)
            .flat_map(|_| {
                [
                    offset.x + f(-5.0, 5.0),
                    offset.y + f(-2.0, 30.0),
                    offset.z + f(-5.0, 5.0),
                ]
            })
            .collect(),
        rotations: (0..n)
            .flat_map(|_| {
                let q = Quat::from_euler(
                    glam::EulerRot::XYZ,
                    f(-3.0, 3.0),
                    f(-3.0, 3.0),
                    f(-3.0, 3.0),
                );
                [q.w, q.x, q.y, q.z]
            })
            .collect(),
        log_scales: (0..n * 3).map(|_| f(-7.0, 0.0)).collect(),
        sh_coeffs: (0..n * 12).map(|_| f(-2.0, 2.0)).collect(),
        raw_opacities: (0..n).map(|_| f(-6.0, 6.0)).collect(),
        min_scale: (0..n).map(|_| f(1e-4, 1e-2)).collect(),
        origin: (0..n)
            .map(|i| (i % 3 != 0).then_some(i as u32 % 5))
            .collect(),
        age: (0..n).map(|i| i as f32).collect(),
        importance: (0..n)
            .map(|i| {
                if i % 4 == 0 {
                    f32::INFINITY
                } else {
                    f(0.0, 1e-3)
                }
            })
            .collect(),
        sh_coeffs_per_channel: 4,
        ..SplatRows::default()
    }
}

fn tile(rows: SplatRows) -> ParkedTile {
    ParkedTile {
        key: TileKey { x: -3, z: 12 },
        version: 4,
        slot_poses: vec![SlotPose {
            slot: 2,
            position: Vec3::new(1.0, 1.5, -2.0),
            rotation: Quat::from_rotation_y(0.3),
        }],
        rows,
    }
}

fn close(a: &[f32], b: &[f32], rel: f32) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= rel * x.abs().max(1.0), "{x} vs {y}");
    }
}

/// A fresh directory for one test (tests run in parallel).
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("park-file-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn a_tile_round_trips_within_fp16() {
    let t = tile(rows(257, Vec3::ZERO));
    let back = decode_tile(&encode_tile(&t)).unwrap();
    assert_eq!(
        (back.key, back.version, &back.slot_poses),
        (t.key, t.version, &t.slot_poses)
    );
    let (a, b) = (&t.rows, &back.rows);
    assert_eq!(b.means, a.means, "means are f32");
    assert_eq!(
        (&b.origin, &b.age, &b.importance),
        (&a.origin, &a.age, &a.importance)
    );
    let normalised: Vec<f32> = a
        .rotations
        .chunks(4)
        .flat_map(|q| {
            let l = q.iter().map(|v| v * v).sum::<f32>().sqrt();
            q.iter().map(move |v| v / l).collect::<Vec<_>>()
        })
        .collect();
    close(&b.rotations, &normalised, 1e-3);
    close(&b.log_scales, &a.log_scales, 1e-3);
    close(&b.raw_opacities, &a.raw_opacities, 2e-3);
    close(&b.sh_coeffs, &a.sh_coeffs, 1e-3);
    close(&b.min_scale, &a.min_scale, 1e-3);
    assert_eq!(b.sh_coeffs_per_channel, 4);
}

/// Means hundreds of metres from the origin round-trip bit for bit.
#[test]
fn means_far_from_the_origin_round_trip_exactly() {
    let t = tile(rows(16, Vec3::new(512.3, 0.0, -733.9)));
    assert_eq!(
        decode_tile(&encode_tile(&t)).unwrap().rows.means,
        t.rows.means
    );
}

#[test]
fn a_tile_without_floor_round_trips() {
    let mut r = rows(5, Vec3::ZERO);
    r.min_scale.clear();
    assert!(
        decode_tile(&encode_tile(&tile(r)))
            .unwrap()
            .rows
            .min_scale
            .is_empty()
    );
}

#[test]
fn an_empty_tile_round_trips() {
    let t = tile(SplatRows {
        sh_coeffs_per_channel: 4,
        ..SplatRows::default()
    });
    assert_eq!(decode_tile(&encode_tile(&t)).unwrap(), t);
}

/// Unnormalised rotations are stored as unit quaternions; the render kernels normalise them anyway.
#[test]
fn rotations_are_stored_normalised() {
    let mut r = rows(1, Vec3::ZERO);
    r.rotations = vec![0.0, 0.0, 3.0, 4.0];
    let back = decode_tile(&encode_tile(&tile(r))).unwrap();
    close(&back.rows.rotations, &[0.0, 0.0, 0.6, 0.8], 1e-3);
}

#[test]
fn truncated_or_foreign_files_are_errors() {
    let bytes = encode_tile(&tile(rows(8, Vec3::ZERO)));
    assert!(decode_tile(&bytes[..bytes.len() - 1]).is_err());
    let mut foreign = bytes.clone();
    foreign[0] = b'X';
    assert!(decode_tile(&foreign).is_err());
}

/// Every prefix is an error, never a panic, and so are unknown versions, unknown flags and trailing bytes.
#[test]
fn every_prefix_and_every_unknown_header_is_an_error() {
    let mut r = rows(3, Vec3::ZERO);
    r.moments = Some(wide_moments(3, 4));
    r.trained = vec![true, false, true];
    let bytes = encode_tile(&tile(r));
    for end in 0..bytes.len() {
        let err = decode_tile(&bytes[..end]).unwrap_err();
        assert_eq!(
            err.kind(),
            io::ErrorKind::InvalidData,
            "prefix of {end} bytes"
        );
    }
    let patched = |at: usize, value: u32| {
        let mut b = bytes.clone();
        b[at..at + 4].copy_from_slice(&value.to_le_bytes());
        b
    };
    let version = decode_tile(&patched(4, TILE_FILE_VERSION + 1)).unwrap_err();
    assert!(version.to_string().contains("version"), "{version}");
    assert!(decode_tile(&patched(28, 0b1000)).is_err(), "unknown flag bit");
    assert!(
        decode_tile(&patched(20, u32::MAX)).is_err(),
        "a row count beyond the file"
    );
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(decode_tile(&trailing).is_err());
    // The trained column follows the header, one pose and 66 bytes per row of degree-1 rows with a floor.
    let trained_at = HEADER_BYTES + 32 + 3 * 66;
    assert_eq!(&bytes[trained_at..trained_at + 3], &[1, 0, 1]);
    let mut two = bytes.clone();
    two[trained_at + 1] = 2;
    let err = decode_tile(&two).unwrap_err();
    assert!(err.to_string().contains("trained flag 2"), "{err}");
}

#[test]
fn files_live_under_their_stem_and_are_removed_with_their_levels() {
    let dir = scratch("levels");
    let t = tile(rows(8, Vec3::ZERO));
    let bytes = write_tile(&dir, &t).unwrap();
    assert_eq!(tile_path(&dir, t.key), dir.join("-3_12.bin"));
    assert_eq!(
        std::fs::metadata(tile_path(&dir, t.key)).unwrap().len(),
        bytes
    );
    assert_eq!(read_tile_count(&dir, t.key).unwrap(), 8);
    assert_eq!(read_tile(&dir, t.key).unwrap().rows.means, t.rows.means);
    std::fs::write(dir.join("-3_12_L1.bin"), [0u8]).unwrap();
    // A neighbour whose stem starts with this one's stays.
    std::fs::write(dir.join("-3_123_L1.bin"), [0u8]).unwrap();
    remove_tile_files(&dir, t.key);
    assert!(!tile_path(&dir, t.key).exists() && !dir.join("-3_12_L1.bin").exists());
    assert!(dir.join("-3_123_L1.bin").exists());
    assert!(
        !dir.join("-3_12.bin.tmp").exists(),
        "no temp file is left behind"
    );
    remove_tile_files(&dir, t.key);
    std::fs::remove_dir_all(dir).ok();
}

/// A file renamed to another tile's stem is refused rather than restored at the wrong place.
#[test]
fn a_file_holding_another_tile_is_an_error() {
    let dir = scratch("other-key");
    let t = tile(rows(4, Vec3::ZERO));
    write_tile(&dir, &t).unwrap();
    let other = TileKey { x: 0, z: 0 };
    std::fs::rename(tile_path(&dir, t.key), tile_path(&dir, other)).unwrap();
    assert_eq!(
        read_tile(&dir, other).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    assert_eq!(
        read_tile_count(&dir, other).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    assert_eq!(
        read_tile_count(&dir, t.key).unwrap_err().kind(),
        io::ErrorKind::NotFound
    );
    std::fs::remove_dir_all(dir).ok();
}

/// Rows whose columns do not fit their row count are refused before anything is written.
#[test]
fn writing_inconsistent_rows_is_an_invalid_input_error() {
    let dir = scratch("invalid");
    let mut r = rows(4, Vec3::ZERO);
    r.age.pop();
    let err = write_tile(&dir, &tile(r)).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        0,
        "nothing written"
    );
    std::fs::remove_dir_all(dir).ok();
}

/// A failed rename (the tile path is taken by a directory) leaves no temporary file behind.
#[test]
fn a_failed_write_leaves_no_temporary_file() {
    let dir = scratch("failed-rename");
    let t = tile(rows(4, Vec3::ZERO));
    std::fs::create_dir_all(tile_path(&dir, t.key).join("blocker")).unwrap();
    assert!(write_tile(&dir, &t).is_err());
    assert!(!dir.join("-3_12.bin.tmp").exists());
    std::fs::remove_dir_all(dir).ok();
}

/// `n` rows of moments whose magnitudes span 1e-30 to 1e-1 with both signs on the first moments, plus exact zeros.
fn wide_moments(n: usize, coeffs: usize) -> RowMoments {
    let mut r = stream(11);
    let mut m = RowMoments::zeros(n, coeffs);
    let mut fill = |v: &mut Vec<f32>, signed: bool| {
        for (i, x) in v.iter_mut().enumerate() {
            let mag = 10f32.powf(-30.0 + 29.0 * r());
            *x = match i % 7 {
                0 => 0.0,
                1 => 1e-10,
                _ if signed && i % 2 == 0 => -mag,
                _ => mag,
            };
        }
    };
    fill(&mut m.transforms_m1, true);
    fill(&mut m.transforms_m2, false);
    fill(&mut m.sh_m1, true);
    fill(&mut m.sh_m2, false);
    fill(&mut m.opacity_m1, true);
    fill(&mut m.opacity_m2, false);
    m
}

/// The moments come back within bfloat16 (relative 2⁻⁸, the zeros and a squared gradient of 1e-10 too, which
/// fp16 would flush to zero), and the trained flags exactly.
#[test]
fn moments_and_trained_flags_round_trip() {
    let mut r = rows(37, Vec3::ZERO);
    r.moments = Some(wide_moments(37, 4));
    r.trained = (0..37).map(|i| i % 3 != 0).collect();
    let back = decode_tile(&encode_tile(&tile(r.clone()))).unwrap().rows;
    assert_eq!(back.trained, r.trained);
    let (got, want) = (back.moments.unwrap(), r.moments.unwrap());
    for ((name, g, _), (_, w, _)) in got.columns(4).iter().zip(want.columns(4)) {
        assert_eq!(g.len(), w.len(), "{name}");
        for (x, y) in g.iter().zip(w) {
            assert!((x - y).abs() <= y.abs() / 256.0, "{name}: {x} vs {y}");
            assert_eq!(*y == 0.0, *x == 0.0, "{name}: {x} vs {y}");
        }
    }
    assert!(want.sh_m2.contains(&1e-10) && got.sh_m2.iter().any(|&v| (v - 1e-10).abs() < 1e-12));
}

/// A version 1 file (the layout without moments and trained flags) still loads, without moments (the trainer
/// starts them at zero) and without flags (the rows count as trained); version 1 knows no flag but the floor.
#[test]
fn a_version_1_file_loads_without_moments() {
    let t = tile(rows(9, Vec3::ZERO));
    let mut v1 = encode_tile(&t);
    v1[4..8].copy_from_slice(&1u32.to_le_bytes());
    let back = decode_tile(&v1).unwrap();
    assert_eq!(back, decode_tile(&encode_tile(&t)).unwrap());
    assert!(back.rows.moments.is_none() && back.rows.trained.is_empty());
    let mut flagged = v1.clone();
    flagged[28..32].copy_from_slice(&0b11u32.to_le_bytes());
    assert!(decode_tile(&flagged).is_err(), "version 1 has no moments flag");
    let mut with = t.clone();
    with.rows.moments = Some(RowMoments::zeros(9, 4));
    assert_eq!(&encode_tile(&with)[4..8], &TILE_FILE_VERSION.to_le_bytes(), "files are written at version 2");
    assert_eq!(TILE_FILE_VERSION, 2);
}
