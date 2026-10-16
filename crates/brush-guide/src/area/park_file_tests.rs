use super::*;

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

/// Review Focus 5.
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
    let bytes = encode_tile(&tile(rows(3, Vec3::ZERO)));
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
    assert!(decode_tile(&patched(28, 0b10)).is_err(), "unknown flag bit");
    assert!(
        decode_tile(&patched(20, u32::MAX)).is_err(),
        "a row count beyond the file"
    );
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(decode_tile(&trailing).is_err());
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
