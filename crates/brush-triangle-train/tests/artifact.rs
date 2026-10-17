use brush_triangle_train::{TrainConfig, artifact::*, schedule::schedule};
use burn::tensor::Device;

fn state() -> TriangleState {
    TriangleState {
        format_version: 1,
        positions: vec![[0., 0., 2.], [1., 0., 2.], [0., 1., 2.]],
        faces: vec![[0, 1, 2]],
        rgb_logits: vec![[0., 3_f32.ln(), -3_f32.ln()]; 3],
        weight_logits: vec![0.; 3],
        config: TrainConfig::default(),
        schedule: schedule(0., &Default::default()).unwrap(),
        iterations: 17,
        renderer_last_render: None,
    }
}

#[test]
fn raw_state_roundtrips_without_losing_geometry_logits_or_progress() {
    let s = state();
    let out = decode_state(&serde_json::to_vec(&s).unwrap()).unwrap();
    assert_eq!(out.positions, s.positions);
    assert_eq!(out.faces, s.faces);
    assert_eq!(out.rgb_logits, s.rgb_logits);
    assert_eq!(out.weight_logits, s.weight_logits);
    assert_eq!(out.iterations, 17);
    let mut bad = state();
    bad.faces[0][0] = 3;
    assert!(decode_state(&serde_json::to_vec(&bad).unwrap()).is_err());
    let mut bad = state();
    bad.format_version = 2;
    assert!(decode_state(&serde_json::to_vec(&bad).unwrap()).is_err());
    let mut bad = state();
    bad.weight_logits.clear();
    assert!(decode_state(&serde_json::to_vec(&bad).unwrap()).is_err());
}

#[test]
fn mesh_ply_contains_indexed_faces_and_activated_rgb_not_logit_values() {
    let data = String::from_utf8(mesh_ply(&state()).unwrap()).unwrap();
    let (header, body) = data.split_once("end_header\n").unwrap();
    assert!(header.contains("element vertex 3") && header.contains("element face 1"));
    let rows: Vec<_> = body
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>())
        .collect();
    assert_eq!(rows.len(), 4);
    assert_eq!(
        rows[0]
            .iter()
            .map(|s| s.parse::<f32>().unwrap())
            .collect::<Vec<_>>(),
        [0., 0., 2., 128., 191., 64.]
    );
    assert_eq!(
        rows[3]
            .iter()
            .map(|s| s.parse::<u32>().unwrap())
            .collect::<Vec<_>>(),
        [3, 0, 1, 2]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn restored_model_has_exact_parameter_values_but_no_optimizer_resume_claim() {
    let d = Device::from(brush_cube::test_helpers::test_device().await);
    let s = state();
    let model = restore_model(&s, &d).unwrap();
    assert_eq!(model.num_faces(), 1);
    assert_eq!(model.num_vertices(), 3);
    let p = model
        .positions
        .into_data_async()
        .await
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap();
    assert_eq!(p, s.positions.into_iter().flatten().collect::<Vec<_>>());
    let c = model
        .rgb_logits
        .into_data_async()
        .await
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap();
    assert_eq!(c, s.rgb_logits.into_iter().flatten().collect::<Vec<_>>());
}
