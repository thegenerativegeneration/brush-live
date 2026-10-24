use super::*;

fn distinct_status() -> ServerHeader {
    ServerHeader::Status(StatusPayload {
        num_keyframes: 1,
        num_splats: 2,
        train_iters_per_s: 5.5,
        last_score_ms: 3,
        train_iters: 4,
        train_ms: 5,
        ingest_ms: 6,
        preview_ms: 7,
        voxel_ms: 8,
        fisher_ms: 9,
        uptime_ms: 10,
        throttle_ms: 11,
        adam_clamped: 12,
        sh_pruned: 13,
        step: StepPhases {
            batch_ms: 14,
            forward_ms: 15,
            loss_ms: 16,
            backward_ms: 17,
            optimizer_ms: 18,
            stats_ms: 19,
            refine_ms: 20,
            area_ms: 21,
            hold_ms: 22,
        },
    })
}

#[test]
fn status_frame_json_is_pinned() {
    let h = distinct_status();
    let frame = encode_frame(&h, &[]);
    let json = std::str::from_utf8(&frame[4..]).unwrap();
    let pinned = concat!(
        r#"{"type":"status","num_keyframes":1,"num_splats":2,"train_iters_per_s":5.5,"last_score_ms":3,"#,
        r#""train_iters":4,"train_ms":5,"ingest_ms":6,"preview_ms":7,"voxel_ms":8,"fisher_ms":9,"#,
        r#""uptime_ms":10,"throttle_ms":11,"adam_clamped":12,"sh_pruned":13,"batch_ms":14,"#,
        r#""forward_ms":15,"loss_ms":16,"backward_ms":17,"optimizer_ms":18,"stats_ms":19,"#,
        r#""refine_ms":20,"area_ms":21,"hold_ms":22}"#,
    );
    assert_eq!(json, pinned);
    let json_frame = {
        let mut f = (pinned.len() as u32).to_le_bytes().to_vec();
        f.extend_from_slice(pinned.as_bytes());
        f
    };
    let (back, _): (ServerHeader, _) = decode_frame(&json_frame).unwrap();
    assert_eq!(back, h);
}
