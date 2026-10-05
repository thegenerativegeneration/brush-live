//! Smoke + invariant tests for the loss kernels.
//!
//! GT lives as `[H, W]` u32 packing `[r g b a]` u8. We feed deterministic u8
//! data through `image_loss` and check structural properties (`SSIM(x, x) ≈ 1`,
//! output range, backward produces finite gradients). Bit-exact reference
//! matching is covered by the integration training tests in `brush-bench-test`.

use brush_loss::{
    ImageLossConfig, TILE_SIZE, image_loss, image_loss_eval, image_loss_partials, psnr,
    psnr_from_mse,
};
use burn::tensor::{Device, Int, Tensor, TensorData};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

fn pack_rgba(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|p| {
            u32::from(p[0]) | u32::from(p[1]) << 8 | u32::from(p[2]) << 16 | u32::from(p[3]) << 24
        })
        .collect()
}

/// Deterministic u8 pattern (avoids RNG so the test is reproducible across
/// machines). Returns `H*W*4` RGBA bytes.
fn make_pattern(h: usize, w: usize, scale: u32, offset: u32) -> Vec<u8> {
    (0..h * w * 4)
        .map(|i| ((i as u32 * scale + offset) % 251) as u8)
        .collect()
}

fn pred_from_bytes(bytes: &[u8], h: usize, w: usize, device: &Device) -> Tensor<3> {
    let rgb: Vec<f32> = bytes
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]].map(|b| b as f32 / 255.0))
        .collect();
    Tensor::<1>::from_floats(rgb.as_slice(), device).reshape([h, w, 3])
}

fn gt_packed_from_bytes(bytes: &[u8], h: usize, w: usize, device: &Device) -> Tensor<2, Int> {
    // Bit-reinterpret the u32 packing as i32 so the dispatch int_from_data
    // path doesn't reject magnitudes > i32::MAX.
    let packed: Vec<i32> = pack_rgba(bytes).into_iter().map(|x| x as i32).collect();
    Tensor::from_data(TensorData::new(packed, [h, w]), device)
}

/// Read a tensor back as a flat `f32` vec.
async fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("readback")
        .try_to_vec()
        .expect("vec")
}

fn ssim_only_cfg() -> ImageLossConfig {
    ImageLossConfig {
        l1_weight: 0.0,
        ssim_weight: 1.0,
        composite_bg: None,
        mask: false,
        alpha_weight: 0.0,
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ssim_identical_inputs_is_one() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (40, 56);
    let bytes = make_pattern(h, w, 11, 13);
    let pred = pred_from_bytes(&bytes, h, w, &device);
    let gt = gt_packed_from_bytes(&bytes, h, w, &device);

    let map = to_vec(image_loss_eval(pred, gt, ssim_only_cfg())).await;
    let mean: f32 = map.iter().sum::<f32>() / (h * w * 3) as f32;
    // Identical inputs SSIM saturates at 1; allow a sub-ULP roundoff.
    assert!(
        (mean - 1.0).abs() < 1e-4,
        "SSIM(x, x) should be 1, got {mean}"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ssim_in_clamp_range() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (40, 56);
    let bytes_a = make_pattern(h, w, 7, 19);
    let bytes_b = make_pattern(h, w, 13, 7);
    let pred = pred_from_bytes(&bytes_a, h, w, &device);
    let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);

    let data = to_vec(image_loss_eval(pred, gt, ssim_only_cfg())).await;
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (-1.0..=1.0).contains(&min) && (-1.0..=1.0).contains(&max),
        "SSIM out of [-1, 1]: min={min} max={max}"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn image_loss_backward_runs() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (32, 48);
    let bytes_a = make_pattern(h, w, 5, 1);
    let bytes_b = make_pattern(h, w, 7, 11);
    let pred = pred_from_bytes(&bytes_a, h, w, &device).require_grad();
    let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);

    let loss = image_loss(
        pred.clone(),
        gt,
        ImageLossConfig {
            l1_weight: 0.8,
            ssim_weight: -0.2,
            composite_bg: None,
            mask: false,
            alpha_weight: 0.0,
        },
    );
    let grads = loss.backward();
    let data = to_vec(pred.grad(&grads).expect("pred should have a gradient")).await;
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(
        max_abs > 0.0,
        "backward should produce non-zero gradients, got all zeros"
    );
    assert!(
        data.iter().all(|v| v.is_finite()),
        "gradients should be finite"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn alpha_match_via_4ch_pred() {
    // Feeding 4-channel `pred` makes the kernel emit `|pred.a - gt.a|`
    // into the alpha channel of the loss map.
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (16, 24);
    let bytes = make_pattern(h, w, 17, 5);
    let rgba: Vec<f32> = bytes.iter().map(|b| *b as f32 / 255.0).collect();
    let pred = Tensor::<1>::from_floats(rgba.as_slice(), &device)
        .reshape([h, w, 4])
        .require_grad();
    let gt = gt_packed_from_bytes(&bytes, h, w, &device);

    let cfg = ImageLossConfig {
        l1_weight: 1.0,
        ssim_weight: 0.0,
        composite_bg: None,
        mask: false,
        alpha_weight: 1.0,
    };
    let map = image_loss_eval(pred.clone(), gt.clone(), cfg);
    assert_eq!(map.dims(), [h, w, 4]);
    let partials = image_loss_partials(pred, gt, cfg);
    assert_eq!(partials.dims()[0], 4);
    let _grads = partials.sum().backward();
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn psnr_matches_known_values() {
    let device = Device::from(brush_cube::test_helpers::test_device().await);
    let scalar = async |t: Tensor<1>| t.into_scalar_async::<f32>().await.expect("readback");

    // Plain formula: MSE 0.01 -> 20 dB, MSE 1 -> 0 dB.
    let db = scalar(psnr_from_mse(Tensor::from_floats([0.01], &device))).await;
    assert!((db - 20.0).abs() < 1e-3, "got {db}");
    let db = scalar(psnr_from_mse(Tensor::from_floats([1.0], &device))).await;
    assert!(db.abs() < 1e-3, "got {db}");

    // Identical images floor at 100 dB instead of going infinite.
    let db = scalar(psnr_from_mse(Tensor::from_floats([0.0], &device))).await;
    assert!((db - 100.0).abs() < 1e-3, "got {db}");

    // Image wrapper: a constant 0.1 offset on every channel is MSE 0.01.
    let a = Tensor::<3>::zeros([4, 6, 3], &device);
    let b = Tensor::<3>::full([4, 6, 3], 0.1, &device);
    let db = scalar(psnr(a.clone(), b)).await;
    assert!((db - 20.0).abs() < 1e-3, "got {db}");
    let db = scalar(psnr(a.clone(), a)).await;
    assert!((db - 100.0).abs() < 1e-3, "got {db}");
}

/// The weighted per-tile partial sums must agree with the per-pixel map's
/// channel means.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn partials_match_map_sums() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (37, 53); // not a multiple of the 16x16 tile
    let bytes = make_pattern(h, w, 17, 5);
    let bytes_b = make_pattern(h, w, 3, 9);
    let rgba: Vec<f32> = bytes.iter().map(|b| *b as f32 / 255.0).collect();
    let pred = Tensor::<1>::from_floats(rgba.as_slice(), &device).reshape([h, w, 4]);
    let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);
    let cfg = ImageLossConfig {
        l1_weight: 0.7,
        ssim_weight: 0.3,
        composite_bg: None,
        mask: false,
        alpha_weight: 1.0,
    };
    let map = image_loss_eval(pred.clone(), gt.clone(), cfg);
    let map_sums = to_vec(map.sum_dims(&[0, 1]).reshape([4])).await;
    let partial_sums = to_vec(image_loss_partials(pred, gt, cfg).sum_dim(1).reshape([4])).await;
    let pixels = (h * w) as f32;
    for c in 0..4 {
        let weight = if c < 3 {
            1.0 / (3.0 * pixels)
        } else {
            cfg.alpha_weight / pixels
        };
        let (a, b) = (map_sums[c] * weight, partial_sums[c]);
        assert!(
            (a - b).abs() <= 1e-3 * a.abs().max(1e-3),
            "channel {c}: weighted map sum {a} vs partial sum {b}"
        );
    }
}

/// Finite-difference check of the loss gradient through the tile partials.
/// Each probe weights only its own channel, and gives every tile a slightly
/// different weight so the backward's per-tile upstream gradient path has
/// something non-uniform to pick up. Keeping it to one channel also keeps the
/// summed loss small enough for f32 finite differences to resolve (weighting
/// all four quantises them to ~0.03).
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn partials_gradient_matches_finite_difference() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (24, 40);
    let bytes = make_pattern(h, w, 5, 1);
    let bytes_b = make_pattern(h, w, 7, 11);
    let mut rgba: Vec<f32> = bytes.iter().map(|b| *b as f32 / 255.0).collect();
    // Keep pred away from exact equality with gt so the L1 sign is defined.
    for (i, v) in rgba.iter_mut().enumerate() {
        *v = (*v + 0.013 * ((i % 7) as f32 + 1.0)).min(0.97);
    }
    let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);
    let cfg = ImageLossConfig {
        l1_weight: 0.6,
        ssim_weight: 0.4,
        composite_bg: None,
        mask: false,
        alpha_weight: 1.0,
    };
    let tiles = w.div_ceil(TILE_SIZE) * h.div_ceil(TILE_SIZE);

    // A few pixels: interior, tile edges, and the alpha channel.
    let probes = [(5, 7, 0), (15, 16, 1), (16, 33, 2), (23, 39, 0), (9, 20, 3)];
    let eps = 2e-3_f32;
    let mut failed = Vec::new();
    for (y, x, c) in probes {
        let mut wts = vec![0.0f32; 4 * tiles];
        for (tile, wt) in wts[c * tiles..(c + 1) * tiles].iter_mut().enumerate() {
            *wt = 1.0 + tile as f32 / tiles as f32;
        }
        let weights = Tensor::<1>::from_floats(wts.as_slice(), &device).reshape([4, tiles]);
        let loss_of = |data: &[f32]| {
            let pred = Tensor::<1>::from_floats(data, &device).reshape([h, w, 4]);
            (image_loss_partials(pred, gt.clone(), cfg) * weights.clone()).sum()
        };

        let pred = Tensor::<1>::from_floats(rgba.as_slice(), &device)
            .reshape([h, w, 4])
            .require_grad();
        let loss = (image_loss_partials(pred.clone(), gt.clone(), cfg) * weights.clone()).sum();
        let grads = loss.backward();
        let grad = to_vec(pred.grad(&grads).expect("grad")).await;

        let i = (y * w + x) * 4 + c;
        let mut plus = rgba.clone();
        plus[i] += eps;
        let mut minus = rgba.clone();
        minus[i] -= eps;
        let lp = loss_of(&plus).into_scalar_async::<f32>().await.expect("rb");
        let lm = loss_of(&minus)
            .into_scalar_async::<f32>()
            .await
            .expect("rb");
        let numerical = (lp - lm) / (2.0 * eps);
        let analytical = grad[i];
        let tol = 5e-3 + 0.02 * numerical.abs().max(analytical.abs());
        if (numerical - analytical).abs() > tol {
            failed.push(format!(
                "pixel ({y},{x}) channel {c}: numerical {numerical:.4} vs analytical {analytical:.4}"
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "loss gradient mismatches:\n  {}",
        failed.join("\n  ")
    );
}

fn l1_only_cfg(mask: bool) -> ImageLossConfig {
    ImageLossConfig {
        l1_weight: 1.0,
        ssim_weight: 0.0,
        composite_bg: None,
        mask,
        alpha_weight: 0.0,
    }
}

/// GT channel decode as the kernel does it (multiply by the f32 reciprocal,
/// which differs from `/ 255.0` by one ulp for some bytes).
fn decode_gt(byte: u8) -> f32 {
    byte as f32 * (1.0 / 255.0)
}

/// CPU reference of the per-pixel L1 map (`|pred - gt|`, times `gt.a` with
/// mask) for `[H, W, 3]` pred built from `bytes_a` and gt from `bytes_b`.
fn l1_reference(bytes_a: &[u8], bytes_b: &[u8], mask: bool) -> Vec<f32> {
    bytes_a
        .chunks_exact(4)
        .zip(bytes_b.chunks_exact(4))
        .flat_map(|(p, g)| {
            let a = if mask { decode_gt(g[3]) } else { 1.0 };
            (0..3).map(move |c| (p[c] as f32 / 255.0 - decode_gt(g[c])).abs() * a)
        })
        .collect()
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn l1_only_map_matches_cpu_reference() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    // Not a multiple of the 16 px forward tile or the 8 px backward tile.
    let (h, w) = (37, 53);
    let bytes_a = make_pattern(h, w, 7, 19);
    let bytes_b = make_pattern(h, w, 13, 7);
    for mask in [false, true] {
        let pred = pred_from_bytes(&bytes_a, h, w, &device);
        let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);
        let map = to_vec(image_loss_eval(pred, gt, l1_only_cfg(mask))).await;
        let want = l1_reference(&bytes_a, &bytes_b, mask);
        let worst = map
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            worst < 1e-6,
            "mask {mask}: L1 map differs from reference by {worst}"
        );
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn l1_only_reduced_loss_and_gradient_match_reference() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (37, 53);
    let bytes_a = make_pattern(h, w, 5, 1);
    let bytes_b = make_pattern(h, w, 7, 11);
    let l1_w = 0.8_f32;
    let cfg = ImageLossConfig {
        l1_weight: l1_w,
        ssim_weight: 0.0,
        composite_bg: None,
        mask: false,
        alpha_weight: 0.0,
    };
    let pred = pred_from_bytes(&bytes_a, h, w, &device).require_grad();
    let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);
    let loss = image_loss(pred.clone(), gt, cfg);
    let value = to_vec(loss.clone()).await[0];
    let grads = loss.backward();
    let grad = to_vec(pred.grad(&grads).expect("grad")).await;

    let n = (h * w * 3) as f32;
    let reference = l1_reference(&bytes_a, &bytes_b, false);
    let want = l1_w * reference.iter().sum::<f32>() / n;
    assert!(
        (value - want).abs() < 1e-5,
        "loss {value} vs reference {want}"
    );

    for ((i, g), (p, t)) in grad.iter().enumerate().zip(
        bytes_a
            .chunks_exact(4)
            .zip(bytes_b.chunks_exact(4))
            .flat_map(|(p, g)| (0..3).map(move |c| (p[c], g[c]))),
    ) {
        let diff = p as f32 / 255.0 - decode_gt(t);
        let sign = diff.signum() * f32::from(diff != 0.0);
        let want = l1_w * sign / n;
        assert!(
            (g - want).abs() <= 1e-6 * want.abs().max(1e-6),
            "grad[{i}] = {g}, want {want}"
        );
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn l1_only_alpha_match_gradient_matches_finite_difference() {
    // Same probe scheme as `partials_gradient_matches_finite_difference`,
    // with SSIM off: covers the L1-only path's reduced forward and backward,
    // tile edges and the alpha channel.
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (24, 40);
    let bytes = make_pattern(h, w, 5, 1);
    let bytes_b = make_pattern(h, w, 7, 11);
    let mut rgba: Vec<f32> = bytes.iter().map(|b| *b as f32 / 255.0).collect();
    for (i, v) in rgba.iter_mut().enumerate() {
        *v = (*v + 0.013 * ((i % 7) as f32 + 1.0)).min(0.97);
    }
    let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);
    let cfg = ImageLossConfig {
        l1_weight: 0.6,
        ssim_weight: 0.0,
        composite_bg: None,
        mask: false,
        alpha_weight: 1.0,
    };
    let tiles = w.div_ceil(TILE_SIZE) * h.div_ceil(TILE_SIZE);
    let probes = [(5, 7, 0), (15, 16, 1), (16, 33, 2), (23, 39, 0), (9, 20, 3)];
    let eps = 2e-3_f32;
    let mut failed = Vec::new();
    for (y, x, c) in probes {
        let mut wts = vec![0.0f32; 4 * tiles];
        for (tile, wt) in wts[c * tiles..(c + 1) * tiles].iter_mut().enumerate() {
            *wt = 1.0 + tile as f32 / tiles as f32;
        }
        let weights = Tensor::<1>::from_floats(wts.as_slice(), &device).reshape([4, tiles]);
        let loss_of = |data: &[f32]| {
            let pred = Tensor::<1>::from_floats(data, &device).reshape([h, w, 4]);
            (image_loss_partials(pred, gt.clone(), cfg) * weights.clone()).sum()
        };
        let pred = Tensor::<1>::from_floats(rgba.as_slice(), &device)
            .reshape([h, w, 4])
            .require_grad();
        let loss = (image_loss_partials(pred.clone(), gt.clone(), cfg) * weights.clone()).sum();
        let grads = loss.backward();
        let grad = to_vec(pred.grad(&grads).expect("grad")).await;
        let i = (y * w + x) * 4 + c;
        let mut plus = rgba.clone();
        plus[i] += eps;
        let mut minus = rgba.clone();
        minus[i] -= eps;
        let lp = loss_of(&plus).into_scalar_async::<f32>().await.expect("rb");
        let lm = loss_of(&minus)
            .into_scalar_async::<f32>()
            .await
            .expect("rb");
        let numerical = (lp - lm) / (2.0 * eps);
        let analytical = grad[i];
        let tol = 5e-3 + 0.02 * numerical.abs().max(analytical.abs());
        if (numerical - analytical).abs() > tol {
            failed.push(format!(
                "pixel ({y},{x}) channel {c}: numerical {numerical:.4} vs analytical {analytical:.4}"
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "loss gradient mismatches:\n  {}",
        failed.join("\n  ")
    );
}
