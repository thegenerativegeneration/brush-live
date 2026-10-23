use super::loss_weights;

#[test]
fn every_step_keeps_todays_weights() {
    for step in 1..10 {
        assert_eq!(loss_weights(0.2, 1, step), (0.8, -0.2));
    }
}

#[test]
fn every_k_uses_ssim_on_multiples_and_keeps_l1_weight() {
    let got: Vec<_> = (1..=8).map(|s| loss_weights(0.2, 4, s)).collect();
    for (i, (l1, ssim)) in got.iter().enumerate() {
        let step = i as u32 + 1;
        assert_eq!(*l1, 0.8, "step {step}");
        assert_eq!(*ssim, if step % 4 == 0 { -0.2 } else { 0.0 }, "step {step}");
    }
}

#[test]
fn never_and_zero_weight_have_no_ssim() {
    for step in 1..10 {
        assert_eq!(loss_weights(0.2, 0, step), (0.8, 0.0));
        assert_eq!(loss_weights(0.0, 1, step), (1.0, 0.0));
        assert_eq!(loss_weights(0.0, 0, step), (1.0, 0.0));
    }
}
