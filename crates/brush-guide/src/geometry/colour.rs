//! sRGB transfer function (IEC 61966-2-1). The splat is trained on
//! sRGB-encoded images, so its rendered colours are sRGB-encoded; they are
//! averaged in linear RGB and encoded again for display.

/// Linear value of the sRGB-encoded channel value `c` in `[0, 1]`.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB-encoded value of the linear channel value `l` in `[0, 1]`.
pub fn linear_to_srgb(l: f32) -> f32 {
    if l <= 0.003_130_8 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// 8-bit sRGB of a linear RGB colour, clamped to `[0, 1]` first.
pub fn linear_to_srgb8(rgb: [f32; 3]) -> [u8; 3] {
    rgb.map(|l| (linear_to_srgb(l.clamp(0.0, 1.0)) * 255.0).round() as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_8_bit_value_survives_the_round_trip() {
        for v in 0..=255u8 {
            let linear = srgb_to_linear(v as f32 / 255.0);
            assert_eq!(linear_to_srgb8([linear; 3]), [v; 3], "value {v}");
        }
    }

    #[test]
    fn known_values() {
        assert!((srgb_to_linear(0.5) - 0.214_041).abs() < 1e-5);
        assert!((linear_to_srgb(0.5) - 0.735_357).abs() < 1e-5);
        assert_eq!(linear_to_srgb8([0.0, 1.0, 2.0]), [0, 255, 255]);
        assert_eq!(linear_to_srgb8([-1.0, f32::MIN_POSITIVE, 0.0]), [0, 0, 0]);
    }
}
