//! Which frame goes out under which keyframe id.

/// (keyframe id, frame index) in send order: `loops` forward passes with ids counting up, then with
/// `return_pass` the frames back to the first in reverse without the last one, as ids `n·loops + (n − 2 − frame)`,
/// leaving out frames `j` with `(j + 1) % skip_every == 0` (the forward pass's held-out frames when `skip_every`
/// is the server's `holdout_every`).
pub(crate) fn send_order(
    n: usize,
    loops: usize,
    return_pass: bool,
    skip_every: Option<usize>,
) -> Vec<(u64, usize)> {
    let mut out: Vec<(u64, usize)> = (0..n * loops).map(|i| (i as u64, i % n)).collect();
    if return_pass && n >= 2 {
        let base = n * loops;
        for frame in (0..n - 1).rev() {
            if skip_every.is_some_and(|k| k > 0 && (frame + 1) % k == 0) {
                continue;
            }
            out.push(((base + n - 2 - frame) as u64, frame));
        }
    }
    out
}

/// When the `k`-th keyframe goes out, from the replay's start: a fixed schedule, so the time spent preparing a
/// keyframe does not slow the rate down.
pub(crate) fn send_time(k: usize, rate: f32) -> std::time::Duration {
    std::time::Duration::from_secs_f64(k as f64 / f64::from(rate))
}
