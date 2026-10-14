//! Which frame goes out under which keyframe id.

use std::collections::HashSet;
use std::ops::RangeInclusive;

/// (keyframe id, frame index) in send order: `loops` forward passes with ids counting up (frame indices, mod n),
/// then with `return_pass` the frames back to the first in reverse without the last one, as ids
/// `n·loops + (n − 2 − frame)`. Frames in `dropped` are never sent. The walk back also leaves out the frames the
/// first forward pass sent as its `skip_every`-th, `2·skip_every`-th, ... keyframe (the forward pass's held-out
/// frames when `skip_every` is the server's `holdout_every`, which counts received keyframes; 1 skips nothing, as the
/// server treats `holdout_every` 1 as off).
pub(crate) fn send_order(
    n: usize,
    loops: usize,
    return_pass: bool,
    skip_every: Option<usize>,
    dropped: &HashSet<usize>,
) -> Vec<(u64, usize)> {
    let mut out: Vec<(u64, usize)> = (0..n * loops)
        .filter(|i| !dropped.contains(&(i % n)))
        .map(|i| (i as u64, i % n))
        .collect();
    if return_pass && n >= 2 {
        let held: HashSet<usize> = match skip_every {
            Some(k) if k > 1 => (0..n)
                .filter(|j| !dropped.contains(j))
                .enumerate()
                .filter(|(sent, _)| (sent + 1) % k == 0)
                .map(|(_, j)| j)
                .collect(),
            _ => HashSet::new(),
        };
        let base = n * loops;
        for frame in (0..n - 1).rev() {
            if dropped.contains(&frame) || held.contains(&frame) {
                continue;
            }
            out.push(((base + n - 2 - frame) as u64, frame));
        }
    }
    out
}

/// `A-B` (inclusive) or `A`, frame indices of transforms.json.
pub(crate) fn parse_frame_range(s: &str) -> Result<RangeInclusive<usize>, String> {
    let parse = |v: &str| v.trim().parse::<usize>().map_err(|e| format!("{s}: {e}"));
    let (a, b) = match s.split_once('-') {
        Some((a, b)) => (parse(a)?, parse(b)?),
        None => (parse(s)?, parse(s)?),
    };
    if a > b {
        return Err(format!("{s}: start after end"));
    }
    Ok(a..=b)
}

/// When the `k`-th keyframe goes out, from the replay's start: a fixed schedule, so the time spent preparing a
/// keyframe does not slow the rate down.
pub(crate) fn send_time(k: usize, rate: f32) -> std::time::Duration {
    std::time::Duration::from_secs_f64(k as f64 / f64::from(rate))
}
