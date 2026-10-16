use super::*;

#[cube(launch)]
pub fn ranges(keys: &Tensor<u32>, ranges: &mut Tensor<u32>) {
    let id = ABSOLUTE_POS;
    if id >= keys.len() {
        terminate!();
    }
    let tile = keys[id] as usize;
    if id == 0 {
        ranges[tile * 2] = 0u32;
    } else {
        let previous = keys[id - 1] as usize;
        if previous != tile {
            ranges[tile * 2] = id as u32;
            ranges[previous * 2 + 1] = id as u32;
        }
    }
    if id == keys.len() - 1 {
        ranges[tile * 2 + 1] = keys.len() as u32;
    }
}
