//! The launch geometry an emitted int8 GEMM declares in its PTX header.
//!
//! Since 2026-11-06 the kernel stages its K tiles in shared memory: it runs
//! on exactly the block its header names (any other traps), needs the
//! dynamic shared memory it names, and is correct on any grid. The header
//! line it is read from:
//!
//! ```text
//! // Launch: block (128,1,1); any other block traps. Grid: any; (64, 1, z) is one 64x64 tile per CTA.
//! // 4 cp.async stages of int8 K tiles. Dynamic shared memory required: 32768 bytes.
//! ```

#![allow(dead_code)]

/// `block` threads in x, `smem` bytes of dynamic shared memory, `tiles`
/// output tiles (the grid `(tiles, 1, 1)` gives each CTA one).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int8Launch {
    pub block: u32,
    pub smem: u32,
    pub tiles: u32,
}

fn number_after(line: &str, key: &str) -> u32 {
    let rest = line.split(key).nth(1).unwrap_or_else(|| panic!("no `{}` in `{}`", key, line));
    rest.chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .unwrap_or_else(|_| panic!("no number after `{}` in `{}`", key, line))
}

/// The geometry `ptx`'s int8 GEMM header declares. Panics if there is none:
/// a test that launched a kernel without it would be guessing.
pub fn int8_launch(ptx: &str) -> Int8Launch {
    let launch = ptx
        .lines()
        .find(|l| l.contains("Launch: block (") && l.contains("Grid: any; ("))
        .unwrap_or_else(|| panic!("no int8 GEMM launch line in the PTX:\n{}", ptx));
    let smem = ptx
        .lines()
        .find(|l| l.contains("Dynamic shared memory required: "))
        .unwrap_or_else(|| panic!("no shared-memory line in the PTX:\n{}", ptx));
    Int8Launch {
        block: number_after(launch, "Launch: block ("),
        smem: number_after(smem, "Dynamic shared memory required: "),
        tiles: number_after(launch, "Grid: any; ("),
    }
}
