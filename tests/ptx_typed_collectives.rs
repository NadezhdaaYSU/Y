//! Warp shuffles and warp reductions in the value's own type.
//!
//! Until 2026-10-26 `shfl_sync_bfly`, `warp_reduce_sum` and `warp_reduce_max`
//! wrote an f32 register whatever they were given. An `I32` came back as the
//! float its bits spell and was then converted to an integer:
//! `shfl_sync_bfly(t + 1000, 1)` compiled, assembled, ran and stored 0 on every
//! lane, and `warp_reduce_sum` of `I32`s added their bits as floats. A 64-bit
//! value reached a `.b32` shuffle, which ptxas rejects. No assembly gate sees
//! the 32-bit case, so the centre of this file runs every type through all
//! three on the GPU, on values chosen so that a float or a 32-bit reading of
//! them is wrong, against a CPU emulation of the same butterfly.

use std::path::PathBuf;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;
#[path = "common/ptxas.rs"]
mod ptxas;

fn bin() -> PathBuf {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("Y")
}

/// Compiles `src` through the real binary against a pinned profile. Returns
/// (succeeded, output, PTX text if written).
fn compile(src: &str, name: &str) -> (bool, String, Option<String>) {
    let dir = pinned::pinned_scratch(name, pinned::SM_PINNED);
    let path = dir.join(format!("{name}.ysu"));
    std::fs::write(&path, src).unwrap();
    let ptx = path.with_extension("ptx");
    let out = Command::new(bin()).arg(&path).arg("--emit-ptx").current_dir(&dir).output().expect("run Y");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text, std::fs::read_to_string(&ptx).ok())
}

fn assemble(name: &str, ptx: &str) -> Result<(), String> {
    let assembler = ptxas::ptxas().expect("checked by the caller");
    let dir = pinned::scratch(&format!("{name}_ptxas"));
    let f = dir.join(format!("{name}.ptx"));
    std::fs::write(&f, ptx).unwrap();
    let res = Command::new(assembler).arg("-arch=sm_89").arg(&f).arg("-o").arg("/dev/null").output().expect("run ptxas");
    if res.status.success() { Ok(()) } else { Err(String::from_utf8_lossy(&res.stderr).to_string()) }
}

/// The instruction lines of `ptx`: no comments, no directives (`.reg .f32
/// %f<1>;` is declared in every kernel, used or not).
fn instructions(ptx: &str) -> Vec<&str> {
    ptx.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with("//") && !l.starts_with('.')).collect()
}

/// The five combining instructions of the reduction `label` (`SUM`, `MAX`):
/// the instructions after its comment that are not part of a shuffle.
fn reduction_ops<'a>(ptx: &'a str, label: &str) -> Vec<&'a str> {
    let at = ptx.find(&format!("REDUCTION {label} ")).unwrap_or_else(|| panic!("no {label} reduction: {ptx}"));
    let start = at + ptx[at..].find('\n').expect("the comment's line ends") + 1;
    instructions(&ptx[start..]).into_iter().filter(|l| !l.starts_with("shfl.") && !l.starts_with("mov.b64 ")).take(5).collect()
}

/// (Y type, PTX arithmetic suffix, 64 bits).
const TYPES: [(&str, &str, bool); 6] = [
    ("F32", "f32", false),
    ("F64", "f64", true),
    ("I32", "s32", false),
    ("U32", "u32", false),
    ("I64", "s64", true),
    ("U64", "u64", true),
];

/// Each lane loads one value and stores its xor-1 and xor-5 neighbours' and
/// the warp's sum and maximum.
fn kernel(ty: &str) -> String {
    format!(
        "kernel coll(In: GlobalMemory<{ty}>, S1: GlobalMemory<{ty}>, S5: GlobalMemory<{ty}>, \
         Sum: GlobalMemory<{ty}>, Mx: GlobalMemory<{ty}>) {{
    let t: I32 = global_thread_id();
    let x: {ty} = In[t];
    S1[t] = shfl_sync_bfly(x, 1);
    S5[t] = shfl_sync_bfly(x, 5);
    Sum[t] = warp_reduce_sum(x);
    Mx[t] = warp_reduce_max(x);
}}

fn main() {{
}}
"
    )
}

#[test]
fn collectives_use_the_values_own_type() {
    for (ty, arith, wide) in TYPES {
        let (ok, out, ptx) = compile(&kernel(ty), &format!("coll_{}", ty.to_lowercase()));
        assert!(ok, "{ty}: {out}");
        let ptx = ptx.expect("no PTX written");
        let ins = instructions(&ptx);
        for (label, op) in [("SUM", "add"), ("MAX", "max")] {
            let ops = reduction_ops(&ptx, label);
            assert!(ops.len() == 5 && ops.iter().all(|l| l.starts_with(&format!("{op}.{arith} "))), "{ty} {label}: {ops:?}");
        }
        if ty != "F32" {
            // No detour through a float register: no f32 instruction at all.
            assert!(!ins.iter().any(|l| l.contains(".f32")), "{ty} went through f32: {ptx}");
        }
        // Two shuffles (S1, S5) and five per reduction: one `shfl` each for a
        // 32-bit value, two (its halves) for a 64-bit one.
        let shuffles = ins.iter().filter(|l| l.starts_with("shfl.sync.bfly.b32 ")).count();
        assert_eq!(shuffles, (if wide { 2 } else { 1 }) * (2 + 5 + 5), "{ty}: {ptx}");
        assert_eq!(ins.iter().filter(|l| l.starts_with("mov.b64 {")).count(), if wide { 12 } else { 0 }, "{ty}: {ptx}");
        if ptxas::ptxas().is_some() {
            assemble(&format!("coll_{}", ty.to_lowercase()), &ptx).unwrap_or_else(|e| panic!("{ty}: ptxas: {e}"));
        } else {
            eprintln!("SKIP: no ptxas - {ty}'s collectives were not assembled.");
        }
    }
}

#[test]
fn a_shuffle_refuses_what_it_cannot_move_and_takes_any_integer_offset() {
    for ty in ["F64", "I64", "U64"] {
        let src = format!(
            "kernel b(In: GlobalMemory<{ty}>, Out: GlobalMemory<{ty}>) {{\n    let t: I32 = global_thread_id();\n    \
             let x: {ty} = In[t];\n    Out[t] = shfl_sync_bfly_b32(x, 1);\n}}\n\nfn main() {{\n}}\n"
        );
        let (ok, out, _) = compile(&src, &format!("b32_{}", ty.to_lowercase()));
        assert!(!ok && out.contains("shfl_sync_bfly_b32") && out.contains("moves 32 bits"), "{ty}: {out}");
    }
    // A lane offset is a lane index: a float one is refused, not read as bits
    // (1.0 would have been lane offset 0) or rounded.
    for name in ["shfl_sync_bfly", "shfl_sync_bfly_b32"] {
        let src = format!(
            "kernel f(In: GlobalMemory<I32>, Out: GlobalMemory<I32>) {{\n    let t: I32 = global_thread_id();\n    \
             let k: F32 = 1.0;\n    Out[t] = {name}(In[t], k);\n}}\n\nfn main() {{\n}}\n"
        );
        let (ok, out, _) = compile(&src, &format!("foff_{name}"));
        assert!(!ok && out.contains("lane offset") && out.contains("F32"), "{name}: {out}");
    }
    // Any integer offset is taken as the u32 lane index `shfl` reads: a 64-bit
    // one too, which as it stands would be a 64-bit register in a .b32 operand.
    let (ok, out, ptx) = compile(
        "kernel o(In: GlobalMemory<I32>, Out: GlobalMemory<I32>) {\n    let t: I32 = global_thread_id();\n    \
         let k: I64 = 5;\n    Out[t] = shfl_sync_bfly(In[t], k);\n}\n\nfn main() {\n}\n",
        "off_i64",
    );
    assert!(ok, "{out}");
    if ptxas::ptxas().is_some() {
        assemble("off_i64", &ptx.unwrap()).unwrap_or_else(|e| panic!("an I64 lane offset: ptxas: {e}"));
    } else {
        eprintln!("SKIP: no ptxas - the I64 lane offset was not assembled.");
    }
    // The control: a 32-bit value through the same spelling compiles.
    let (ok, out, _) = compile(
        "kernel b(In: GlobalMemory<I32>, Out: GlobalMemory<I32>) {\n    let t: I32 = global_thread_id();\n    \
         let x: I32 = In[t];\n    Out[t] = shfl_sync_bfly_b32(x, 1);\n}\n\nfn main() {\n}\n",
        "b32_i32",
    );
    assert!(ok, "{out}");
}

/// The values one test type takes, as raw little-endian bytes, 64 lanes (two
/// warps): negative and positive, above 2^24, the top bit set, both halves of
/// the 64-bit types in use.
fn lanes(ty: &str) -> Vec<u64> {
    (0..64u64)
        .map(|t| match ty {
            "F32" => ((t as f32) * 1.25 - 17.5 + (t as f32) * 1e-3).to_bits() as u64,
            "F64" => (0.1 * (t as f64 + 1.0) - 3.3 + 1e-12 * (t as f64)).to_bits(),
            "I32" => (t.wrapping_mul(0x9E37_79B1) as u32) as u64,
            "U32" => (0xF000_0000u32.wrapping_add((t as u32).wrapping_mul(0x0101_0101))) as u64,
            "I64" => ((t as i64 - 30).wrapping_mul(0x1_0000_0003) ^ (t as i64) << 40) as u64,
            "U64" => 0xFFFF_0000_0000_0000u64.wrapping_add(t.wrapping_mul(0x1_0000_0001)),
            _ => unreachable!(),
        })
        .collect()
}

/// `op(a, b)` on the bits of one type, as the kernel computes it.
fn op(ty: &str, max: bool, a: u64, b: u64) -> u64 {
    match ty {
        "F32" => {
            let (x, y) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
            (if max { x.max(y) } else { x + y }).to_bits() as u64
        }
        "F64" => {
            let (x, y) = (f64::from_bits(a), f64::from_bits(b));
            (if max { x.max(y) } else { x + y }).to_bits()
        }
        "I32" => {
            let (x, y) = (a as u32 as i32, b as u32 as i32);
            (if max { x.max(y) } else { x.wrapping_add(y) }) as u32 as u64
        }
        "U32" => {
            let (x, y) = (a as u32, b as u32);
            (if max { x.max(y) } else { x.wrapping_add(y) }) as u64
        }
        "I64" => {
            let (x, y) = (a as i64, b as i64);
            (if max { x.max(y) } else { x.wrapping_add(y) }) as u64
        }
        "U64" => if max { a.max(b) } else { a.wrapping_add(b) },
        _ => unreachable!(),
    }
}

/// The butterfly the kernel runs, lane by lane: at each offset every lane
/// combines its own value with its xor partner's.
fn butterfly(ty: &str, max: bool, v: &[u64]) -> Vec<u64> {
    let mut cur = v.to_vec();
    for off in [16usize, 8, 4, 2, 1] {
        cur = (0..cur.len()).map(|l| op(ty, max, cur[l], cur[l ^ off])).collect();
    }
    cur
}

#[test]
fn every_type_shuffles_and_reduces_exactly_on_the_gpu() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - the typed collectives were not run.");
        return;
    };
    for (ty, _, wide) in TYPES {
        let (ok, out, ptx) = compile(&kernel(ty), &format!("coll_gpu_{}", ty.to_lowercase()));
        assert!(ok, "{ty}: {out}");
        let module = ctx.load_ptx(&ptx.unwrap(), "coll").unwrap_or_else(|e| panic!("{ty}: {e}"));
        let w = if wide { 8 } else { 4 };
        let v = lanes(ty);
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()[..w].to_vec()).collect();
        let d_in = ctx.alloc(bytes.len()).unwrap();
        ctx.memcpy_htod_at(&d_in, 0, &bytes).unwrap();
        let outs: Vec<_> = (0..4).map(|_| ctx.alloc(bytes.len()).unwrap()).collect();
        for o in &outs {
            ctx.memset_u8(o, 0xA5).unwrap();
        }
        let mut args = vec![d_in.device_ptr()];
        args.extend(outs.iter().map(|o| o.device_ptr()));
        ctx.launch(&module, (2, 1, 1), (32, 1, 1), 0, &args).expect("launch");
        ctx.synchronize().expect("kernel did not complete");
        let got: Vec<Vec<u64>> = outs
            .iter()
            .map(|o| {
                let mut raw = vec![0u8; bytes.len()];
                ctx.memcpy_dtoh_at(&mut raw, o, 0).unwrap();
                raw.chunks(w).map(|c| {
                    let mut b = [0u8; 8];
                    b[..w].copy_from_slice(c);
                    u64::from_le_bytes(b)
                }).collect()
            })
            .collect();
        let s1: Vec<u64> = (0..64).map(|l| v[l ^ 1]).collect();
        let s5: Vec<u64> = (0..64).map(|l| v[l ^ 5]).collect();
        let (sum, mx): (Vec<u64>, Vec<u64>) = (
            v.chunks(32).flat_map(|c| butterfly(ty, false, c)).collect(),
            v.chunks(32).flat_map(|c| butterfly(ty, true, c)).collect(),
        );
        for (name, g, want) in [("xor-1", &got[0], &s1), ("xor-5", &got[1], &s5), ("sum", &got[2], &sum), ("max", &got[3], &mx)] {
            assert_eq!(g, want, "{ty} {name}: bits differ from the CPU butterfly");
        }
    }
}
