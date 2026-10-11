//! A struct-field attribute is lowered where it is accepted, or refused.
//!
//! Measured before this file existed:
//!
//! * **`@align(N)` promised an alignment nothing provided.** The LLVM backend
//!   put `align N` on every load and store of the field, while the struct type
//!   kept the field at its natural offset (`%S = type { i32, i32, i32 }` put
//!   an `@align(64)` second field at offset 4) and the struct's storage was an
//!   `alloca` at its natural alignment. A false `align` is undefined behaviour
//!   in LLVM, and the padding the manual said the attribute provided (§9.4,
//!   "resides in a separate cache line") was never emitted. It is refused now.
//! * **`--emit-cpu` dropped every field attribute.** A struct passed by
//!   reference was written as `pub a: i32` for `@atomic a: I32`, with a clean
//!   compile. The LLVM backend is the only one that lowers them, so the CPU
//!   backend refuses them by name.
use std::path::Path;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

fn compile(dir: &Path, tag: &str, src: &str, args: &[&str]) -> (bool, String) {
    let path = dir.join(format!("{tag}.ysu"));
    std::fs::write(&path, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y")).arg(&path).args(args).current_dir(dir).output().expect("run Y");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

/// A struct the CPU backend can emit - it is only passed by reference - with
/// `attrs` in front of its first field.
fn by_reference(attrs: &str) -> String {
    format!(
        "struct Pair {{\n    {attrs}a: I32,\n    b: I32,\n}}\n\n\
         fn sum(p: &Pair) -> I32 {{\n    return p.a + p.b;\n}}\n\n\
         fn bump(p: &mut Pair) {{\n    p.a = p.a + 1;\n}}\n"
    )
}

#[test]
fn align_is_refused_on_every_backend() {
    let src = "struct S {\n    x: I32,\n    @align(64) b: I32,\n}\n\nfn main() -> I32 {\n    let s = S { x: 1, b: 2 };\n    return s.b;\n}\n";
    for (tag, args) in [
        ("default", vec!["-o", "s.bin"]),
        ("emit_llvm", vec!["--emit-llvm"]),
        ("emit_cpu", vec!["--emit-cpu"]),
        ("native", vec!["--emit-native", "-o", "s.elf"]),
    ] {
        let dir = pinned::pinned_scratch(&format!("field_attr_align_{tag}"), pinned::SM_PINNED);
        let (ok, out) = compile(&dir, "s", src, &args);
        assert!(!ok, "{tag}: `@align` compiled, but nothing lays the field out at it:\n{out}");
        assert!(out.contains("`@align` is not implemented"), "{tag}: refused without naming `@align`:\n{out}");
        assert!(!dir.join("s.ll").exists() && !dir.join("s.bin").exists(), "{tag}: refused, but wrote an artifact");
    }
}

#[test]
fn emit_cpu_refuses_a_field_attribute_it_would_drop() {
    for (attr, name) in [("@atomic ", "@atomic"), ("@atomic(acq_rel) ", "@atomic"), ("@gpu_uncached ", "@gpu_uncached")] {
        let dir = pinned::pinned_scratch("field_attr_cpu", pinned::SM_PINNED);
        let (ok, out) = compile(&dir, "pair", &by_reference(attr), &["--emit-cpu"]);
        assert!(!ok, "`{attr}` was accepted by --emit-cpu, which emits an ordinary field:\n{out}");
        assert!(
            out.contains(&format!("`{name}` on field `Pair.a`")),
            "`{attr}` refused without naming the attribute and the field:\n{out}"
        );
        assert!(!out.contains("GENERATED RUST BLOB"), "`{attr}`: refused, but the blob was printed:\n{out}");
    }
    // The control: the same struct without an attribute is emitted, so the
    // refusal above is about the attribute and not about the struct.
    let dir = pinned::pinned_scratch("field_attr_cpu_ctl", pinned::SM_PINNED);
    let (ok, out) = compile(&dir, "pair", &by_reference(""), &["--emit-cpu"]);
    assert!(ok && out.contains("pub a: i32,"), "the attribute-free struct must still be emitted:\n{out}");
}

/// The LLVM backend still lowers the attributes it was always lowering, and
/// no access claims more alignment than the type's own.
#[test]
fn the_llvm_backend_still_lowers_atomic_and_uncached_fields() {
    let dir = pinned::pinned_scratch("field_attr_llvm", pinned::SM_PINNED);
    let src = "struct Pair {\n    @atomic a: I32,\n    @gpu_uncached b: I32,\n}\n\n\
               fn sum(p: &Pair) -> I32 {\n    return p.a + p.b;\n}\n\n\
               fn bump(p: &mut Pair) {\n    p.a = p.a + 1;\n    p.b = 2;\n}\n";
    let (ok, out) = compile(&dir, "pair", src, &["--emit-llvm"]);
    assert!(ok, "{out}");
    let ir = std::fs::read_to_string(dir.join("pair.ll")).expect("read the module");
    assert!(ir.contains("load atomic i32") && ir.contains("store atomic i32"), "`@atomic` is no longer atomic:\n{ir}");
    assert!(ir.contains("load volatile i32") && ir.contains("store volatile i32"), "`@gpu_uncached` is no longer volatile:\n{ir}");
    assert!(!ir.contains(", align 64"), "an access claims an alignment nothing provides:\n{ir}");
}

/// The two fixtures that used `@align` already padded by hand - 64 bytes
/// between the fields, which no 64-byte line can span - and still compile
/// without it.
#[test]
fn the_hand_padded_fixtures_still_compile() {
    for fixture in ["tests/ring_buffer.ysu", "tests/test_align.ysu"] {
        let dir = pinned::pinned_scratch("field_attr_fixture", pinned::SM_PINNED);
        let src = pinned::copy_fixture(&dir, fixture);
        let out = Command::new(env!("CARGO_BIN_EXE_Y")).arg(&src).arg("--emit-llvm").current_dir(&dir).output().unwrap();
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        assert!(out.status.success(), "{fixture}:\n{text}");
    }
}
