//! The Coq proofs are part of the product, so they have to be part of the suite.
//!
//! `proofs/ZkControlFlow.v` was for a long time the one machine-checked
//! artifact in this repo, and it was run by a command in CLAUDE.md and by
//! **nothing else** - no test, no build script, no CI. That is the shape this
//! repo already catalogues under other names: `@ZeroDrift` before it was
//! implemented, the `scheme = "plonkish"` flag, the `wgmma...s4` that existed
//! on no hardware. An unrun proof is the paperwork of a proof.
//!
//! It matters more here than for an ordinary test because a proof is about a
//! MODEL of a lowering rather than about the Rust. The model can drift from the
//! code silently; the least this gate can do is guarantee it still type-checks
//! and still rests on nothing.
//!
//! **This gate sweeps `proofs/` rather than naming one file.** The first
//! version named `ZkControlFlow.v`, which is exactly how the second proof would
//! have gone unrun - the same mistake one level up. Adding a `.v` file now puts
//! it under all three checks automatically; only its content control has to be
//! written by hand, and `every_proof_has_a_content_control` refuses to let that
//! be skipped.
//!
//! Three claims per file, and the third is the one that rots quietly:
//!
//! 1. `coqc` accepts it.
//! 2. Every `Print Assumptions` reports `Closed under the global context` -
//!    counted against the number the SOURCE asks for, so deleting a
//!    `Print Assumptions` line cannot quietly reduce what is checked. An
//!    `Axioms:` section appearing would mean a proof now depends on an
//!    assumption nobody stated.
//! 3. Nothing is `Admitted`. `coqc` accepts an admitted lemma happily and
//!    prints a warning that no exit code reflects.
//!
//! Local Cargo runs print a skip when `coqc` is absent. With
//! `Y_VERIFICATION_STRICT=1`, missing Rocq fails this gate.

use std::path::PathBuf;
use std::process::Command;

#[path = "common/verification.rs"]
mod verification;

#[path = "common/coq.rs"]
mod coq;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn proofs_dir() -> PathBuf {
    repo().join("proofs")
}

/// Every `.v` in `proofs/`, sorted so a run is reproducible.
fn proof_sources() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(proofs_dir())
        .expect("read proofs/")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "v").unwrap_or(false))
        .collect();
    v.sort();
    v
}

fn have_coqc() -> bool {
    Command::new("coqc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Compile every proof in a private directory.
///
/// Not in `proofs/` on purpose: `coqc` writes `.vo`/`.glob`/`.vok`/`.vos`
/// beside the source, so compiling in place would race a developer running
/// `coqc` by hand and would litter a tracked directory. The `.ptx` race in the
/// GPU harness was exactly this, and it presented as an intermittent failure.
///
/// All sources are copied in together and the loop retries, so one proof may
/// `Require` another without this gate having to know the order.
fn compile_all_proofs() -> Vec<(String, bool, String)> {
    let dir = std::env::temp_dir().join(format!("y_coq_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    let sources = proof_sources();
    for src in &sources {
        let dst = dir.join(src.file_name().unwrap());
        std::fs::copy(src, &dst).expect("copy a proof");
    }

    let mut pending: Vec<String> = sources
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    let mut done: Vec<(String, bool, String)> = Vec::new();

    // At most one pass per file is enough for any acyclic Require graph.
    for _ in 0..pending.len().max(1) {
        let mut still: Vec<String> = Vec::new();
        let mut progressed = false;
        for name in pending.drain(..) {
            let out = Command::new("coqc")
                .args(["-Q", ".", ""])
                .arg(&name)
                .current_dir(&dir)
                .output()
                .expect("run coqc");
            let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
            text.push_str(&String::from_utf8_lossy(&out.stderr));
            if out.status.success() {
                progressed = true;
                done.push((name, true, text));
            } else {
                still.push(name.clone());
                done.push((name, false, text));
            }
        }
        if still.is_empty() || !progressed {
            break;
        }
        // Retry the failures; drop their earlier (failed) rows first.
        done.retain(|(n, ok, _)| *ok || !still.contains(n));
        pending = still;
    }

    let _ = std::fs::remove_dir_all(&dir);
    done
}

#[test]
fn every_coq_proof_still_checks_and_rests_on_no_axioms() {
    if !verification::prerequisite_available(have_coqc(), "coqc on PATH to check every Coq proof") {
        return;
    }
    let results = compile_all_proofs();
    assert!(
        !results.is_empty(),
        "no `.v` files found in proofs/ - this gate would pass perfectly while \
         checking nothing"
    );

    for (name, ok, output) in &results {
        // `Print Assumptions` prints either `Closed under the global context` or
        // an `Axioms:` section listing what the proof leans on. Count what the
        // SOURCE asks for rather than hardcoding a number here, so deleting a
        // `Print Assumptions` line is caught by the content control below and
        // cannot silently shrink what this check covers.
        let src = std::fs::read_to_string(proofs_dir().join(name)).expect("read a proof");
        let asked = coq_tokens(&src)
            .windows(2)
            .filter(|tokens| tokens[0] == "Print" && tokens[1] == "Assumptions")
            .count();
        coq::check_coq_output(*ok, output, asked)
            .unwrap_or_else(|error| panic!("proofs/{name}: {error}"));
    }
}

/// Blank out everything inside `(* ... *)`, keeping every newline so line
/// numbers survive.
///
/// **Stripping comments LINE-LOCALLY is not enough, and that was found the way
/// these things are found: by writing a docstring.** The first version split
/// each line on `(*` and scanned the prefix, so the *opening* line of a comment
/// was handled and every continuation line was scanned as if it were code - a
/// paragraph containing the ordinary English word "admit" failed the gate.
/// Contorting the prose would have left the hole; Coq comments also NEST, so
/// the depth counter is not optional.
fn strip_coq_comments(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut depth = 0usize;
    let mut in_string = false;
    let mut i = 0usize;
    while i < b.len() {
        if depth == 0 && b[i] == '"' {
            out.push(b[i]);
            i += 1;
            // Rocq escapes a quote inside a string by doubling it.
            if in_string && i < b.len() && b[i] == '"' {
                out.push(b[i]);
                i += 1;
            } else {
                in_string = !in_string;
            }
        } else if !in_string && b[i] == '(' && i + 1 < b.len() && b[i + 1] == '*' {
            depth += 1;
            out.push(' ');
            out.push(' ');
            i += 2;
        } else if depth > 0 && b[i] == '*' && i + 1 < b.len() && b[i + 1] == ')' {
            depth -= 1;
            out.push(' ');
            out.push(' ');
            i += 2;
        } else {
            // Newlines are kept even inside a comment, so `lines()` still
            // agrees with the file.
            out.push(if depth > 0 && b[i] != '\n' { ' ' } else { b[i] });
            i += 1;
        }
    }
    out
}

/// Active identifiers and punctuation, excluding nested comments and strings.
/// Neither a quoted diagnostic nor documentation can supply a proof command.
fn coq_tokens(src: &str) -> Vec<String> {
    let stripped = strip_coq_comments(src);
    let mut chars = stripped.chars().peekable();
    let mut tokens = Vec::new();
    while let Some(c) = chars.next() {
        if c == '"' {
            while let Some(c) = chars.next() {
                if c == '"' {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                    } else {
                        break;
                    }
                }
            }
        } else if c.is_alphanumeric() || c == '_' || c == '\'' {
            let mut token = String::from(c);
            while let Some(&next) = chars.peek() {
                if !(next.is_alphanumeric() || next == '_' || next == '\'') {
                    break;
                }
                token.push(chars.next().unwrap());
            }
            tokens.push(token);
        } else if !c.is_whitespace() {
            tokens.push(String::from(c));
        }
    }
    tokens
}

/// `coqc` accepts an admitted lemma and exits 0. Only the source says.
#[test]
fn nothing_in_any_proof_is_admitted() {
    let mut offenders: Vec<String> = Vec::new();
    for path in proof_sources() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let src = std::fs::read_to_string(&path).expect("read a proof");
        let stripped = strip_coq_comments(&src);
        for (i, (code, line)) in stripped.lines().zip(src.lines()).enumerate() {
            // Match the TOKEN, not a line shape. The previous version tested
            // `code == "Admitted."`, `starts_with("admit")` and
            // `contains(" admit.")`, which between them miss the commonest way
            // of stubbing a Coq lemma at all: `Proof. Admitted.` on ONE line is
            // none of the three. Found by probing the gate with the thing it
            // exists to catch, which is the only way a hole like this surfaces -
            // it passes every real file perfectly.
            //
            // `Abort.` is here for the same reason one level up: it discards the
            // lemma outright, so there is no theorem left to `Print Assumptions`
            // and the file still compiles.
            let holed = code
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .any(|t| t == "Admitted" || t == "admit" || t == "Abort" || t == "give_up");
            if holed {
                offenders.push(format!("{name}:{}: {}", i + 1, line.trim()));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "an admitted proof is the paperwork of a proof without the proof:\n  {}",
        offenders.join("\n  ")
    );
}

/// The content controls. Both tests above pass perfectly against a file
/// containing no theorems at all - "no axioms" and "nothing admitted" are
/// properties an EMPTY file has. Same shape as
/// `the_artifact_sweep_actually_finds_artifacts` and
/// `ordinary_loop_bodies_still_verify`.
///
/// Each entry names the load-bearing theorem AND its refutation counterpart:
/// proving the new thing right without proving the old thing wrong is what let
/// the flat-vs-nested distinction survive Z3 and a fresh test file.
fn content_controls() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        (
            // `exact_pv`: the first kernel in the repository with BOTH a Rocq
            // proof of what its PTX computes and a per-translation validation
            // of that PTX against its SASS. The capstone is the value claim;
            // the two REFUTATIONS beside it are what stop the file being
            // satisfied by licences nothing can violate, and both are the
            // device's own numbers reproduced from the wrap functions alone -
            // `the_measured_index_wrap_reads_v0_twice` is the ceiling nobody
            // had written down (the source states only the accumulator one,
            // which at head_dim 64 is 8x looser), and
            // `the_accumulator_licence_is_load_bearing` is the ceiling it does
            // state, at its own boundary. `the_loop_lengths_exist` is what
            // stops that second one reading as vacuous, since its lengths are
            // abstract - a `nat` literal of sixteen million is unary and does
            // not survive `Qed`.
            "ExactPvExact.v",
            &[
                "Print Assumptions the_emitted_exact_pv_holds_the_source_dot_product",
                "Print Assumptions the_index_ceiling_binds_first",
                "Print Assumptions the_measured_index_wrap_reads_v0_twice",
                "Print Assumptions without_the_index_licence_the_answer_is_wrong",
                "Print Assumptions the_measured_overflow_is_two_s_complement",
                "Print Assumptions the_accumulator_licence_is_load_bearing",
                "Print Assumptions the_loop_lengths_exist",
                // The tie comes first: without it the partition below is a
                // fact about an index SHAPE that nothing says this kernel has.
                "Print Assumptions the_output_index_is_the_mathematical_one",
                "Print Assumptions every_output_element_is_written_by_one_thread",
            ],
        ),
        (
            "ExactPvLaunchContract.v",
            &[
                "Print Assumptions checked_shape_licenses_every_live_index",
                "Print Assumptions checked_shape_licenses_the_output_index",
                "Print Assumptions checked_shape_instantiates_the_source_dot_product",
                "Print Assumptions geometry_covers_the_output_rectangle",
                "Print Assumptions geometry_output_coordinates_are_unique",
                "Print Assumptions disjoint_live_byte_spans_never_alias",
            ],
        ),
        (
            // The int8 GEMM's VALUE, as against `Int8GemmSchedule.v`'s
            // schedule. The four entries are the four claims, and the last two
            // are what stop the file being satisfied by something nothing
            // violates: `bounded_products_accumulate_exactly` is where the
            // licence is load-bearing (without it the capstone is false, and
            // the compiler was not checking it), and
            // `the_measured_overflow_is_two_s_complement` reproduces the value
            // the DEVICE returned past the bound from `wrap32` alone - a model
            // that merely said "it overflows" would agree with any wrong
            // answer.
            "Int8GemmExact.v",
            &[
                "Print Assumptions the_lanes_cover_the_a_fragment",
                // Since 2026-11-06 the fragments come through a swizzled
                // shared stage: the copy's swizzle and ldmatrix's lane
                // addresses must cancel, and no copy may overwrite another.
                "Print Assumptions the_ldmatrix_a_fragment_is_the_source_element",
                "Print Assumptions the_ldmatrix_b_fragment_is_the_source_element",
                "Print Assumptions the_stage_slots_are_disjoint",
                "Print Assumptions bounded_products_accumulate_exactly",
                "Print Assumptions the_measured_overflow_is_two_s_complement",
                "Print Assumptions the_emitted_int8_gemm_holds_the_source_dot_products",
                // The split-K accumulation in int32 rather than in Z, added
                // when the flat-accumulation caveat was closed. The refutation
                // beside it is the hypothesis writing that theorem forced out:
                // the licence is sufficient only for a ZEROED destination, and
                // a caller who splits K across LAUNCHES into the same int32 C
                // has every launch licensed and the accumulation not.
                "Print Assumptions the_split_k_accumulation_is_exact_in_int32",
                "Print Assumptions the_combine_needs_a_zeroed_destination",
                "Print Assumptions from_zero_the_same_partial_is_exact",
                // The same two folds for any owner map: the staged kernel
                // stripes K TILES over z, not K steps.
                "Print Assumptions the_owned_split_is_exact_in_int32",
            ],
        ),
        (
            // The GPU warp tiling. The partition theorem is the load-bearing
            // one - it is the GPU twin of `c_written_exactly_once`, which the
            // CPU chain has had since the tiling increment and the GPU had no
            // equivalent for. The REFUTATION is listed beside it because the
            // partition alone would be satisfied by a tiling nothing violates:
            // `the_measured_gap` states the instance that was actually
            // compiled, holes and all, so the file is pinned to a defect
            // rather than to a symmetry.
            // The int8 tensor-core GEMM's schedule. It is the only GPU GEMM
            // whose EXACTNESS is available to prove (the other `mma.sync`
            // are floating point). Since 2026-11-06 the kernel is the staged
            // one, and the entries are its claims, each with the refutation
            // that makes it load-bearing: the block-size trap confines a warp
            // to its own tile (a fifth warp lands in the next tile) and gives
            // every stage row a copier (a short block leaves rows uncopied);
            // every output element has one owner and every tile is computed
            // once on any grid; the atomic reduction is order-independent,
            // and a rounding accumulate would break it.
            "Int8GemmSchedule.v",
            &[
                "Print Assumptions the_trap_is_what_confines_a_warp_to_its_own_tile",
                "Print Assumptions without_the_trap_a_fifth_warp_lands_in_the_next_tile",
                "Print Assumptions the_copies_cover_the_stage_once",
                "Print Assumptions without_the_trap_a_short_block_leaves_rows_uncopied",
                "Print Assumptions c_element_has_exactly_one_owner",
                "Print Assumptions grouped_is_a_bijection",
                "Print Assumptions every_output_tile_is_computed_exactly_once",
                "Print Assumptions the_split_k_sums_to_the_whole_contraction",
                "Print Assumptions the_atomic_reduction_is_order_independent",
                "Print Assumptions a_rounding_accumulate_would_break_the_landing_order",
                // The FUSED epilogue, added when a device measurement showed
                // it stores where its sibling reduces. The refutation is the
                // load-bearing half: it is the defect that was measured (a
                // split CTA holds only part of the contraction), and the
                // second one is why the observed answer moved BETWEEN
                // launches rather than being merely wrong.
                "Print Assumptions a_storing_cta_computes_the_whole_contraction",
                "Print Assumptions a_split_cta_holds_only_part_of_the_contraction",
                "Print Assumptions two_split_ctas_would_store_different_values",
                "Print Assumptions the_reducing_epilogue_still_splits",
                // The plain form STORES when K is not split: its one class
                // is the whole contraction, the value a reduction into a
                // zeroed C would leave.
                "Print Assumptions the_unsplit_launch_stores_the_whole_contraction",
            ],
        ),
        (
            "GpuWarpTiling.v",
            &[
                "Print Assumptions cta_rows_written_exactly_once",
                "Print Assumptions the_measured_gap",
                // The tie: divisibility is exactly what the compiler tests
                // before it derives anything. Without this the theorems are
                // about an arithmetic nobody performs.
                "Print Assumptions the_emitted_count_is_the_model",
            ],
        ),
        (
            // The allocation bound. Its two halves are graded differently on
            // purpose: the IN-BOUNDS direction is what a heap corruption would
            // eventually reveal, while the EXACTNESS direction - the buffer is
            // not one element larger than the write set - is invisible to every
            // answer this kernel can produce, and an over-allocation of exactly
            // that shape is in this repository's history.
            "ExactGemmAllocation.v",
            &[
                "Print Assumptions pack_a_write_is_inside_the_allocation",
                "Print Assumptions the_a_allocation_is_exactly_the_write_set",
                // The join. Without it the file proves that some numbers fit
                // inside other numbers; with it, the buffer the driver
                // allocates is the buffer the packing proof describes.
                "the_stride_is_the_packing_proofs_panel_extent",
                // The refutations, so the bounds are known tight rather than
                // merely sufficient.
                "one_element_short_overflows_the_a_panel",
            ][..],
        ),
        (
            // The threading layer. The two halves of this file's claim are
            // deliberately both named: the LAYOUT bound - a worker's record
            // cannot overlap another's - and the JOIN, which says the emitted
            // reduction loop is the fold `ExactGemmKsplit` proves exact rather
            // than a fold that resembles it.
            "ExactGemmThreading.v",
            &[
                "Print Assumptions job_slot_injective",
                "Print Assumptions the_reduction_is_the_ksplit_sum",
                // The tightness half, invisible to every answer.
                "the_private_c_allocation_is_exactly_the_reduced_set",
                // The refutation that names the historical heap overflow, and
                // the one that makes the separate zeroing loop load-bearing.
                "reading_the_private_buffer_at_the_callers_stride_leaves_it",
                "a_destination_that_is_not_zeroed_is_wrong",
                // The join loop's predicate, and the refutation of the
                // `tid == 0` sentinel it replaced - which was an assumption
                // about the C library rather than about this schedule.
                "the_flag_joins_exactly_the_threads_that_started",
                "a_live_thread_with_id_zero_is_not_joined",
            ][..],
        ),
        (
            "ZkControlFlow.v",
            &[
                // The shipped lowering agrees with the operational semantics.
                "low_correct",
                "Print Assumptions low_correct",
                // The machine-checked refutation of the lowering it replaced.
                "low_tail_is_wrong_when_nested",
            ][..],
        ),
        (
            // GENERATED from `src/cpu_gemm.rs` by
            // `tests/exact_gemm_schedule_proof.rs`, which also gates it on
            // byte-identity. It is the one file here whose CONTENT is not
            // written by hand, and that is exactly why it takes no exemption
            // from this control.
            //
            // A definitions-only file would fail `every_proof_has_a_content_control`,
            // and that gate is right: "compiles" and "no axioms" are properties
            // an EMPTY file has, and a generator emitting nonsense emits it
            // confidently. So the generated file carries theorems, and two of
            // the three are STRUCTURAL - they constrain the shape of the
            // emitted expressions rather than restating their values, so they
            // are not made true merely by being generated alongside what they
            // describe.
            "ExactGemmSchedule.v",
            &[
                // The emitted `vpdpwssd` vector-group form of the B slot map
                // IS the plain interleave. The two sides come from different
                // places in `cpu_gemm.rs` - `pack_b_slot` and the bare `/2`
                // that `panel_slot_decode` inverts it with - so this is a
                // real check on the generator, not a restatement of a value.
                // The Rust asserts their agreement in a doc comment; this
                // proves it.
                "Print Assumptions slot_b_is_the_plain_interleave",
                // The constants are internally consistent. Catches a
                // generator that emitted `NR` and `NRV` from constants that
                // had stopped agreeing, or any degenerate zero.
                "the_tile_geometry_is_consistent",
                // The index arithmetic rendered from the SAME expressions
                // the emitter renders to LLVM (`cpu_gemm::tile_width_ix`,
                // `panel_index_ix`). Everything else in that file is
                // generated from a constant; these are generated from the
                // emitted CODE's arithmetic, and these two theorems are the
                // join to the tiling model's tile-index view.
                "Print Assumptions the_emitted_width_is_the_tiling_model_at_the_loop_variable",
                "the_emitted_panel_index_is_the_tile_index",
                // ...and the same for the two sites emitted as raw LLVM: the
                // micro-kernel's flush clamp, and the threaded wrapper's
                // K-split bands. Every theorem in ExactGemmKsplit.v is about
                // `blen`; the second of these is what says the emitted spawn
                // loop computes it.
                "the_emitted_tile_count_is_the_tiling_model",
                "the_emitted_chunk_end_is_the_flush_model",
                "the_emitted_band_length_is_the_ksplit_model",
                "the_emitted_a_index_is_the_pair_element",
                // The realizability constraint that was prose in a test
                // comment and stated nowhere: MR*NRV accumulators + NRV B
                // vectors + 1 A broadcast must fit 32 zmm. Measured by
                // sweeping VNNI_MR and reading real spill traffic - the cliff
                // is exactly where this flips. It makes the generator unable
                // to emit a schedule that does not fit the register file.
                "the_tile_fits_the_register_file",
                // A genuine cross-file join: every theorem in
                // `ExactGemmKsplit.v` is stated under `0 < nthr`, and nothing
                // proved the emitted thread count satisfies it - the floor
                // was argued in a comment.
                "ksplit_threads_is_never_zero",
                // The non-vacuity control, and it is honestly the weakest of
                // the four: under generation it is self-fulfilling. Its job is
                // the other direction - it makes the shipped values
                // load-bearing inside Coq, so a hand-edit of the committed
                // file fails `coqc` as well as failing byte-identity. That
                // matters most for `MR`, which the measurement recorded in
                // `ExactGemmComposition.v`'s header found was pinned by no
                // theorem in its own file.
                "the_schedule_is_the_shipped_one",
            ][..],
        ),
        (
            "ExactGemmMSplit.v",
            &[
                // The row bands PARTITION [0, M): every row in exactly one band.
                "owner_unique",
                "Print Assumptions owner_unique",
                "owner_exists",
                // ...and the schedule is exact for an ARBITRARY accumulate,
                // which is the claim that separates this axis from the K one.
                "msplit_needs_no_algebra",
                "Print Assumptions msplit_needs_no_algebra",
                // The refutation that stops that being vacuous: the same
                // rounding accumulate that breaks a K-split leaves this alone.
                "an_msplit_survives_the_accumulate_that_breaks_a_ksplit",
                "the_rounding_accumulate_is_not_associative",
            ][..],
        ),
        (
            "ExactGemmKsplit.v",
            &[
                // The K-split reduction equals the naive nest, every K, every nthr.
                "ksplit_exact",
                "Print Assumptions ksplit_exact",
                // The bands cover [0, K) exactly - the obligation a dropped
                // remainder violates.
                "bands_tile",
                // ...and that mutation refuted concretely.
                "dropping_the_remainder_loses_terms",
                // The roadmap's central claim: the SAME schedule is wrong under
                // a rounding accumulate and right under an exact one.
                "rounding_breaks_the_split",
                "exact_survives_the_same_split",
            ][..],
        ),
        (
            "MixedRadix.v",
            &[
                // Positional indices, proved once. `quot_rem_unique` had TWO
                // independent copies - ExactGemmTiling and ExactGemmPacking,
                // neither requiring the other.
                "quot_rem_unique",
                "pack_unpack",
                "unpack_pack",
                // The leg that was rewritten in three files, and the one the
                // "it is only six lines" argument for a local copy did not
                // cover.
                "pack_onto",
                "pack_in_range",
                // The two-digit peel `pack_b_slot_bijective` was doing by hand.
                "two_digit_unique",
                // ...and that the bound on the low digit is load-bearing, on
                // the high one absent, rather than either being carried along.
                "the_low_digit_bound_cannot_be_dropped",
                "without_the_digit_bound_it_is_not_injective",
                "the_high_digit_needs_no_bound",
            ][..],
        ),
        (
            "Decomposition.v",
            &[
                // The partition obligation, proved once and instantiated by
                // all three kernels. Two theorems, not one: contiguous parts
                // are a re-bracketing and cost only associativity; an
                // arbitrary owner map reorders the terms and costs
                // commutativity as well.
                "contiguous_exact",
                "decomposition_exact",
                "parts_tile",
                "acc_parts_prefix",
                // ...and that the two really are different claims, or
                // "use the general one everywhere" would be free.
                "the_interleaved_split_is_not_contiguous",
                // The refutation every kernel used to carry its own copy of:
                // the theorems are about the ACCUMULATE, not the indices.
                // The clamped edge family, which two obligations turned out to
                // share: the int32 flush interval and the output tiling.
                "clamped_width",
                "widths_cover_the_extent",
                "rounding_breaks_a_contiguous_split",
                "rounding_breaks_an_interleaved_split",
                "exact_survives_the_same_split",
                "exact_survives_the_interleaved_split",
                "the_two_accumulates_differ_on_this_input",
            ][..],
        ),
        (
            "CountingSort.v",
            &[
                // The obligation the schema did not have: a decomposition
                // whose consequence is PLACEMENT, not a folded value.
                // `Decomposition.widths_cover_the_extent` says the widths add
                // up, which is strictly weaker - a decomposition writing one
                // slot twice and another never has exactly the right total.
                "slot_injective",
                "slot_onto",
                "every_slot_is_written_exactly_once",
                // ...and that it IS the same schema and not a rival to it:
                // the width-derived edges satisfy `Decomposition`'s three
                // hypotheses, and its fold theorem instantiates on them.
                "the_bucket_edges_are_a_decomposition_edge_function",
                "the_reduction_theorem_still_applies",
                // The claim a code comment in `scatter` had been making with
                // nothing checking it: the cursors ALONE - each group stopped
                // where the next one started, the last at `off[b+1]` - imply
                // the runs tile the bucket. It assumes nothing about the
                // histogram, which is what makes the runtime check worth its
                // 0.2-1.0 ms.
                "the_chained_runs_tile_the_bucket",
                "the_chained_runs_do_not_overlap",
                "a_chained_run_stays_in_the_bucket",
                // ...and both refutations that make the chaining hypothesis
                // load-bearing, stated as refutations of the WEAKENED theorem
                // rather than as witnesses, so no fixture drift can make them
                // vacuous.
                "without_the_chaining_a_slot_can_go_unwritten",
                "without_the_chaining_two_groups_can_write_one_slot",
                // The empty-group case, which is what `scatter`'s own
                // post-condition got wrong the first time it was written.
                "an_empty_group_still_chains",
                // Two levels - buckets outside, scatter groups inside. No
                // instantiation had composed two before, and the whole cost
                // of the second one is the single hypothesis below, which is
                // `scatter`'s own grouping assertion.
                "dest_injective",
                "dest_onto",
                "every_idx_slot_is_written_exactly_once",
                "the_group_widths_exhaust_every_bucket",
                // `MixedRadix` does not cover it, and that is a property of
                // the decomposition rather than of how it is written: the
                // inner extent is the bucket's own width, and bucket widths
                // differ because they count data.
                "no_uniform_radix_describes_this",
                // Concrete, because every theorem above is satisfied by a
                // histogram that counted nothing and by one bucket with one
                // group.
                "the_msm_instance_is_not_vacuous",
            ][..],
        ),
        (
            "AttentionSchedule.v",
            &[
                // GENERATED from the emitter's own launch-geometry
                // expressions. The bijection is the whole obligation: every
                // thread is a worker the reduction folds, no two share an id,
                // and no id is unclaimed.
                "the_worker_is_below_the_worker_count",
                "distinct_threads_are_distinct_workers",
                "every_worker_is_some_thread",
                // The pairing is load-bearing, stated by refuting the
                // weakened claim rather than by exhibiting a collision.
                "dropping_the_z_extent_overflows_the_worker_count",
                // ... and the control that stops that refutation being read
                // as "attn_scores is wrong": at one CTA in z the two
                // schedules are the same map, so one proof covers both.
                "the_scores_schedule_is_the_accumulate_schedule_at_one_z",
            ],
        ),
        (
            "GridStrideSplit.v",
            &[
                // The third kernel, and the first that is not a GEMM: the
                // residue classes partition the sequence, so any worker count
                // gives the naive sum.
                "grid_stride_exact",
                "stride_classes_partition",
                "any_worker_count_agrees",
                // The instantiation itself: this kernel's classes ARE the
                // schema's parts at the owner map `i |-> i mod n`.
                "combine_is_decomposition",
                // The property neither GEMM proof needed. The atomics supply
                // their operands in whatever order the hardware chooses, so
                // this one needs commutativity and not just associativity.
                "atomics_may_land_in_any_order",
                // The join to the emitted kernel. Everything else in that
                // file is stated for an ABSTRACT worker count; this is the
                // one theorem that says the compiler emits one of them.
                "the_emitted_launch_geometry_visits_every_key_exactly_once",
                "every_folded_worker_is_a_real_thread",
                // Both halves are about the ACCUMULATE - and the second is a
                // failure the GEMM kernels cannot exhibit at all, since they
                // fold their bands in index order.
                "rounding_breaks_the_stride_split",
                "rounding_is_order_dependent",
                "exact_is_order_independent",
                // The accumulator ceiling, which needs 2.7e8 keys to reach and
                // so can never be demonstrated on a device.
                "the_bound_is_one_unit_wide",
            ][..],
        ),
        (
            // Phase 4's first theorem, and the first BOUND in this directory -
            // every other file states an exactness or a coverage claim.
            "SoftmaxErrorBound.v",
            &[
                // The headline: the emitted chain's output against the ideal
                // softmax, and the same expression evaluated at a long context.
                "the_attention_output_is_within_the_bound",
                "Print Assumptions the_attention_output_is_within_the_bound",
                "the_output_quotient_is_within_the_bound",
                "the_bound_at_a_long_context",
                // The three joints, each of which was covered by nothing. The
                // exp itself is NOT one of them - it is discharged
                // exhaustively in Rust, which is stronger than a proof here.
                "an_unsaturated_argument_is_within_a_half_ulp",
                "the_weight_is_within_a_relative_ulp",
                "the_max_subtraction_puts_a_floor_under_the_total_weight",
                // The domain gap: the exhaustive sweep stops at 31<<16 and the
                // emitted saturate admits 2^30. Two lines, and they existed
                // nowhere.
                "the_swept_domain_covers_the_admitted_one",
                "saturation_means_the_exact_exponent_is_astronomically_large",
                // The refutation that makes the max subtraction load-bearing
                // rather than decorative: without its floor the same bracket
                // admits an all-zero denominator.
                "a_total_weight_below_one_ulp_admits_a_zero_denominator",
                // The answer to "does the Decomposition schema extend to
                // approximate arithmetic": the reduction is still EXACT, so
                // GridStrideSplit applies verbatim and the per-element error
                // enters the fold as data.
                "the_bound_holds_at_every_launch_geometry",
                // A hypothesis nothing is shown to satisfy is the proof-shaped
                // version of a licence nothing can violate. Both directions:
                // the obvious interface is INCONSISTENT over Q, and the one
                // used here has a model.
                "exact_homogeneity_is_unsatisfiable_over_Q",
                "no_rational_squares_to_two",
                "the_interface_is_satisfiable",
                // ...and the constant the bracket turns on is not magic.
                "the_per_unit_factor_is_what_one_unit_of_log2_costs",
                // The TEMPERATURE is itself quantized, and the shape of that
                // error is what makes it a head_dim question rather than a
                // KFix one: rounding `C * 2^32` moves the exponent by at most
                // half a score delta, INDEPENDENTLY of KFix.
                "the_quantized_temperature_shifts_the_exponent_by_at_most_half_a_delta",
                "the_ideal_weights_at_two_close_exponents_bracket_each_other",
                "at_head_dim_128_the_temperature_moves_a_weight_by_under_a_two_thousandth",
                // ...and the two multipliers that are not approximations of
                // anything. Both were a bare `continue` in a Python script.
                "a_zero_multiplier_gives_every_key_the_same_weight",
                "the_two_readings_of_the_multiplier_disagree_above_two_to_the_thirty_one",
                "the_signed_reading_zeroes_a_weight_the_unsigned_one_keeps",
            ][..],
        ),
        (
            "GemmBandSplit.v",
            &[
                // The f32 kernel is the SECOND kernel, and its K-split is a
                // different decomposition. The tiling obligation composed; its
                // proof did not transfer.
                "prop_ksplit_exact",
                "prop_bands_tile",
                // The measurement the schema exists to produce: the M/N split
                // had edge theorems and no exactness theorem, because writing
                // the reduction out a third time was not worth it. Under
                // `Decomposition` it costs one application and the three edge
                // facts already proved.
                "granule_split_exact",
                // ...and that it really is a different partition, or the
                // theorem above would be the exact kernel's wearing a hat.
                "the_two_splits_are_different",
                // The two last-band clamps the emitter writes are redundant -
                // the property that was a comment in `emit_entry`.
                "pedge_last",
                "gedge_last",
                // A band boundary inside a tile would make one thread write a
                // partial tile; also only a comment before.
                "every_edge_snaps_to_a_granule_or_the_extent",
                // The exactness obligation provably does NOT transfer, at the
                // same f/K/nthr as the exact kernel's own refutation, with the
                // control that says the failure is the accumulate's.
                "rounding_breaks_the_proportional_split_too",
                "exact_survives_the_proportional_split",
            ][..],
        ),
        (
            "ExactGemmTiling.v",
            &[
                // The output tiles account for the axis exactly...
                "tiles_cover",
                // ...and, the real obligation, name each position ONCE.
                "tile_index_injective",
                "tile_index_surjective",
                "Print Assumptions c_written_exactly_once",
                // The emitter's own comment about the ragged tail, as a
                // machine-checked refutation rather than a comment.
                "unclamped_tail_writes_out_of_bounds",
                // ...and the precondition nothing in the compiler states.
                "row_stride_below_n_aliases",
                // The fold-back loop, whose BOUND carries the obligation: it
                // runs exactly the clamped tile width, so the copy into C
                // never leaves the rectangle the tile owns.
                "the_emitted_fold_back_runs_the_tile_width",
                "the_fold_back_stays_inside_the_live_rectangle",
                // ...with the trip count doing the work, one unit wide.
                "the_fold_back_trip_count_is_what_keeps_it_in_bounds",
                            // The FIRST tie between a proof and the SHAPE of an emitted
                // loop rather than the value of an emitted expression:
                // `SCH.row_panel_*` is rendered from `cpu_gemm::CountedLoop`,
                // the description the driver opens its loops from.
                "the_emitted_row_loop_enumerates_the_tiles",
                "the_emitted_row_loop_runs_once_per_tile",
                "the_emitted_column_loop_enumerates_the_tiles",
                "the_emitted_column_loop_runs_once_per_tile",
                "the_two_panel_loops_are_different",
][..],
        ),
        (
            "ExactGemmPacking.v",
            &[
                // Both packers' destination maps are bijections onto their panel.
                "pack_a_slot_bijective",
                "pack_b_slot_bijective",
                // THE theorem: the full padded product equals the live dot
                // product, which is what licenses running a ragged tile at full
                // width.
                "Print Assumptions padded_product_is_the_live_dot_product",
                // ...and the zero-fill that makes it true, refuted concretely.
                "garbage_in_the_pad_changes_the_answer",
                "the_masked_version_agrees_on_the_same_input",
                // The precise statement of what the packing proof CANNOT say
                // about lane layout: the emitted vector-group form IS the
                // plain interleave, so there is no arithmetic difference for
                // any proof to capture. Deleting this turns the file's own
                // "what this does not prove" section into an unchecked remark.
                "slot_b_is_the_plain_interleave",
                // The panel as a FUNCTION - what slot `s` actually holds. This
                // is the statement the composition needed and had to assume;
                // the uniqueness theorem is what makes it a claim about the
                // emitted loop rather than about a convenient model.
                "Print Assumptions panel_decodes_its_own_write",
                "panel_is_the_only_solution",
                // The group bound refuted concretely: without `idx < width` the
                // contract is FALSE of every non-zero operand, because the next
                // index is the following k-pair group's first slot. The control
                // shows the same panel agrees inside the group.
                "the_group_bound_is_load_bearing",
                "inside_the_group_the_same_panel_agrees",
            ][..],
        ),
        (
            "ExactGemmMicro.v",
            &[
                // The flush chunks sum to the whole k-pair range, with no
                // hypothesis that the interval divides it.
                "Print Assumptions flush_exact",
                // The int32 accumulator agrees with Z exactly when nothing
                // leaves the range...
                "flush_exact_in_int32",
                // ...and the licence's own bound is what supplies that.
                "operand_bound_gives_no_overflow",
                "the_licence_makes_the_chunk_exact",
                // The overflow refuted concretely rather than assumed, and the
                // boundary pinned at one unit wide.
                "overflow_breaks_the_flush",
                "the_4096_case_exceeds_by_exactly_one",
                // The cross-file tie: the column pack_b routes to a lane is the
                // column the store reads that lane back out to.
                "the_packed_column_is_the_stored_column",
                "a_wrong_lane_stride_permutes_the_columns",
            ][..],
        ),
        (
            "ExactGemmRegisterTile.v",
            &[
                // THE routing theorem: lane l of vector v consumes exactly the
                // two packed slots of the column the store sends it to.
                "Print Assumptions the_lane_consumes_its_own_column",
                // The i32 load of A aliases the packed pair...
                "the_i32_load_is_the_packed_pair",
                // ...and the endianness that rests on is load-bearing, with a
                // control showing a symmetric operand would hide it.
                "swapping_the_pair_halves_computes_a_different_function",
                "a_symmetric_operand_hides_the_swap",
                // The 24 accumulators of 16 lanes tile the 64 columns.
                "tile_position_injective",
                "tile_position_surjective",
                // The masked tails, and why the packers' row/column masks are
                // redundant while the phantom k-half's is not.
                "only_the_live_rectangle_is_stored",
                "a_padding_column_never_reaches_c",
            ][..],
        ),
        (
            "ExactGemmComposition.v",
            &[
                // The five files must AGREE on the definitions they each
                // declared separately - three copies of the B slot map, two of
                // MR/NR/col_of. Each copy turns out to be pinned by a theorem
                // in its own file already (that was measured, not assumed), so
                // these make an incidental pinning explicit and cross-file.
                "packing_and_register_tile_agree_on_the_b_slot",
                "micro_and_register_tile_agree_on_the_b_slot",
                "the_tile_shape_is_the_same_everywhere",
                "the_agreement_is_not_vacuous",
                // Packing says what a slot holds, the register tile says which
                // lane reads it; together, the lane reads the right source
                // elements. Neither half can state this alone.
                "Print Assumptions the_lane_accumulates_the_source_elements",
                "a_dead_row_contributes_nothing",
                // ...and the packers' contract is DISCHARGED rather than
                // assumed. Deleting this leaves the composition resting on a
                // hypothesis nothing is shown to satisfy - which is the
                // proof-shaped version of a licence nothing can violate, and
                // is the state this file was committed in once.
                "the_packed_panels_route_to_the_right_source_elements",
                            // The two obligations that turned out to be one decomposition:
                // an int32 overflow budget and a memory partition, both
                // `t |-> min(t*X, ext)`, spelled as an END in one emitter site
                // and as a WIDTH in the other.
                "the_flush_interval_and_the_output_tile_are_the_same_family",
                "the_agreement_covers_the_ragged_case",
][..],
        ),
        (
            "ExactGemmChain.v",
            &[
                // THE headline: the emitted flush schedule, accumulating each
                // chunk in int32, over the panels the emitted packers produce,
                // through the emitted routing, computes the SOURCE dot product
                // for one accumulator lane.
                "Print Assumptions the_emitted_lane_computes_the_source_dot_product",
                // The two joins that are the file's own new content. The
                // sibling models were written independently and it was not
                // obvious they would meet: sum_pairs counts k-PAIRS with both
                // halves, sum_from is a flat range.
                "kloop_is_the_padded_product",
                "kloop_is_sum_from_step",
                "the_k_pair_loop_computes_the_source_dot_product",
                "the_flush_schedule_computes_the_source_dot_product",
                // The masking survives to loop scale rather than being
                // re-argued per layer.
                "a_dead_row_accumulates_nothing",
                "a_dead_column_accumulates_nothing",
                // The licence made load-bearing for the WHOLE chain, with the
                // one-unit boundary and its control. Without these the chain
                // is a theorem about an idealised Z accumulator.
                "violating_the_licence_breaks_the_chain",
                "at_the_licensed_magnitude_the_chain_holds",
                // ...and the chain evaluated on concrete numbers, because
                // every equality above is satisfied by a model computing 0.
                "the_whole_chain_is_not_vacuous",
                // The tile lift: the same statement over the whole MR x NR
                // rectangle, with the ACCUMULATE visible (C0 + dot) and dead
                // positions left exactly as they were.
                "Print Assumptions the_tile_holds_the_source_dot_products",
                "a_dead_position_leaves_c_untouched",
                // The join it needs: RT.tile_position_surjective says an
                // inverse of col_of exists, this names it and proves it
                // inverts BOTH ways - which is what licenses "no two tile
                // positions share an accumulator lane".
                "the_lane_map_is_a_two_sided_inverse",
                "distinct_columns_use_distinct_lanes",
                "the_tile_lift_is_not_vacuous",
                // The clamp does no work: a dead position accumulates zero by
                // the packers' masks, so the tile theorem describes the
                // micro-kernel's own effect on C. The proved twin of the
                // redundancy exact_gemm_packing_model.rs measured.
                "the_store_predicate_is_redundant",
            ][..],
        ),
        (
            "ExactGemmWhole.v",
            &[
                // THE capstone: all six exact-GEMM proofs chained. Every
                // element of C, over every thread's K band, is the source dot
                // product - no hypothesis that MR divides M, NR divides N,
                // nthr divides K, or that K is even.
                "Print Assumptions the_threaded_gemm_holds_the_source_dot_products",
                "the_whole_output_holds_the_source_dot_products",
                // Every position is live in its OWN tile, which is what makes
                // the store predicate true for it.
                "the_position_is_live_in_its_own_tile",
                // ...and the (r, c) view addresses the same element the tiling
                // proof's (tile, offset) view does, so
                // ExactGemmTiling.c_written_exactly_once applies here.
                "the_position_decomposition_is_the_tilings",
                // Concrete, because every equality above is satisfied by a
                // model computing 0 - and with BOTH bands contributing, so the
                // K-split is a real reduction rather than one band doing all
                // the work.
                "the_whole_chain_is_not_vacuous",
                "both_bands_contribute",
                // The scratch tile starts at ZERO - a hypothesis `gemm_position`
                // hid inside a definition, now tied to the emitted `vg.z` loop.
                // `%Ctile` is reused across tiles, so one trip short carries the
                // previous tile's accumulator into this one.
                "the_scratch_is_zeroed_wherever_the_fold_back_reads",
                "the_zeroing_loop_covers_exactly_the_tile",
                "one_trip_short_leaves_the_corner_stale",
            ][..],
        ),
        (
            // ExactRMSNorm's sum of squares (`model.sum_of_squares`,
            // `norm_kernel._grid_sum_of_squares`): the kernel's two 40-bit
            // limbs are the grid value, which is the floor its comments
            // define; the carried limb sums are the row's exact grid sum in
            // any order and fit int64 and float64. The last entry is what
            // stops the 8192-value licence being decoration: past it the high
            // limb reaches 2^53, where its float64 conversion would round.
            "NormGridSum.v",
            &[
                "Print Assumptions the_limbs_are_the_grid_value",
                "Print Assumptions the_grid_value_is_the_floor",
                "Print Assumptions the_torch_mask_is_the_kernel_mask",
                "Print Assumptions the_carried_limbs_are_the_row_sum",
                "Print Assumptions the_sum_does_not_depend_on_the_order",
                "Print Assumptions the_limb_sums_fit",
                "Print Assumptions past_8192_values_big_can_reach_2_53",
            ][..],
        ),
        (
            // Any order and any association of a sum, in int32, int64,
            // float32 or float64, gives the integer sum while the absolute
            // values sum to at most the carrier's cap: the fact that covers
            // the kernels of the exact decode path that are not Y's. The
            // refutations are the cap being load-bearing: one unit past 2^24
            // two float32 associations of the same three terms disagree.
            "ExactReduce.v",
            &[
                "Print Assumptions any_schedule_gives_the_sum",
                "Print Assumptions no_partial_leaves_the_range",
                "Print Assumptions every_carrier_is_exact_within_its_cap",
                "Print Assumptions past_the_cap_two_schedules_disagree",
                "Print Assumptions float32_addition_rounds_past_its_cap",
                "Print Assumptions any_schedule_gives_the_dot_product",
                "Print Assumptions dot_is_the_kernel_proofs_sum",
                "Print Assumptions any_max_tree_is_a_max",
                "Print Assumptions max_key_unique",
                "Print Assumptions first_argmax_unique",
                // A schedule that scales a partial sum mid-reduction (the
                // two-digit int8 GEMM's 128 * (hi . w) + lo . w) still gives
                // the weighted sum, inside the carrier's cap.
                "Print Assumptions any_scaled_schedule_gives_the_weighted_sum",
                "Print Assumptions every_carrier_scales_exactly_within_its_cap",
            ][..],
        ),
        (
            // Exact training's arithmetic (`python/y_lang/exact/train_lm.py`):
            // the balanced base-128 digit split holds exactly the values in
            // its reach and `ndigits` is enough and the fewest for 9 to 64
            // bits; the shifted digit-pair blocks of the stacked GEMM are the
            // dot product of the values, exact in any order inside the bound
            // the code checks; the rounding shift is round-half-even, its
            // clamp and sign handling change nothing; the fixed-point sums are
            // exact in any int64 order. The refutations are each bound being
            // load-bearing: the old digit count leaves 1 over at 14 bits, round
            // half up differs on a tie, two terms of 2^62 wrap int64, and a
            // scale bound of 512 would let a product's scale leave float64.
            "ExactTrainArith.v",
            &[
                "Print Assumptions digits_rebuild",
                "Print Assumptions nothing_left_iff_in_reach",
                "Print Assumptions ndigits_hold",
                "Print Assumptions ndigits_is_the_fewest",
                "Print Assumptions ndigits_table",
                "Print Assumptions the_old_count_was_one_short",
                "Print Assumptions the_shifted_blocks_are_the_dot",
                "Print Assumptions every_order_of_the_blocks_is_exact",
                "Print Assumptions t2_widest_product_is_inside_float64",
                "Print Assumptions shift_round_is_nearest",
                "Print Assumptions nearest_even_unique",
                "Print Assumptions a_clamped_shift_rounds_the_same",
                "Print Assumptions round_half_up_differs_on_a_tie",
                "Print Assumptions fixed_point_sums_are_exact",
                "Print Assumptions past_the_int64_cap_the_sum_wraps",
                "Print Assumptions a_scale_of_512_could_leave_float64",
            ][..],
        ),
        (
            // The quantizer's storage types (`train_lm.q_dtype`, 2026-11-27):
            // a clamped value fits the type its width gets; every value the
            // digit split forms stays in int16 up to 15 bits, joined to
            // `q_dtype`'s cut; every digit fits int8. The refutations: at 16
            // bits the split leaves int16, and int8 cannot hold 9 bits.
            "ExactTrainWidths.v",
            &[
                "Print Assumptions q8_fits_its_type",
                "Print Assumptions int16_holds_every_step",
                "Print Assumptions the_split_is_safe_wherever_q_dtype_gives_int16",
                "Print Assumptions the_stacked_digits_fit_int8",
                "Print Assumptions int16_wraps_at_16_bits",
                "Print Assumptions int8_cannot_hold_9_bits",
            ][..],
        ),
        (
            // A piece of one row runs as two (`lora.SFT._accumulate`,
            // 2026-11-26): row-wise results, maxima over the rows with a copy
            // of a row, quantized values and integer sums with a zero gradient
            // row are the row alone's. The refutations: a scale that depends
            // on the row count (`exact_sum`'s) changes, and a row that is not
            // a copy can move the maximum.
            "ExactRowCopy.v",
            &[
                "Print Assumptions rowwise_keeps_the_rows",
                "Print Assumptions a_copy_keeps_the_max",
                "Print Assumptions a_copy_keeps_the_quantized",
                "Print Assumptions a_zero_row_keeps_the_sum",
                "Print Assumptions a_length_scale_does_not_survive",
                "Print Assumptions a_different_copy_moves_the_max",
            ][..],
        ),
        (
            // The MLP kernel's integer arithmetic (`mlp_kernel.py`,
            // 2026-11-27): the masked K loop is the dot product, every
            // partial sum fits int32 inside `usable`'s K bound, the sigmoid
            // table index stays in bounds and its interpolation in int32,
            // the scale's float32 field is normal exactly when `ok` holds.
            // The refutations: one more term can overflow, no lower clamp
            // makes the index negative.
            "ExactMlpKernel.v",
            &[
                "Print Assumptions the_blocked_loop_is_the_dot",
                "Print Assumptions every_partial_sum_fits_int32",
                "Print Assumptions past_133144_a_sum_can_overflow",
                "Print Assumptions the_table_index_is_in_bounds",
                "Print Assumptions without_the_lower_clamp_the_index_is_negative",
                "Print Assumptions the_interpolation_fits_int32",
                "Print Assumptions the_float32_field_is_normal_exactly_when_ok",
            ][..],
        ),
        (
            // qmm's in-register split (`qmm.py`) is ExactTrainArith's digit
            // split, its digits are int8, its int32 accumulators hold their
            // sums to MAX_K and the int64 join is exact in float64. The
            // refutation: one-digit sums past 133,144 terms can overflow.
            "ExactQmmSplit.v",
            &[
                "Print Assumptions qmm_lo_is_the_digit",
                "Print Assumptions qmm_hi_is_the_carry",
                "Print Assumptions the_join_is_the_value",
                "Print Assumptions the_12_bit_digits",
                "Print Assumptions the_accumulators_fit_int32",
                "Print Assumptions one_digit_past_133144_can_overflow",
                "Print Assumptions the_join_fails_only_past_2_35",
            ][..],
        ),
        (
            // The fused draw's two-level search (`vocab_draw.py`) is the flat
            // inverse CDF `rl.draw_rows` defines, its block and offset in
            // range. The refutations: a negative weight, and u = l.
            "ExactDrawSearch.v",
            &[
                "Print Assumptions count_spec",
                "Print Assumptions the_block_is_in_range",
                "Print Assumptions the_offset_is_in_range",
                "Print Assumptions the_two_level_search_is_the_flat_one",
                "Print Assumptions a_negative_weight_breaks_it",
                "Print Assumptions u_at_the_total_leaves_the_blocks",
            ][..],
        ),
        (
            // Zero padding of K, rows and columns and K's pieces in
            // `lora.mm_y_any` change no product; a piece fits int32, the total
            // int64. The refutation: a piece one term longer can overflow.
            "ExactMmPadding.v",
            &[
                "Print Assumptions zero_padding_k_changes_no_sum",
                "Print Assumptions a_padded_row_leaves_the_real_rows",
                "Print Assumptions the_pieces_of_k_are_the_whole_sum",
                "Print Assumptions each_piece_fits_int32",
                "Print Assumptions the_int64_total_holds",
                "Print Assumptions a_piece_one_term_longer_can_overflow",
            ][..],
        ),
        (
            // Float bits compare as the values (any mantissa width), and the
            // field's exponent bounds the value: the quantizers' maxima. The
            // refutations: the sign bit, and the all-ones field.
            "ExactFloatOrder.v",
            &[
                "Print Assumptions the_bits_are_ordered_as_the_values",
                "Print Assumptions the_max_of_the_bits_is_the_bits_of_the_max",
                "Print Assumptions the_field_bounds_the_value",
                "Print Assumptions a_normal_value_reaches_half_its_bound",
                "Print Assumptions with_the_sign_bit_the_order_breaks",
                "Print Assumptions an_infinite_field_is_not_ordered_by_the_formula",
            ][..],
        ),
        (
            // `_ceil_log2`'s eager and compiled definitions agree, both the
            // least power reaching n. The refutation: floor(log2) at 3.
            "ExactCeilLog2.v",
            &[
                "Print Assumptions the_loop_is_the_least_power",
                "Print Assumptions bit_length_is_the_least_power",
                "Print Assumptions the_two_definitions_agree",
                "Print Assumptions floor_log2_is_not_it",
            ][..],
        ),
        (
            // `rl.pieces`: the rows in order, none empty, multi-row pieces
            // within the bound, one piece with none. The refutation: a single
            // long row.
            "ExactPieces.v",
            &[
                "Print Assumptions the_pieces_are_the_rows_in_order",
                "Print Assumptions no_piece_is_empty",
                "Print Assumptions every_piece_of_two_or_more_rows_is_within_the_bound",
                "Print Assumptions no_bound_is_one_piece",
                "Print Assumptions a_single_long_row_passes_the_bound",
            ][..],
        ),
        (
            // q8's rounding reaches at most one past top, the clamped value is
            // in range with an error below one step. The refutation: without
            // the clamp 127.5 rounds to 128.
            "ExactQ8Clamp.v",
            &[
                "Print Assumptions the_rounding_reaches_at_most_one_past_top",
                "Print Assumptions the_clamped_value_is_in_range",
                "Print Assumptions the_clamped_error_is_below_one_step",
                "Print Assumptions without_the_clamp_top_is_passed",
            ][..],
        ),
        (
            // The packed KV cache's bytes (`kv.py`): the signed low byte, K's
            // int16 values and the scales' bits rebuilt exactly. The
            // refutations: an unsigned top piece, a value past int16.
            "ExactKvPack.v",
            &[
                "Print Assumptions signed_byte_is_the_low_byte",
                "Print Assumptions a_byte_is_kept",
                "Print Assumptions k_bytes_rebuild_the_value",
                "Print Assumptions v_scale_bytes_rebuild_the_bits",
                "Print Assumptions k_scale_halves_rebuild_the_bits",
                "Print Assumptions an_unsigned_top_piece_breaks_a_negative_scale",
                "Print Assumptions a_value_past_int16_loses_its_high_byte",
            ][..],
        ),
        (
            // The RMS norm kernel's bits (`norm_kernel.py`): the guarded
            // multiply's float32 is the divide's, the guard's cases, x widened
            // by its bits, the row's scale. The refutations: a guard of two
            // misses a tie, a subnormal x does not widen by its bits.
            "ExactNormKernel.v",
            &[
                "Print Assumptions the_sign_does_not_reach_the_guard_bits",
                "Print Assumptions the_rounding_is_a_normal_float32",
                "Print Assumptions the_gap_is_the_guard_distance",
                "Print Assumptions the_kernel_float32_is_the_nearest_to_every_value_within_four_ulps",
                "Print Assumptions the_multiply_is_within_three_ulps_of_the_divide",
                "Print Assumptions the_multiply_rounds_as_the_divide",
                "Print Assumptions the_guard_admits_only_zero_and_the_normal_binades",
                "Print Assumptions zero_passes_and_rounds_to_zero",
                "Print Assumptions a_normal_x_widens_by_its_bits",
                "Print Assumptions the_scale_is_a_power_of_two",
                "Print Assumptions a_guard_of_two_would_miss_a_tie",
                "Print Assumptions a_subnormal_x_does_not_widen_by_its_bits",
            ][..],
        ),
        (
            // The integer exp2 in int64 (`attention.exp2_neg_q16_16`): every
            // intermediate in 0 .. 2^63 - 1, the largest values, the model's
            // result. The refutations: 2^60 and 2^61 are reached, and the
            // product is no float64.
            "ExactExp2Int64.v",
            &[
                "Print Assumptions every_intermediate_fits_int64",
                "Print Assumptions the_reciprocal_multiply_is_division_by_six",
                "Print Assumptions the_series_divides_by_six",
                "Print Assumptions the_reciprocal_multiply_fails_past_2_31",
                "Print Assumptions the_largest_values",
                "Print Assumptions the_steps_are_the_model",
                "Print Assumptions the_product_reaches_2_60",
                "Print Assumptions the_rounding_constant_reaches_2_61",
                "Print Assumptions every_product_is_past_the_float64_exact_range",
                "Print Assumptions float64_would_round_the_product",
            ][..],
        ),
        (
            // `scaled_round`'s float32 formula is the exact round-half-even,
            // for any monotone rounding that keeps float32s. The refutations:
            // one factor alone, a rounding that is not monotone.
            "ExactScaledRound.v",
            &[
                "Print Assumptions the_two_factors_are_the_scale",
                "Print Assumptions an_integer_times_a_power_of_two_is_a_float32",
                "Print Assumptions the_float32_formula_is_the_exact_rounding",
                "Print Assumptions one_factor_alone_loses_the_value",
                "Print Assumptions a_rounding_that_is_not_monotone_breaks_it",
            ][..],
        ),
        (
            // The softmax kernel's float32 claims (`softmax_kernel.py`): the
            // 2^23 rounding is round-half-even, the row sum is 2^28 .. 2^38 so
            // no divide operand or result is subnormal, the row's q8 scale is
            // small. The refutations: no branch, a row of 2048 keys.
            "ExactSoftmaxKernel.v",
            &[
                "Print Assumptions the_magic_add_rounds_half_even",
                "Print Assumptions without_the_branch_an_odd_integer_moves",
                "Print Assumptions the_row_sum_is_between_2_28_and_2_38",
                "Print Assumptions a_row_of_2048_keys_can_pass_2_38",
                "Print Assumptions the_converted_operands_are_normal",
                "Print Assumptions the_rounded_quotient_is_zero_or_normal",
                "Print Assumptions the_softmax_row_scale_is_small",
            ][..],
        ),
        (
            // The RoPE table (`model.cos_sin_f32`): the guard's window, the
            // libm's error budget, every kept entry correctly rounded. The
            // refutation: a margin of the libm's error alone admits a tie.
            "ExactRopeTable.v",
            &[
                "Print Assumptions the_float32_is_the_nearest_to_every_fraction_within_the_margin",
                "Print Assumptions the_libm_error_moves_y_by_less_than_eleven_ulps",
                "Print Assumptions the_rope_entry_is_correctly_rounded",
                "Print Assumptions a_margin_of_the_libm_error_alone_admits_a_tie",
            ][..],
        ),
        (
            // The per-row quantizers' clamp: with a normal scale every rounded
            // quotient is within -L .. L. The refutations: a subnormal scale
            // needs the clamp, the margin is gone at L = 2^23.
            "ExactRowClamp.v",
            &[
                "Print Assumptions the_quotient_stays_below_L_and_a_half",
                "Print Assumptions the_clamp_does_nothing_when_the_scale_is_normal",
                "Print Assumptions a_subnormal_scale_needs_the_clamp",
                "Print Assumptions the_margin_is_gone_at_L_2_23",
            ][..],
        ),
        (
            // The V scale fold (`kv.fold_scale_into_weights`): p * sv is a
            // float64, and the rounded weights are within 0 .. 2^28. The
            // refutation: in float32 the bound would pass 2^28.
            "ExactVFold.v",
            &[
                "Print Assumptions an_integer_times_a_power_of_two_is_a_float",
                "Print Assumptions the_weight_product_is_exact_in_float64",
                "Print Assumptions the_scaled_weight_is_below_2_28_and_a_half",
                "Print Assumptions the_fold_stays_within_2_28",
                "Print Assumptions in_float32_the_fold_could_pass_2_28",
            ][..],
        ),
        (
            // The sampler's draw probabilities (`rl.py`): each token owns its
            // block of u, the residue count, the bias below p_t / N and below
            // 2^-16. The refutation: the modulo's bias is real.
            "ExactDrawBias.v",
            &[
                "Print Assumptions the_token_is_the_block_holding_u",
                "Print Assumptions the_count_in_a_residue_block",
                "Print Assumptions each_token_is_drawn_p_over_l_up_to_the_bias",
                "Print Assumptions the_bias_is_below_2_minus_16",
                "Print Assumptions the_modulo_has_a_bias",
            ][..],
        ),
        (
            // GRPO's advantages: the reward sum, the mean over a power-of-two
            // group and each advantage are exact. The refutation: a group of
            // three rounds the mean.
            "ExactGroupMean.v",
            &[
                "Print Assumptions the_reward_sum_is_exact",
                "Print Assumptions the_mean_is_exact",
                "Print Assumptions every_advantage_is_exact",
                "Print Assumptions a_group_of_three_rounds_the_mean",
            ][..],
        ),
        (
            // `train.py`'s helpers: the binary length, the right shift rounds
            // half up, a quantized row is within 128. The refutations: the
            // clamp acts at both ends, a right shift can leave int64.
            "ExactTrainShift.v",
            &[
                "Print Assumptions bit_length_is_the_binary_length",
                "Print Assumptions the_right_shift_rounds_half_up",
                "Print Assumptions the_quantized_row_is_within_128",
                "Print Assumptions the_clamp_acts_at_both_ends",
                "Print Assumptions a_right_shift_can_leave_int64",
            ][..],
        ),
        (
            // `det_sqrt`: one correction moves one step, two give the floor
            // square root, the scaling is exact, the result within 2^-30. The
            // refutation: one correction is not enough at two.
            "ExactDetSqrt.v",
            &[
                "Print Assumptions one_correction_moves_one_step",
                "Print Assumptions isqrt_is_the_floor_square_root",
                "Print Assumptions one_correction_is_not_enough_at_two",
                "Print Assumptions the_scaled_value_is_an_integer_in_2_60_2_62",
                "Print Assumptions det_sqrt_is_within_2_minus_30",
            ][..],
        ),
        (
            // The SiLU table's interpolation: between the knots, a float32
            // after the 2^-24 scale, q the exact rounding of x * 2^20. The
            // refutation: without the mask the offset overshoots.
            "ExactSiluTable.v",
            &[
                "Print Assumptions the_interpolation_is_between_the_knots",
                "Print Assumptions the_interpolated_sigmoid_is_a_float32",
                "Print Assumptions the_quantized_x_is_the_exact_rounding",
                "Print Assumptions without_the_mask_the_offset_overshoots",
            ][..],
        ),
        (
            // `_scaled_round_f64`: the float64 two-factor formula is the exact
            // round-half-even. The refutation: one factor alone.
            "ExactScaledRoundF64.v",
            &[
                "Print Assumptions the_two_factors_are_the_scale",
                "Print Assumptions rep64",
                "Print Assumptions the_float64_formula_is_the_exact_rounding",
                "Print Assumptions one_factor_alone_loses_the_value",
            ][..],
        ),
        (
            // The power-of-two factors from their bits, float64 and float32.
            // The refutation: past the clamps the bits are 0 or infinity.
            "ExactPow2Bits.v",
            &[
                "Print Assumptions pow2_is_two_to_the_k",
                "Print Assumptions pow2_f32_is_two_to_the_k",
                "Print Assumptions past_the_clamps_the_bits_are_not_the_power",
            ][..],
        ),
        (
            // `int8_bmm`'s K bound: every int32 schedule exact within MAX_K,
            // the largest safe K. The refutations: one more term wraps, the
            // 127 * 127 bound would admit it.
            "ExactInt8BmmBound.v",
            &[
                "Print Assumptions an_int8_product_is_at_most_2_14",
                "Print Assumptions max_k_is_the_largest_safe_k",
                "Print Assumptions every_order_gives_the_int32_sum",
                "Print Assumptions one_more_term_wraps",
                "Print Assumptions the_127_bound_would_admit_a_wrap",
            ][..],
        ),
        (
            // The fused draw's int64 sums: every order of the total and of
            // every running sum is exact. The refutation: 2^35 weights wrap.
            "ExactDrawSums.v",
            &[
                "Print Assumptions every_order_of_the_total_is_exact",
                "Print Assumptions every_running_sum_is_exact",
                "Print Assumptions past_2_35_weights_the_sum_wraps",
            ][..],
        ),
        (
            // The attention softmax kernel: the double float32 logit's error,
            // the maximum's, the guard's decision, far keys' weight 0, the
            // budget. The refutation: without the guard a near tie flips.
            "ExactAttnWeights.v",
            &[
                "Print Assumptions the_double_float_logit_is_within_5e_14",
                "Print Assumptions the_maximum_moves_by_at_most_the_error",
                "Print Assumptions the_kernel_integer_is_the_rounding",
                "Print Assumptions far_keys_have_weight_zero",
                "Print Assumptions the_budget_fits_the_guard",
                "Print Assumptions without_the_guard_a_near_tie_rounds_the_other_way",
            ][..],
        ),
        (
            // The attention kernel's pair maximum: the largest hi, then the
            // largest lo, is the largest value. The refutations: without the
            // rounding, and with the first key's lo.
            "ExactAttnPairMax.v",
            &[
                "Print Assumptions a_smaller_hi_has_a_value_no_larger",
                "Print Assumptions the_pair_maximum_is_the_maximum_value",
                "Print Assumptions without_the_rounding_the_pair_maximum_misses",
                "Print Assumptions the_low_parts_need_their_own_maximum",
            ][..],
        ),
        (
            // The attention kernel's row indices: the key scale row and the
            // mask row are the score row's, every row one query, the offsets
            // in bounds. The refutation: the other formulas read other rows.
            "ExactAttnRows.v",
            &[
                "Print Assumptions the_key_scale_row_is_the_rows_head",
                "Print Assumptions the_mask_row_is_the_rows_query",
                "Print Assumptions every_row_is_one_query",
                "Print Assumptions the_offsets_stay_in_the_tensors",
                "Print Assumptions the_int32_rows_stay_below_r",
                "Print Assumptions the_key_scale_formula_reads_the_wrong_mask_row",
            ][..],
        ),
        (
            // The attention kernel's float32 difference: within 2^-21 of the
            // pairs' difference, r in -1 .. 1. The refutation: a plain
            // subtraction loses 1.
            "ExactAttnDiff.v",
            &[
                "Print Assumptions the_lo_parts_are_at_most_1",
                "Print Assumptions the_float32_difference_is_within_2_21",
                "Print Assumptions the_budget_value_holds",
                "Print Assumptions a_plain_subtraction_would_lose_1",
            ][..],
        ),
        (
            // qmm's float32 epilogue: rounding commutes with the scale, never
            // subnormal, fits 24 bits, the precision is 24 bits. The
            // refutations: a shift of 0, and the subnormal grid.
            "ExactQmmEpilogue.v",
            &[
                "Print Assumptions rounding_commutes_with_the_scale",
                "Print Assumptions the_epilogue_is_the_float32_of_the_product",
                "Print Assumptions the_product_is_never_subnormal",
                "Print Assumptions the_rounded_value_fits_24_bits",
                "Print Assumptions a_24_bit_integer_is_kept",
                "Print Assumptions r_zero_is_not_a_rounding",
                "Print Assumptions below_2_126_the_grid_is_fixed",
            ][..],
        ),
        (
            // x - rint(x) in float32: a float32 past 2^23 is an integer, the
            // nearest integer of x >= 0 is >= 0, the remainder is a float32.
            // The refutation: a far integer leaves no float32.
            "ExactRintRemainder.v",
            &[
                "Print Assumptions a_float32_past_2_23_is_an_integer",
                "Print Assumptions the_nearest_integer_is_not_negative",
                "Print Assumptions the_remainder_is_a_float32",
                "Print Assumptions a_far_integer_leaves_no_float32",
            ][..],
        ),
        (
            // isqrt's float estimate: the error factor within 4 * 2^-50, the
            // truncated estimate within 1 of floor(sqrt n). The refutation:
            // a sqrt off by 2^-20 misses by 2047.
            "ExactSqrtEstimate.v",
            &[
                "Print Assumptions the_error_factor_is_within_4_dl",
                "Print Assumptions the_truncated_estimate_is_within_1",
                "Print Assumptions a_cruder_sqrt_misses_by_more",
            ][..],
        ),
        (
            // Generated from python/y_lang/exact by
            // tools/exact_decode_constants.py; the byte comparison is
            // python/tests/test_exact_decode_constants.py.
            "ExactDecodeConstants.v",
            &[
                "Print Assumptions the_exp_table_has_64_entries_from_one",
                "Print Assumptions the_samples_are_not_empty",
            ][..],
        ),
        (
            // The widths the exact path derives satisfy the bounds its
            // reductions need, for every shape; and these transcriptions
            // return what the Python returned on every generated sample.
            // The refutations: without the `- 128` a width that wraps is
            // admitted, and past DIGIT_MAX the high digit leaves int8.
            "ExactDecodeWidths.v",
            &[
                "Print Assumptions the_derived_activation_width_fits_the_accumulator",
                "Print Assumptions dropping_the_128_admits_a_width_that_wraps",
                "Print Assumptions the_derived_key_width_fits_float32",
                "Print Assumptions one_more_key_level_leaves_float32",
                "Print Assumptions the_digit_width_fits_float32",
                "Print Assumptions the_split_terms_fit_the_accumulator",
                "Print Assumptions a_width_past_the_digit_ceiling_wraps_the_high_digit",
                "Print Assumptions the_integer_exp_is_a_q28_weight",
                "Print Assumptions the_norm_limbs_are_in_range",
                "Print Assumptions the_width_functions_return_what_the_python_returns",
            ][..],
        ),
        (
            // The capstone: one decode step of the exact path, every valid
            // execution, the same next token and cache rows. The entries
            // after it are what make it describe the code: the carriers the
            // code uses meet every bound, the shapes of Qwen2.5 0.5B-7B are
            // accepted, and the premises are satisfiable. The refutation is
            // the planted float sum: a float32 softmax denominator is not
            // exact at T = 2, and the code never uses one.
            "ExactDecodeStep.v",
            &[
                "Print Assumptions every_valid_execution_takes_the_same_decode_step",
                "Print Assumptions the_reference_machine_is_valid",
                "Print Assumptions the_digits_recombine",
                // Triton's interleaved two-digit kernel: one scaled schedule
                // over both digits, exact within the linear layer's bound.
                "Print Assumptions sched_dot2_exact",
                "Print Assumptions schedules_in_fitting_carriers_are_valid",
                "Print Assumptions the_code's_carriers_meet_every_need",
                "Print Assumptions code_executions_agree",
                "Print Assumptions qwen25_shapes_are_accepted",
                "Print Assumptions the_premises_are_satisfiable_at_qwen25_0_5b",
                "Print Assumptions a_float32_denominator_is_not_exact",
                // Ragged batches: a left-padded row decodes as the prompt
                // alone, given the run-level conditions attend_pad_ok.
                "Print Assumptions attend_padding",
                "Print Assumptions left_padding_changes_nothing",
            ][..],
        ),
    ]
}

#[test]
fn each_proof_still_proves_the_thing_it_exists_for() {
    for (name, required) in content_controls() {
        let path = proofs_dir().join(name);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read proofs/{name}: {e}"));
        for needle in required {
            assert!(
                names_something_real(&src, needle),
                "proofs/{name} no longer STATES `{needle}`, which is what makes \
                 the file worth checking. It is gone or renamed; point this gate \
                 at whatever replaced it rather than deleting the entry. \
                 (A mention in the file's header prose does not count - see \
                 `names_something_real`.)"
            );
        }
    }
}

/// A needle must appear at a **declaration site**, not merely somewhere in the
/// file.
///
/// This used to be a bare `src.contains(needle)`, and that is satisfied by the
/// theorem's own name in the header doc comment - which every proof here has,
/// because each file's header explains what it proves and refers to its
/// theorems as `[the_name]`. Measured rather than reasoned about: deleting
/// `a_total_weight_below_one_ulp_admits_a_zero_denominator` from
/// `SoftmaxErrorBound.v` **together with its `Print Assumptions` line** left
/// all four tests in this file green, because the header still named it. The
/// `Print Assumptions` line has to go too or `coqc` catches the dangling
/// reference; with both gone, nothing did.
///
/// It is a hole in the gate that guards the whole directory, not in one entry,
/// and it is pre-existing - it was found by mutating a new file and applies to
/// every entry. (Seventeen at the time; eighteen since
/// `proofs/ExactGemmAllocation.v`. The count is deliberately not restated - it
/// is derived from the directory, and a number here would go stale silently.)
fn names_something_real(src: &str, needle: &str) -> bool {
    let tokens = coq_tokens(src);
    // Match the whole command and identifier. Substring matching accepted
    // commented-out commands and `foo_bar` as a control for `foo`.
    let printed_name = needle.strip_prefix("Print Assumptions ").unwrap_or(needle);
    let printed = tokens.windows(4).any(|command| {
        command[0] == "Print"
            && command[1] == "Assumptions"
            && command[2] == printed_name
            && command[3] == "."
    });
    if needle.starts_with("Print Assumptions ") {
        return printed;
    }
    const KEYWORDS: [&str; 9] = [
        "Theorem", "Lemma", "Corollary", "Definition", "Fixpoint", "Example",
        "Remark", "Fact", "Proposition",
    ];
    // ...and a bare name is also satisfied by its own `Print Assumptions`,
    // which cannot survive the theorem being deleted.
    if printed {
        return true;
    }
    tokens.windows(2).any(|declaration| {
        KEYWORDS.contains(&declaration[0].as_str()) && declaration[1] == needle
    })
}

#[test]
fn proof_content_controls_ignore_comments_and_require_whole_names() {
    let commented = "(*\nLemma required : True. Proof. exact I. Qed.\n\
                     (* nested documentation *)\nPrint Assumptions required.\n*)";
    for needle in ["required", "Print Assumptions required"] {
        assert!(!names_something_real(commented, needle));
        assert!(!names_something_real(
            "Lemma required_else : True. Proof. exact I. Qed.\n\
             Print Assumptions required_else.",
            needle
        ));
        assert!(!names_something_real(
            "Goal True. Proof. idtac \"Print Assumptions required.\". exact I. Qed.",
            needle
        ));
        assert!(names_something_real(
            "  Lemma\nrequired : True. Proof. exact I. Qed. Print\nAssumptions required.",
            needle
        ));
    }
    assert!(!names_something_real("Lemma required' : True.", "required"));
    assert!(!names_something_real(
        "Print Assumptions required'.",
        "Print Assumptions required"
    ));
}

#[test]
fn assumption_command_count_ignores_comments_and_handles_coq_whitespace() {
    let src = "(* Print Assumptions hidden.\nPrint Assumptions also_hidden. *)\n\
               Print (* nested (* comment *) *) Assumptions first.\n\
               Print\nAssumptions second. Print Assumptions third.\n\
               Goal True. Proof. idtac \"(* \"\"Print Assumptions quoted.\"\"\". exact I. Qed.";
    let asked = coq_tokens(src)
        .windows(2)
        .filter(|tokens| tokens[0] == "Print" && tokens[1] == "Assumptions")
        .count();
    assert_eq!(asked, 3);
}

/// A new `.v` file must arrive with a content control, or it gets the two weak
/// checks (compiles, no axioms) that an empty file also passes.
#[test]
fn every_proof_has_a_content_control() {
    let controlled: Vec<&str> = content_controls().iter().map(|(n, _)| *n).collect();
    let missing: Vec<String> = proof_sources()
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .filter(|n| !controlled.contains(&n.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "these proofs have no content control, so they would pass this suite \
         while containing no theorems: {missing:?}. Add each to \
         `content_controls()` naming its load-bearing theorem."
    );
}

// ────────────────────────────────────────────────────────────────────────────
//  The scope disclaimers, which are the programme's own work queue
// ────────────────────────────────────────────────────────────────────────────
//
// Every proof here carries a "what this does NOT prove" section, and
// `docs/verified_kernel_process.md` step 12 says to read those back as the next
// increment's queue - which has produced two increments already. A FALSE item
// on that queue costs a whole one: it sends the next session at work that is
// already done, and that has happened. Audited 2026-09-01: SEVEN false claims
// across four files, every one UNDERSTATING what is proved.
//
//   ExactGemmChain.v    named a "kc-panel loop" as proved by nothing. No such
//                       loop exists - the driver passes the full K to both
//                       packers - and the session that wrote `ExactGemmWhole.v`
//                       had already established that and recorded it THERE.
//                       Also said the tiling and band layers "are not joined
//                       to this one"; `ExactGemmWhole.v` joins them.
//   ExactGemmKsplit.v   "THE MICRO-KERNEL IS NOT VERIFIED. Packing, the 2-D
//                       register tile, the masked tails and the periodic
//                       int32 -> int64 flush are all assumed" - all four are
//                       proved, in Packing, RegisterTile, Micro and Chain. And
//                       "the M and N partitioning ... is not modelled", which
//                       is `ExactGemmTiling.c_written_exactly_once`.
//   ExactGemmMicro.v    put the register tile and the masked tails on the ISA
//                       side of the TCB boundary, overstating the TCB by three
//                       proof files.
//   GridStrideSplit.v   said the per-thread body was unmodelled, when the file
//                       one level ABOVE it bounds two thirds of it; and called
//                       its schedule "a PTX string template rather than an
//                       emitted expression", contradicted by the file's own
//                       `Require AttentionSchedule`.
//
// The cause is structural rather than careless: each file states a GLOBAL
// negative, and nothing updates a sibling's prose when a later file discharges
// what it names. The files whose lists stayed true are the dependency ROOTS -
// `ExactGemmWhole.v`, `SoftmaxErrorBound.v`, `CountingSort.v` - for the reason
// that makes them the right owners: a capstone is rewritten every time the
// chain below it closes.
//
// **What these two gates can and cannot see.** They catch the FORM of the
// failure - a proof with something above it making a claim about the whole
// programme, and a citation to a file that no longer exists. They cannot see a
// wrong claim phrased in words they do not know. Prose staleness is not
// mechanically decidable in general, which is why `ExactGemmSchedule.v` is
// GENERATED rather than checked; these are the reachable part, and the
// unreachable part is stated here rather than implied.

/// A global negative may only be stated by a file with nothing above it.
///
/// The capstones are DERIVED, not listed: a proof is one exactly when no other
/// proof `Require`s it. That is the principled form of the rule - a file with
/// something above it cannot know what the programme leaves open, because the
/// file above it may have closed it. Deriving also keeps this from becoming a
/// third copy of the dependency graph, which is the mistake `ExactGemmSchedule.v`
/// exists to avoid.
///
/// **It earned its place on the first run.** `GridStrideSplit.v` said the
/// per-thread body - "the integer exp, the Q0.28 weight, the int8 `V` load" -
/// was not modelled. `SoftmaxErrorBound.v` `Require`s it and bounds the first
/// two. That hit was classified as a false positive and it was not; checking
/// before dismissing turned it into the seventh finding.
///
/// The escape is a SCOPING marker on the line, not an exemption list: a claim
/// that says "not modelled HERE" is owned by the file making it and cannot rot
/// when a sibling changes. `ExactGemmTiling.v` writes it that way and passes
/// unchanged - that is the control, and without it this gate would ban a
/// correct sentence.
#[test]
fn only_the_capstone_states_a_global_negative() {
    /// Phrases that assert something about the PROGRAMME rather than about the
    /// file containing them. Every one is verbatim from a claim that was false
    /// when audited, so this list is evidence rather than taste.
    const GLOBAL: &[&str] = &[
        "not proved anywhere",
        "IS NOT VERIFIED",
        "is not modelled",
        "are not joined to this one",
        "rather than an emitted expression",
        "Phase 1's other half",
    ];
    /// A claim carrying one of these is scoped to its own file and cannot rot
    /// when a sibling changes.
    ///
    /// The markers are DELIMITED, and that is not fussiness: a bare `"here"`
    /// is a substring of **"anywhere"**, so the first version of this gate
    /// exempted the exact phrase `"not proved anywhere"` it exists to catch,
    /// and the mutation restoring that paragraph came back green. A scoping
    /// word that occurs inside the global claim is not a scoping word.
    const SCOPED: &[&str] = &[" here ", " here,", " here.", "HERE", "this file", "in this one"];

    let sources = proof_sources();

    // Derive the capstones: a proof nothing else `Require`s.
    let mut required: std::collections::BTreeSet<String> = Default::default();
    let mut files: Vec<(String, String)> = Vec::new();
    for p in &sources {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let src = std::fs::read_to_string(p).expect("proof source is utf-8");
        for line in src.lines() {
            let l = line.trim();
            let rest = l
                .strip_prefix("Require Import ")
                .or_else(|| l.strip_prefix("Require "));
            if let Some(rest) = rest {
                for m in rest.trim_end_matches('.').split_whitespace() {
                    required.insert(format!("{m}.v"));
                }
            }
        }
        files.push((name, src));
    }
    let capstones: Vec<&str> = files
        .iter()
        .map(|(n, _)| n.as_str())
        .filter(|n| !required.contains(*n))
        .collect();

    // Non-vacuity, BOTH directions. A sweep over no files agrees perfectly,
    // and a `Require` parse that found nothing would make every file exempt.
    assert!(
        files.len() >= 15,
        "only {} proof files were scanned; the sweep is not looking where it thinks",
        files.len()
    );
    // The parse must find the graph: proofs that another proof `Require`s. A
    // count, not a share of the files: every standalone proof added (one that
    // nothing else needs) is a capstone by this rule, and at 29 of 57 the old
    // share check failed with the parse intact (2026-11-27, 28 required).
    let required_files = files.iter().filter(|(n, _)| required.contains(n)).count();
    assert!(
        required_files >= 20,
        "only {} of {} proofs are `Require`d by another - the `Require` scan is not \
         parsing the dependency graph, so nearly every file would be exempt",
        required_files,
        files.len()
    );
    assert!(
        capstones.len() < files.len(),
        "every one of {} proofs came out as a capstone - the `Require` scan found no edge",
        files.len()
    );

    let mut offenders = Vec::new();
    for (name, src) in &files {
        if capstones.contains(&name.as_str()) {
            continue;
        }
        let lines: Vec<&str> = src.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if !GLOBAL.iter().any(|g| line.contains(g)) {
                continue;
            }
            if SCOPED.iter().any(|w| line.contains(w)) {
                continue;
            }
            // A correction note is allowed to QUOTE the claim it retracts,
            // otherwise the fix cannot explain itself - the same scoping the
            // README layout gate needed for the same reason.
            let lo = i.saturating_sub(4);
            let near = lines[lo..(i + 2).min(lines.len())].join(" ");
            if near.contains("CORRECTED") || near.contains("previously") {
                continue;
            }
            offenders.push(format!("{name}:{}: {}", i + 1, line.trim()));
        }
    }

    assert!(
        offenders.is_empty(),
        "a proof that is not a dependency ROOT states a negative about the whole \
         programme. Those rot silently - nothing updates a sibling's prose when a \
         later file discharges what it names, and seven such claims were false when \
         audited, every one understating what is proved. Scope it to this file \
         (\"not modelled HERE; `X.v` proves it\") and leave the global list to the \
         capstone. Capstones here: {capstones:?}\n  {}",
        offenders.join("\n  ")
    );
}

/// A repo-relative path named in a proof, a gate or a source file must exist.
///
/// The cheapest kind of documentation rot, and one this repo has recorded three
/// times: `docs/` naming three verification scripts that were never in the tree,
/// `self_hosted/` rotting because it was named only in a doc, and a
/// `src/zk_fuzz.rs` docstring citing a test file that did not exist. Found one
/// live instance when written - a citation to `zk_divmod_soundness.rs`, renamed
/// to `zk_gadget_soundness.rs` when it grew past division.
///
/// **Anchor the START of the path.** The first version of this sweep did not, so
/// a path under `crates/y-gpu/tests/` matched as though it were under `tests/`
/// and was reported missing while existing - two of that sweep's three findings
/// were that. A blanket pattern is only safe when it names one thing.
#[test]
fn every_path_a_proof_or_a_gate_cites_exists() {
    const ROOTS: &[&str] = &[
        "proofs", "tests", "src", "tools", "docs", "crates", "fuzz", "python",
    ];
    const EXT: &[&str] = &[".v", ".rs", ".py", ".md", ".circom"];

    let root = repo();
    let mut checked = 0usize;
    let mut missing = Vec::new();

    let mut files: Vec<PathBuf> = proof_sources();
    for dir in ["tests", "src"] {
        for e in std::fs::read_dir(root.join(dir)).expect("directory is readable") {
            let p = e.expect("entry").path();
            if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                files.push(p);
            }
        }
    }

    for f in &files {
        let src = std::fs::read_to_string(f).unwrap_or_default();
        let here = f.strip_prefix(&root).unwrap_or(f).display().to_string();
        // Splitting on non-path characters is what anchors the START: a token
        // is a whole path or it is not one at all.
        for tok in src.split(|c: char| !(c.is_alphanumeric() || "._/-".contains(c))) {
            let Some(slash) = tok.find('/') else { continue };
            let head = &tok[..slash];
            if !ROOTS.contains(&head) {
                continue;
            }
            if !EXT.iter().any(|e| tok.ends_with(e)) {
                continue;
            }
            checked += 1;
            // `python/` used to be a SIBLING of the cargo project and was
            // resolved against the parent directory as a fallback. The merge
            // 37651fb moved it inside, and the fallback became a hole: a
            // citation to a `python/` file missing from the repository would
            // pass whenever an old outer checkout happened to sit beside it.
            let exists = root.join(tok).exists();
            if !exists {
                missing.push(format!("{here} -> {tok}"));
            }
        }
    }

    // Non-vacuity: counts CITATIONS RESOLVED, not candidate files - a sweep
    // that parsed nothing reports "none missing" perfectly.
    assert!(
        checked > 200,
        "only {checked} path citations were resolved; the scan is not reading what \
         it thinks it is"
    );
    missing.sort();
    missing.dedup();
    assert!(
        missing.is_empty(),
        "{} citation(s) name a file that does not exist. A docstring pointing at a \
         renamed or deleted file is how a claim about what pins something becomes \
         unfalsifiable:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}
