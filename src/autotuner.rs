// ============================================================
//  Y  —  JIT Dynamic Autotuning Pass
//  autotuner.rs
//
//  Selects the CTA tile layout, warp split and pipeline depth for a
//  `@tile`d GEMM.
//
//  There are two selectors here and they are not equal in standing:
//
//   * EMPIRICAL (`crate::empirical_autotune`, enabled with
//     `set_tuning_mode`/`Y_AUTOTUNE`) compiles every candidate through the
//     real codegen path, runs it on the device that is actually present,
//     correctness-checks it, and keeps the fastest. Its answer is a
//     measurement. Results are cached per (M, N, K, precision, GPU) in
//     `.ysu_hw_profile` so the cost is paid once per shape per machine.
//
//   * HEURISTIC (`score_candidate`) is the fallback for when measurement is
//     impossible or unwanted: no NVIDIA driver, no device, a shape nobody
//     has tuned yet, or `--no-autotune` (`TuningMode::Analytic`, for builds
//     that must not depend on this machine's cache).
//
//     It is a hand-fitted analytic model whose
//     coefficients were fitted to 15 configurations measured at M=N=K=4096
//     on one GPU; it took three correction rounds before it stopped
//     regressing the small shapes, and every coefficient in it encodes that
//     card's register file, shared-memory ceiling and scheduler count. It
//     ranks; it does not measure. Treat any number it produces as a guess.
// ============================================================


use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use crate::ptx_emitter::CtaTileConfig;
use crate::sentinel::HardwareProfile;

/// Compile-time operand precision for a `@tile`d kernel. Currently plumbed
/// through as call-site compatibility only (`generate_candidates`/`autotune`
/// accept it but don't yet branch the search space on it) - this file was
/// restored from an older checkpoint that predates whatever precision-aware
/// tuning logic the working tree previously had (see git history), so this
/// is a deliberately conservative reintroduction of the public API surface
/// `ptx_emitter.rs`/`c_api.rs`/`src/bin/autotune_verify.rs` call with, not a
/// claim that per-precision tuning is implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F16,
    F32,
    BF16,
    Fp8,
}

/// The warp tiling's own precondition, checked rather than assumed.
///
/// Every tensor-core GEMM in this backend cuts a CTA tile into
/// `warps_m x warps_n` warp tiles and each warp tile into `frag_m x frag_n`
/// MMA fragments, deriving the counts by TRUNCATING integer division:
///
/// ```text
/// per_warp_m = cta_m / warps_m;   num_i = per_warp_m / frag_m
/// ```
///
/// A warp then writes `num_i * frag_m` rows, the next warp's base is
/// `per_warp_m` further on, and the next CTA's is `cta_m` further on. Unless
/// `frag_m * warps_m` divides `cta_m` EXACTLY, those three strides disagree
/// and the output tile has HOLES - the emitter still produces a well-formed
/// kernel, `ptxas` still accepts it, and the rows in the gaps keep whatever
/// was in the output buffer.
///
/// This is `ExactGemmTiling.c_written_exactly_once` failing, the theorem the
/// CPU chain has had since the tiling increment, on the GPU where there was
/// no equivalent.
///
/// Measured on `tests/gemm_f16_1024.ysu` with
/// `Y_CTA_OVERRIDE=96,128,32,4,2,3`: each CTA advances 96 rows and writes
/// 64, in four striped gaps (rows 16-23, 40-47, 64-71, 88-95), under
/// "Compilation Successful!" and exit 0.
///
/// It was three `debug_assert_eq!`s at each of three emitters. `Cargo.toml`
/// has no `[profile.release]`, so rustc's default `debug-assertions = false`
/// applies and **none of them is in the shipping binary** - the same
/// gates-nothing shape as `@require`. All 23 built-in candidates satisfy the
/// constraint; the paths that bypass the list are `Y_CTA_OVERRIDE`,
/// `Y_SWIGLU_TILE` and a persisted `.ysu_hw_profile` autotune line, none of
/// which validated anything beyond the field count.
///
/// The fragment shape is a PARAMETER because the three consumers disagree:
/// the F16 and SwiGLU GEMMs are m16n16k16 and the FP8 one is m16n8k32.
/// Hardcoding 16 here would pass an FP8 tile that does not tile.
pub fn validate_warp_tiling(
    cta_m: u32,
    cta_n: u32,
    cta_k: u32,
    warps_m: u32,
    warps_n: u32,
    frag_m: u32,
    frag_n: u32,
    frag_k: u32,
) -> Result<(), String> {
    if warps_m == 0 || warps_n == 0 {
        return Err(format!(
            "warp split {}x{} has a zero dimension; a CTA needs at least one warp on each axis",
            warps_m, warps_n
        ));
    }
    for (name, ext, warps, frag, axis) in [
        ("cta_m", cta_m, warps_m, frag_m, "rows"),
        ("cta_n", cta_n, warps_n, frag_n, "columns"),
        ("cta_k", cta_k, 1, frag_k, "k-elements"),
    ] {
        let granule = frag * warps;
        if ext == 0 || ext % granule != 0 {
            let covered = (ext / warps / frag) * frag * warps;
            return Err(format!(
                "CTA tile {}={} does not tile: each warp covers {} {} of the {} it \
                 strides, so {} {} per CTA are NEVER WRITTEN and keep whatever was in \
                 the output buffer. {} must be a non-zero multiple of {} \
                 (frag {} x warps {}). Nearest legal values: {} or {}.",
                name, ext, covered, axis, ext,
                ext - covered, axis,
                name, granule, frag, warps,
                (ext / granule) * granule,
                ((ext / granule) + 1) * granule,
            ));
        }
    }
    Ok(())
}

/// Represents a candidate configuration parameter set for autotuning.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AutotuneCandidate {
    pub cta_m: u32,
    pub cta_n: u32,
    pub cta_k: u32,
    pub warps_m: u32,
    pub warps_n: u32,
    pub num_stages: u32,
    pub num_warps: u32,
}

impl AutotuneCandidate {
    pub fn to_cta_tile_config(&self) -> CtaTileConfig {
        CtaTileConfig {
            cta_m: self.cta_m,
            cta_n: self.cta_n,
            cta_k: self.cta_k,
            warps_m: self.warps_m,
            warps_n: self.warps_n,
            mma_m: 16,
            mma_n: 16,
            mma_k: 16,
            num_stages: self.num_stages,
            num_warps: self.num_warps,
        }
    }
}

impl Precision {
    /// Stable tag for the on-disk cache key. Kept short and explicit rather
    /// than derived from `Debug`, so a rename of the enum variant cannot
    /// silently invalidate (or worse, silently alias) everyone's cache.
    pub fn tag(&self) -> &'static str {
        match self {
            Precision::F16 => "F16",
            Precision::F32 => "F32",
            Precision::BF16 => "BF16",
            Precision::Fp8 => "FP8",
        }
    }
}

/// Each resolved tile, and whether it was measured on this shape (persisted
/// or just now) rather than borrowed or modelled: the split-K rule's tile
/// (`Autotuner::autotune_split_planned`) yields only to a measured one.
static AUTOTUNE_CACHE: OnceLock<Mutex<HashMap<(u32, u32, u32, u32), (AutotuneCandidate, bool)>>> = OnceLock::new();

fn get_cache() -> &'static Mutex<HashMap<(u32, u32, u32, u32), (AutotuneCandidate, bool)>> {
    AUTOTUNE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// How `autotune` is allowed to decide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuningMode {
    /// Use a persisted measurement for this (shape, precision, GPU) if one
    /// is on disk; otherwise fall back to the analytic model. Never measures
    /// anything itself.
    ///
    /// The default, so that linking the library or running `cargo test`
    /// never silently starts benchmarking on someone's device, while an
    /// ordinary build still gets the benefit of any tuning already done on
    /// that machine.
    Cached,
    /// Analytic model only - ignore the persisted cache as well as the GPU.
    ///
    /// This is the mode for reproducible codegen: the output depends only on
    /// the source and the hardware profile, not on whether someone happened
    /// to run `--autotune` on this checkout. Wanted for CI, for
    /// bit-comparable builds, and when cross-compiling for a card that is
    /// not the one in this machine.
    Analytic,
    /// Measure on the real device, but trust a persisted result for this
    /// (M, N, K, precision, GPU) if one exists.
    Measure,
    /// Measure and overwrite whatever was persisted. For re-tuning after a
    /// codegen change, which the cache cannot detect on its own.
    Remeasure,
}

static TUNING_MODE: AtomicU8 = AtomicU8::new(0);

fn mode_to_u8(m: TuningMode) -> u8 {
    match m {
        TuningMode::Cached => 0,
        TuningMode::Analytic => 1,
        TuningMode::Measure => 2,
        TuningMode::Remeasure => 3,
    }
}

fn u8_to_mode(v: u8) -> TuningMode {
    match v {
        1 => TuningMode::Analytic,
        2 => TuningMode::Measure,
        3 => TuningMode::Remeasure,
        _ => TuningMode::Cached,
    }
}

impl TuningMode {
    /// Whether this mode may read a persisted measurement.
    fn may_use_cache(self) -> bool {
        matches!(self, TuningMode::Cached | TuningMode::Measure)
    }
    /// Whether this mode may run candidates on the GPU.
    fn may_measure(self) -> bool {
        matches!(self, TuningMode::Measure | TuningMode::Remeasure)
    }
}

/// Sets the process-wide tuning mode. The CLI calls this; `Y_AUTOTUNE`
/// overrides it so a measurement run can be forced (or suppressed) without
/// recompiling the compiler.
pub fn set_tuning_mode(mode: TuningMode) {
    TUNING_MODE.store(mode_to_u8(mode), Ordering::Relaxed);
}

pub fn tuning_mode() -> TuningMode {
    match std::env::var("Y_AUTOTUNE").ok().as_deref() {
        Some("off") | Some("analytic") => return TuningMode::Analytic,
        Some("cached") => return TuningMode::Cached,
        Some("measure") | Some("on") => return TuningMode::Measure,
        Some("force") | Some("remeasure") => return TuningMode::Remeasure,
        Some(other) => {
            eprintln!(
                "[Y autotuner] ignoring unknown Y_AUTOTUNE='{}' \
                 (want off|cached|measure|force)",
                other
            );
        }
        None => {}
    }
    u8_to_mode(TUNING_MODE.load(Ordering::Relaxed))
}

thread_local! {
    static FORCED_CONFIG: RefCell<Option<AutotuneCandidate>> = const { RefCell::new(None) };
}

/// Runs `f` with `autotune` pinned to `candidate`.
///
/// This is what lets the empirical tuner compile a specific candidate
/// through the *real* emitter: `emit_tensor_core_gemm_kernel` calls
/// `autotune` itself, so without this hook, emitting a candidate in order to
/// measure it would re-enter tuning and recurse without bound. The guard
/// clears on unwind as well as on normal return, so a panic inside codegen
/// cannot leave the process permanently pinned to one tile.
pub fn with_forced_config<R>(candidate: &AutotuneCandidate, f: impl FnOnce() -> R) -> R {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            FORCED_CONFIG.with(|c| *c.borrow_mut() = None);
        }
    }
    FORCED_CONFIG.with(|c| *c.borrow_mut() = Some(candidate.clone()));
    let _guard = Guard;
    f()
}

fn forced_config() -> Option<AutotuneCandidate> {
    FORCED_CONFIG.with(|c| c.borrow().clone())
}

// ── persistent, device-keyed result cache ──────────────────
//
// Stored in `.ysu_hw_profile` alongside the sentinel's hardware
// measurements, which gives cache invalidation for free and for the right
// reason: that file is rewritten only when the hardware is re-probed, so
// deleting it (the documented way to force a re-probe after a driver, GPU or
// governor change) discards tuning results measured on the old
// configuration at the same time. The sentinel's own parser looks values up
// by key and ignores lines it does not recognise, so appending here cannot
// disturb it.

const PROFILE_PATH: &str = ".ysu_hw_profile";

/// `AUTOTUNE_F16_4096x4096x4096_NVIDIAGeForceRTX4070TiSUPER`.
///
/// The GPU is part of the key, not just an implicit property of the file, so
/// that a profile copied between machines - or a machine with two different
/// cards - cannot serve one card's tile choice to the other.
/// `tag` is `Precision::tag`, or `split_tag`'s for a split-K GEMM.
fn persist_key(m: u32, n: u32, k: u32, tag: &str, hw_profile: &HardwareProfile) -> String {
    let gpu: String = hw_profile
        .gpu_name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    format!(
        "AUTOTUNE_{}_{}x{}x{}_{}",
        tag,
        m,
        n,
        k,
        if gpu.is_empty() { "UnknownGPU" } else { &gpu }
    )
}

fn parse_persisted_value(value: &str) -> Option<AutotuneCandidate> {
    let parts: Vec<u32> = value.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    if parts.len() != 6 {
        return None;
    }
    Some(AutotuneCandidate {
        cta_m: parts[0],
        cta_n: parts[1],
        cta_k: parts[2],
        warps_m: parts[3],
        warps_n: parts[4],
        num_stages: parts[5],
        num_warps: parts[3] * parts[4],
    })
}

/// The split counts the split-K rule (`Autotuner::split_k_plan`) tries.
pub const SPLIT_K_COUNTS: [u32; 3] = [2, 4, 8];

/// The least depth of one split the split-K rule accepts. The partials and
/// the reduction cost the same per split at any K, so their share of the
/// time falls as K / S grows: from the two 512 x 512 records in 4 splits,
/// about 5% at K / S = 4096 and, scaled, 21% at 1024, against about 11%
/// gained (docs/benchmarks/f16_auto_splitk_2026-10-25/).
pub const SPLIT_K_MIN_DEPTH: u32 = 4096;

/// The least arithmetic intensity at which the split-K rule splits: flops
/// per byte of the GEMM's compulsory DRAM traffic (A and B once in f16, C
/// once in float32). Below it the unsplit grid streams from DRAM near its
/// peak, and the partials only add traffic: at 64 x 4096 x 11008 and 14336
/// (62 flops a byte) 64 unsplit CTAs read about 600 GB/s of the RTX 4070 Ti
/// SUPER's 672, and every split took longer; at 256 x 256 x 16384 (126) and
/// 512 x 512 x K (248) splits ran faster
/// (docs/benchmarks/f16_auto_splitk_2026-10-25/). The intensity is at most
/// M, so a GEMM of at most 32 rows is never split.
pub const SPLIT_K_MIN_INTENSITY: u64 = 96;

/// The split tiles the split-K rule chooses from, largest first, eight warps
/// each: the fastest in 4 splits of 512 x 512 x 65536 and in 2 and 4 of
/// 512 x 512 x 16384 (docs/benchmarks/f16_splitk_2026-10-20/ and
/// f16_auto_splitk_2026-10-25/), with few enough accumulators a thread that
/// a split's block sums fit (`PtxEmitter::f16_block_sums_fit`; a test holds
/// every tile to it and to `validate_warp_tiling`).
pub const SPLIT_K_TILES: [AutotuneCandidate; 3] = [
    AutotuneCandidate { cta_m: 128, cta_n: 128, cta_k: 32, warps_m: 4, warps_n: 2, num_stages: 3, num_warps: 8 },
    AutotuneCandidate { cta_m: 64, cta_n: 128, cta_k: 32, warps_m: 4, warps_n: 2, num_stages: 4, num_warps: 8 },
    AutotuneCandidate { cta_m: 64, cta_n: 64, cta_k: 32, warps_m: 4, warps_n: 2, num_stages: 4, num_warps: 8 },
];

/// The least depth of one split the measured choice tries
/// (`Autotuner::split_k_measure_candidates`). The rule itself stops at
/// `SPLIT_K_MIN_DEPTH`, but measured, 8 splits of 2048 were the fastest at
/// 256 x 256 x 16384 with A and B from DRAM
/// (docs/benchmarks/f16_auto_splitk_2026-10-25/followup/).
pub const SPLIT_K_MEASURE_MIN_DEPTH: u32 = 1024;

/// The persisted-key namespace of a measured split choice:
/// `AUTOTUNE_F16_SPLITK_AUTO_<M>x<N>x<K>_<gpu>` (`Autotuner::split_k_plan`).
const SPLIT_K_AUTO_TAG: &str = "F16_SPLITK_AUTO";

/// What the split-K rule chose for an F16 GEMM, or what was measured
/// (`Autotuner::split_k_plan`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitKPlan {
    /// The split count; 1 when K is not split.
    pub splits: u32,
    /// The split kernel's tile; `None` when K is not split.
    pub tile: Option<AutotuneCandidate>,
    /// The CTA counts or the measurement behind the choice, for the PTX
    /// header and the CLI.
    pub reason: String,
    /// Whether the choice was measured on this shape rather than the rule's.
    pub measured: bool,
}

/// The split choices measured in this process (`Autotuner::split_k_plan`):
/// the CLI resolves each kernel before codegen does, and one shape is
/// measured once.
static SPLIT_K_MEASURED: OnceLock<Mutex<HashMap<(u32, u32, u32), SplitKPlan>>> = OnceLock::new();

fn split_k_measured() -> &'static Mutex<HashMap<(u32, u32, u32), SplitKPlan>> {
    SPLIT_K_MEASURED.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The persisted-key namespace of an F16 GEMM split `splits` ways in K:
/// `F16` unsplit, `F16_SPLITK<S>` otherwise. A split kernel runs `splits`
/// times the CTAs of the unsplit one, so neither may borrow the other's
/// tile: 512 x 512 x 65536 in 4 splits measured 0.86-0.93x cuBLAS on the
/// unsplit kernel's 64x64 tile and 0.98-0.99x on 128x128
/// (docs/benchmarks/f16_splitk_2026-10-20/).
fn split_tag(precision: Precision, splits: u32) -> String {
    if splits > 1 {
        format!("{}_SPLITK{}", precision.tag(), splits)
    } else {
        precision.tag().to_string()
    }
}

fn load_persisted(
    m: u32,
    n: u32,
    k: u32,
    tag: &str,
    hw_profile: &HardwareProfile,
) -> Option<AutotuneCandidate> {
    parse_persisted_value(&load_profile_value(&persist_key(m, n, k, tag, hw_profile))?)
}

/// The profile's value for `key`, if it has one.
fn load_profile_value(key: &str) -> Option<String> {
    let contents = std::fs::read_to_string(PROFILE_PATH).ok()?;
    contents
        .lines()
        .filter_map(|line| line.split_once('='))
        .find(|(found, _)| found.trim() == key)
        .map(|(_, value)| value.trim().to_string())
}

/// Nearest *measured* shape, for when this exact (M, N, K) has no entry.
///
/// Exact-match-or-nothing means any shape that is not one of the handful in
/// `tests/gemm_f16_*.ysu` silently falls through to `score_candidate`, the
/// hand-fitted analytic model this whole module exists to demote. That is a
/// real cost, not a theoretical one: at M=N=K=1000 the analytic model picks
/// `128x128x32` where measurement picks `64x128x32` - 36.0us against 31.6us,
/// **0.77x vs 0.89x of cuBLAS**. Ragged shapes are also exactly the shapes a
/// real workload has, since almost nothing outside a benchmark is a power of
/// two.
///
/// A tile measured at a *nearby* shape is a far better prior than the
/// analytic model, because tile choice varies slowly with shape - the whole
/// 2048..16384 range measures to the same `128x128x32 2x2 s2`. It is still a
/// guess, so it is bounded:
///
///   * every dimension must be within 2x of the requested one, so a decode
///     shape can never inherit a large square's tile (or vice versa) - those
///     are genuinely different regimes and the measured entries disagree;
///   * it never crosses M = 32, which 2x alone does not stop: up to 32 rows
///     the K loop adds each mma's result in float32
///     (`PtxEmitter::f16_accumulation`) and the tiles were measured for
///     decode. Borrowed at 64 x 4096 x 4096, the M = 32 tile took 1.21-1.22x
///     cuBLAS's time, the analytic model's 1.02-1.03x
///     (`docs/benchmarks/f16_llm_shapes_2026-10-24/m64/`);
///   * the winner minimises the largest per-dimension log-ratio, i.e. the
///     shape that is closest in the dimension it is *furthest* off in;
///   * the borrowed tile must satisfy `is_emittable` at the REQUESTED K, so
///     borrowing can never produce a kernel codegen would reject; and
///   * it is reported distinctly from an exact hit, naming the shape it came
///     from, so a surprising tile is traceable to the entry it was borrowed
///     from rather than looking like a measurement that never happened.
/// `k_emit` is the K a borrowed tile must be emittable at: K itself, or a
/// split-K GEMM's split depth.
fn load_persisted_nearest(
    m: u32,
    n: u32,
    k: u32,
    k_emit: u32,
    tag: &str,
    hw_profile: &HardwareProfile,
) -> Option<(AutotuneCandidate, (u32, u32, u32))> {
    let gpu: String = hw_profile
        .gpu_name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    let gpu = if gpu.is_empty() { "UnknownGPU".to_string() } else { gpu };
    let prefix = format!("AUTOTUNE_{}_", tag);
    let suffix = format!("_{}", gpu);

    let contents = std::fs::read_to_string(PROFILE_PATH).ok()?;
    let mut best: Option<(f64, AutotuneCandidate, (u32, u32, u32))> = None;

    for line in contents.lines() {
        let (name, value) = match line.split_once('=') {
            Some(v) => v,
            None => continue,
        };
        let name = name.trim();
        let shape = match name.strip_prefix(&prefix).and_then(|s| s.strip_suffix(&suffix)) {
            Some(s) => s,
            None => continue,
        };
        let dims: Vec<u32> = shape.split('x').filter_map(|d| d.parse().ok()).collect();
        if dims.len() != 3 {
            continue;
        }
        let (cm, cn, ck) = (dims[0], dims[1], dims[2]);
        if cm == 0 || cn == 0 || ck == 0 {
            continue;
        }
        // Decode's tiles stay with decode (above).
        if (cm <= 32) != (m <= 32) {
            continue;
        }

        // Bounded to 2x per dimension, and scored by the worst dimension.
        let ratio = |a: u32, b: u32| (a as f64 / b as f64).ln().abs();
        let d = ratio(cm, m).max(ratio(cn, n)).max(ratio(ck, k));
        if d > std::f64::consts::LN_2 {
            continue;
        }

        let cand = match parse_persisted_value(value) {
            Some(c) => c,
            None => continue,
        };
        if !crate::empirical_autotune::is_emittable(&cand, k_emit) {
            continue;
        }
        if best.as_ref().map_or(true, |(bd, _, _)| d < *bd) {
            best = Some((d, cand, (cm, cn, ck)));
        }
    }

    best.map(|(_, c, s)| (c, s))
}

fn store_persisted(
    m: u32,
    n: u32,
    k: u32,
    tag: &str,
    hw_profile: &HardwareProfile,
    candidate: &AutotuneCandidate,
) {
    let key = persist_key(m, n, k, tag, hw_profile);
    let value = format!(
        "{},{},{},{},{},{}",
        candidate.cta_m,
        candidate.cta_n,
        candidate.cta_k,
        candidate.warps_m,
        candidate.warps_n,
        candidate.num_stages
    );
    store_profile_value(&key, &value);
}

/// Sets `key=value` in the profile, replacing a line for `key`.
fn store_profile_value(key: &str, value: &str) {
    let key = key.to_string();
    let entry = format!("{}={}", key, value);

    let existing = std::fs::read_to_string(PROFILE_PATH).unwrap_or_default();
    if existing.is_empty() {
        // No hardware profile yet - do not create a half-written one that
        // `check_or_probe_hardware` would then treat as a completed probe and
        // skip probing entirely. Losing one cached tile choice is cheap;
        // losing the whole hardware profile is not.
        return;
    }

    let mut lines: Vec<String> = Vec::new();
    let mut replaced = false;
    for line in existing.lines() {
        if line.split_once('=').map(|(f, _)| f.trim() == key).unwrap_or(false) {
            lines.push(entry.clone());
            replaced = true;
        } else {
            lines.push(line.to_string());
        }
    }
    if !replaced {
        lines.push(entry);
    }
    let mut out = lines.join("\n");
    out.push('\n');

    // Write-then-rename. This file holds the sentinel's whole hardware
    // profile, and a torn write does not merely lose a cached tile: a
    // truncated `.ysu_hw_profile` still *exists*, so `check_or_probe_hardware`
    // would skip probing and silently compile against half-parsed defaults.
    if let Err(e) = crate::sentinel::write_profile_atomically(PROFILE_PATH, &out) {
        eprintln!("[Y autotuner] could not persist tuning result to {}: {}", PROFILE_PATH, e);
    }
}

pub struct Autotuner;

impl Autotuner {
    /// Generates candidate search space for matrix dimensions M, N, K.
    /// `_precision` is accepted for call-site compatibility (see
    /// `Precision`'s doc comment) but does not yet affect the search space.
    pub fn generate_candidates(m: u32, n: u32, _k: u32, _precision: Precision) -> Vec<AutotuneCandidate> {
        let mut candidates = Vec::new();

        let tile_sizes = if m <= 256 || n <= 256 {
            vec![
                (16, 32, 32, 1, 1),
                (32, 32, 32, 2, 1),
                (32, 64, 32, 2, 2),
                // `cta_k = 64` variants at small `cta_m`. These were absent
                // entirely, which mattered: on a measured sweep of this
                // bucket at decode shapes (M=1..32, N=K=4096, working set
                // larger than L2) `cta_k` is the single dominant axis -
                // holding everything else fixed, 16x64x32 -> 16x64x64 went
                // 97.9us -> 76.8us and 32x64x32 -> 32x64x64 went 90.8us ->
                // 66.2us. The kernel is DRAM-bandwidth-bound here, so what
                // helps is bytes in flight per CTA (a wider K tile issues
                // more concurrent `cp.async` per stage), not more CTAs.
                (16, 64, 64, 1, 2),
                (32, 64, 64, 2, 2),
                (64, 64, 32, 2, 2),
                (64, 64, 64, 2, 2),
            ]
        } else if m <= 512 || n <= 512 {
            vec![
                (64, 64, 32, 2, 2),
                (64, 64, 64, 2, 2),
                // Second warp SPLIT for the 64x64 tile, at the same tile and
                // therefore the same grid and the same L2 re-read count.
                // Everywhere else in this function each (cta_m, cta_n) pairs
                // with exactly one split, implicitly `warps = cta/32` (a
                // 32x32 per-warp tile). That rule is wrong specifically at
                // M=N=K=512, and only there, because of the regime: a 64x64
                // tile makes a 8x8 = 64-CTA grid against 66 SMs, i.e.
                // `launch__waves_per_multiprocessor` = 0.97 - ONE CTA per SM,
                // one wave. At 2x2 that is 4 warps/SM (`sm__warps_active`
                // 8.33% of peak, half what the large squares get) with
                // nothing else resident to hide latency behind, and the
                // kernel measures 22.7% of tensor peak with a 1.62
                // long-scoreboard stall. 4x2 doubles the warps per SM without
                // touching the grid or the byte traffic, and measures
                // **1.038x** on an interleaved A/B ranked by minimum
                // (dispersion 0.5-0.65%, so it clears the tie band).
                //
                // Deliberately NOT added to the other buckets: the same
                // variants were measured at 1024, where the grid is already
                // 128 CTAs at 2 CTAs/SM, and every one LOSES (0.95x for both
                // 2x4 and 4x2) - past 4 warps/SM the smaller per-warp tile
                // costs more in operand reuse than the extra warps return.
                // The `ldmatrix` count per K-tile is fixed at 24 for both
                // splits while the MMA count halves with the per-warp tile,
                // which is the mechanism.
                (64, 64, 64, 4, 2),
                (64, 64, 32, 4, 2),
                (64, 128, 32, 2, 4),
                (64, 128, 64, 2, 4),
                (128, 64, 32, 4, 2),
                (128, 64, 64, 4, 2),
            ]
        } else {
            vec![
                // 4-warp (128-thread) variants. These were absent, and their
                // absence cost ~1.19x at every large square size: 128x128x32
                // at 2 stages fits 37888 B of smem, which is the only shape
                // in this bucket that leaves room for TWO CTAs per SM. On a
                // measured sweep at M=N=K=4096 it runs 75.8 TFLOPS against
                // 63.4 for the 128x256x32 4x4 tile this bucket used to
                // settle on - and nVidia's own cuBLAS kernel for this shape
                // (`ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn`,
                // read off ncu) is the same 128x128 tile at 128 threads with
                // 234 registers/thread and 2 blocks/SM. Fewer, fatter warps
                // beat more, thinner ones here: at a fixed 128x128x32 tile,
                // 2x2 warps measured 75.8, 4x2 measured 70.5 and 4x4 only
                // 52.5, because a `bar.sync` over 4 warps is far cheaper
                // than over 16 (ncu barrier-stall ratio 6.27 for the
                // 512-thread config vs 1.27 for cuBLAS's 128-thread one).
                (128, 128, 32, 2, 2),
                (64, 128, 32, 2, 2),
                (128, 64, 32, 2, 2),
                (128, 128, 32, 4, 2),
                (128, 128, 64, 4, 2),
                (128, 128, 128, 4, 2),
                (128, 256, 32, 4, 4),
                (128, 256, 64, 4, 4),
                (256, 128, 32, 4, 4),
                (256, 128, 64, 4, 4),
            ]
        };

        for (cta_m, cta_n, cta_k, warps_m, warps_n) in tile_sizes {
            let num_warps = warps_m * warps_n;
            // Small `cta_m` used to be hard-capped at 2 stages, which cut off
            // the exact configuration that measures fastest at decode shapes:
            // 32x64x64 at 3 stages (57.3us at M=1) beats the 64x64x64 4-stage
            // tile this bucket previously settled on (61.9us), with
            // non-overlapping run-to-run ranges. The cap made that
            // unreachable, not merely unlikely. `score_candidate` still
            // clamps to what the real smem budget can sustain, so widening
            // the option list here cannot request a depth codegen can't
            // deliver.
            let stage_options = if cta_m <= 32 { vec![2, 3, 4] } else { vec![2, 3, 4] };
            for num_stages in stage_options {
                candidates.push(AutotuneCandidate {
                    cta_m,
                    cta_n,
                    cta_k,
                    warps_m,
                    warps_n,
                    num_stages,
                    num_warps,
                });
            }
        }

        candidates
    }

    /// Candidates for an `m x n` GEMM split in K to depth `k_split`: the
    /// unsplit bucket for `m x n`, plus the large-square bucket's tiles that
    /// fit in `m x n`. Splitting multiplies the grid, so a tile too big for
    /// the unsplit grid can be the right one: at 512 x 512 x 65536 in 4
    /// splits a 128x128 tile (4x2 warps), outside the 512 bucket, measured
    /// fastest (docs/benchmarks/f16_splitk_2026-10-20/).
    pub fn generate_split_candidates(m: u32, n: u32, k_split: u32, precision: Precision) -> Vec<AutotuneCandidate> {
        let mut candidates = Self::generate_candidates(m, n, k_split, precision);
        for c in Self::generate_candidates(m.max(1024), n.max(1024), k_split, precision) {
            if c.cta_m <= m && c.cta_n <= n && !candidates.contains(&c) {
                candidates.push(c);
            }
        }
        candidates
    }

    /// Evaluates candidate configurations using hardware latency model with SM wave alignment.
    ///
    /// The shared-memory/pipeline-depth accounting below is deliberately a
    /// byte-for-byte match of `emit_tensor_core_gemm_kernel`'s real
    /// `per_stage_bytes`/`effective_stages` derivation in `ptx_emitter.rs`
    /// (same row layout, same `hw_profile.max_smem_per_sm_bytes
    /// - 4096` safety margin, same `k_tiles` clamp) rather than a cheaper
    /// approximation. A prior version of this function used unpadded byte
    /// counts and a `cta_k >= 64 -> 1 byte/elem` "FP8 detection" heuristic
    /// that this GEMM path never actually exercises (`generate_candidates`
    /// accepts but ignores `Precision` - see its doc comment - and
    /// `emit_tensor_core_gemm_kernel` is hardcoded F16-in regardless), which
    /// underestimated a 128x256x64 tile's real per-stage footprint by
    /// roughly 2.1x (24576 modeled vs 52224 real bytes at M=N=K=4096 on
    /// this project's dev GPU) and rewarded it with a full 4-stage
    /// `stage_multiplier` bonus the real codegen could never deliver (the
    /// real, padded footprint only ever fits ONE stage there, forcing
    /// `emit_tensor_core_gemm_kernel` down to its non-pipelined
    /// single-buffered fallback) - concretely, this caused the autotuner to
    /// prefer 128x256x64 (single-buffered, no latency hiding at all) over
    /// 128x256x32 (real 3-stage cp.async pipelining), a measured regression
    /// against this project's own prior-session benchmark numbers (see git
    /// history / commit 430605e's message). Rewarding only the ACHIEVABLE
    /// stage count (not the requested `candidate.num_stages`) closes that
    /// gap: a candidate whose real smem footprint can't sustain the pipeline
    /// depth it asked for now scores accordingly, instead of scoring as if
    /// it got what it asked for.
    /// The bytes one pipeline stage of `candidate` takes in shared memory, and
    /// the stages `emit_tensor_core_gemm_kernel` actually uses at depth `k`:
    /// the emitter's own derivation (`PtxEmitter::gemm_smem_tiles` layouts,
    /// the `max_smem_per_sm_bytes - 4096` ceiling, the `ceil(k / cta_k)`
    /// K-tile clamp), so the score never rewards a depth the codegen cannot
    /// deliver. `tests/ptx_smem_swizzle.rs` holds the two to each other.
    pub fn pipeline_footprint(candidate: &AutotuneCandidate, k: u32, hw_profile: &HardwareProfile) -> (u32, u32) {
        let (smem_a, smem_b) = crate::ptx_emitter::PtxEmitter::gemm_smem_tiles(candidate.cta_k, candidate.cta_n);
        let per_stage_bytes = candidate.cta_m * smem_a.stride * 2 + candidate.cta_k * smem_b.stride * 2;
        let max_smem_per_sm = if hw_profile.max_smem_per_sm_bytes > 0 { hw_profile.max_smem_per_sm_bytes } else { 49152 };
        let max_stages_by_smem = (max_smem_per_sm.saturating_sub(4096) / per_stage_bytes.max(1)).max(1);
        let k_tiles = if candidate.cta_k > 0 { k.div_ceil(candidate.cta_k).max(1) } else { 1 };
        (per_stage_bytes, candidate.num_stages.min(k_tiles).min(max_stages_by_smem).max(1))
    }

    pub fn score_candidate(candidate: &AutotuneCandidate, m: u32, n: u32, k: u32, hw_profile: &HardwareProfile) -> f64 {
        let num_tiles_m = (m + candidate.cta_m - 1) / candidate.cta_m;
        let num_tiles_n = (n + candidate.cta_n - 1) / candidate.cta_n;
        let total_ctas = num_tiles_m * num_tiles_n;

        let num_sms = if hw_profile.sm_count > 0 { hw_profile.sm_count } else { 66 };
        let ctas_per_sm = (total_ctas as f64 / num_sms as f64).ceil();

        // SM Wave Alignment Efficiency: penalize partial tail waves where many SMs stay idle
        let full_waves = total_ctas / num_sms;
        let remainder = total_ctas % num_sms;
        let wave_efficiency = if remainder == 0 {
            1.0
        } else if full_waves == 0 {
            (remainder as f64 / num_sms as f64).max(0.3)
        } else {
            (full_waves as f64 * num_sms as f64 + remainder as f64) / ((full_waves + 1) as f64 * num_sms as f64)
        };

        // Real per-stage shared-memory footprint and achievable pipeline
        // depth - see `pipeline_footprint` and this function's doc comment
        // for why matching the emitter exactly (not approximating) matters.
        let (per_stage, stages) = Self::pipeline_footprint(candidate, k, hw_profile);
        let per_stage_bytes = per_stage as f64;
        let total_load_bytes = per_stage_bytes; // one stage's worth of A+B, unpadded-equivalent for compute_intensity below
        let achievable_stages = stages as f64;
        let max_smem_per_sm = if hw_profile.max_smem_per_sm_bytes > 0 { hw_profile.max_smem_per_sm_bytes as f64 } else { 49152.0 };
        let safe_smem_ceiling = (max_smem_per_sm - 4096.0).max(0.0);

        // Total shared memory actually resident per CTA at the achievable
        // depth (in KB), penalized only if it exceeds the same real ceiling
        // `emit_tensor_core_gemm_kernel` enforces (not a flat, no-longer-
        // meaningful 48KB static-`.shared` cap - see this function's doc
        // comment: every candidate this scorer ranks now compiles through
        // the dynamic-shared-memory path, which is real-ceiling-limited,
        // not 48KB-limited).
        let smem_kb = (per_stage_bytes * achievable_stages) / 1024.0;
        let smem_penalty = if smem_kb > (safe_smem_ceiling / 1024.0) { 0.85 } else { 1.0 };

        // Pipeline stage latency hiding score - rewards the ACHIEVABLE
        // stage count, not the requested one.
        let stage_multiplier = match achievable_stages as u32 {
            // Past ~3 stages there is no DRAM latency left to hide at decode
            // shapes, so the 4th stage buys nothing and still costs smem
            // (halving resident CTAs) and `cp.async.wait_group` traffic.
            // Little's law says so and measurement agrees: at M=1, N=K=4096,
            // 16x64x64 goes 59.4us at 3 stages vs 62.6us at 4, and 32x64x64
            // 57.3us vs 61.3us - both with non-overlapping ranges. At larger
            // M there IS real MMA work for a deeper pipeline to overlap
            // with, so the full bonus stands there and every previously
            // validated square pick is unaffected.
            n if n >= 4 => if m <= 64 { 1.25 } else { 1.35 },
            3 => 1.30, // 3-stage software pipeline hides L2/DRAM memory latency
            2 => 1.15, // 2-stage double buffering
            _ => 1.0,  // 1 stage: no overlap, no latency-hiding bonus
        };

        // Grid occupancy factor: punish configurations that leave >30% of SMs idle
        let sm_occupancy = (total_ctas as f64 / (ctas_per_sm * num_sms as f64)).min(1.0);

        // Compute FLOPs per CTA tile iteration: 2 * M * N * K - but counting
        // only the rows and columns that hold REAL data. A `cta_m`-row tile
        // against an M-row problem computes `cta_m` rows per `mma`
        // regardless of how many are boundary-masked zero padding, so at
        // decode shapes (M=1, cta_m=64) 63 of every 64 rows were being
        // scored as useful work. That is what made this term increase
        // monotonically with `cta_m` no matter how skinny the problem was,
        // and it is why this bucket picked the largest tile it had for every
        // one of M=1/4/8/32.
        //
        // This correction is a no-op whenever `cta_m <= m && cta_n <= n`,
        // i.e. for every square/large shape the tile fits inside - it only
        // engages where the tile genuinely over-covers the problem.
        let useful_rows = (candidate.cta_m.min(m.max(1))) as f64;
        let useful_cols = (candidate.cta_n.min(n.max(1))) as f64;
        let flops_per_tile = 2.0 * useful_rows * useful_cols * (candidate.cta_k as f64);
        let compute_intensity = flops_per_tile / total_load_bytes;

        // Real resident-CTA count. `emit_tensor_core_gemm_kernel` sizes its
        // dynamic smem as max(pipeline stages, epilogue scratch tile), so
        // the epilogue tile has to be counted here too or this
        // over-estimates how many CTAs fit (same
        // match-the-emitter-exactly rule as `per_stage_bytes` above).
        let per_warp_n = (candidate.cta_n / candidate.warps_n.max(1)) as f64;
        let smem_c_bytes = candidate.cta_m as f64 * (per_warp_n + 4.0) * 4.0;
        let resident_bytes = (per_stage_bytes * achievable_stages).max(smem_c_bytes);
        let ctas_resident = if resident_bytes > 0.0 {
            (safe_smem_ceiling / resident_bytes).floor().max(1.0)
        } else {
            1.0
        };

        // ---- regime split ----
        //
        // These two regimes are limited by different things and must not be
        // scored by the same formula.
        //
        // A skinny/decode shape (some dimension below one CTA tile) is
        // DRAM-bandwidth-bound: what matters is bytes in flight, so data
        // reuse per tile and pipeline depth are the right objectives.
        //
        // A large square shape is not. Measured with ncu on this project's
        // dev GPU at M=N=K=4096, the 128x256x32 tile ran DRAM at 16.6% of
        // peak and L1 at 20.4%, with a shared-memory stall ratio of 0.74
        // against a math-pipe-throttle ratio of 12.15 - i.e. nowhere near
        // memory-bound in any direction. Scoring it on `compute_intensity`
        // (reuse per byte staged) optimises for a wall it never hits, and
        // the old `1.0 / ctas_per_sm` factor has no physical basis at all
        // for it: total work is invariant to tile choice, so preferring
        // fewer, larger CTAs is not preferring less work. Together those two
        // terms rated the measured-best config 3741 against 11809 for a
        // config 19% slower - a 3.2x error no bonus term could correct.
        // The utilisation model below assumes STEADY-STATE throughput over
        // many CTA waves. Two shapes fall outside that and keep the older
        // reuse-based heuristic, which empirically ranks them correctly:
        //
        //  * Skinny/decode shapes (a dimension under one CTA tile), which
        //    are DRAM-bandwidth-bound, not throughput-bound.
        //  * Small squares. At M=N=K=256 the whole GEMM is ~5us over well
        //    under one full wave of CTAs, so launch and tail effects
        //    dominate and neither model's assumptions hold; forcing the
        //    utilisation model on it picked a 16x64x64 tile that measured
        //    6.08us against 5.11us for the legacy pick. 512 is the same
        //    story. Both are re-benchmarked per size rather than assumed.
        let steady_state = m.min(n).min(k) >= 1024;

        if !steady_state {
            let concurrency_bonus = if m <= 64 && ctas_resident >= 2.0 { 1.15 } else { 1.0 };
            return compute_intensity
                * stage_multiplier
                * concurrency_bonus
                * sm_occupancy
                * wave_efficiency
                * smem_penalty
                * (1.0 / ctas_per_sm)
                * 1000.0;
        }

        // Compute-bound: score predicted tensor-pipe utilisation instead.
        // Two effects, both isolated by controlled measurement at
        // M=N=K=4096 (one variable changed at a time):
        //
        //  * Per-warp MMA parallelism `num_i * num_j` - how many independent
        //    accumulator chains a warp can interleave. Holding the tile and
        //    the smem footprint fixed and varying only the warp split:
        //    16 chains -> 75.8 TFLOPS, 8 -> 70.5, 4 -> 52.5. It saturates at
        //    16 (a 64x64 per-warp tile); past that the accumulators alone
        //    exceed the register file and it regresses (32 chains -> 40.2).
        //
        //  * Resident CTAs per SM. Holding tile AND warp split fixed and
        //    changing only pipeline depth so a second CTA fits:
        //    1 CTA -> 51.9 TFLOPS, 2 CTAs -> 75.8, a clean 1.45x. A second
        //    CTA covers the first one's `bar.sync` - which is exactly the
        //    gap ncu shows against cuBLAS (barrier-stall ratio 6.27 for Y
        //    vs 1.27 for cuBLAS's 2-CTA 128-thread kernel).
        //
        // The coefficients below are empirical, fitted to that sweep - they
        // are not derived from first principles, and they are load-bearing
        // only for RANKING. Validation is that this ranks the measured top
        // five in exactly the measured order (Spearman rho 0.757 over all
        // 15 swept configs) and that the resulting picks are re-benchmarked
        // per size rather than trusted.
        let num_i = (candidate.cta_m / candidate.warps_m.max(1) / 16).max(1) as f64;
        let num_j = (candidate.cta_n / candidate.warps_n.max(1) / 16).max(1) as f64;
        let ilp = num_i * num_j;
        let mut ilp_factor = (1.0 + ilp.min(16.0).log2()) / 5.0;
        if ilp > 16.0 {
            ilp_factor *= 16.0 / ilp; // accumulators spill past a 64x64 warp tile
        }
        // Residency only pays if the GRID actually supplies a second CTA per
        // SM. Having room for two is worthless when the launch has fewer
        // CTAs than SMs: at M=N=K=1024 a 128x128 tile yields 64 CTAs across
        // 66 SMs - under one full wave - so every SM gets exactly one CTA no
        // matter how much smem is left over, and the deeper pipeline wins
        // instead. Ignoring this regressed 1024 (48.02us vs 43.59us) and 512
        // (10.78us vs 10.30us) while still winning at 2048 and above, where
        // the grid is several waves deep.
        let supplied_ctas_per_sm = total_ctas as f64 / num_sms as f64;
        let effective_residency = ctas_resident.min(supplied_ctas_per_sm).min(2.0);
        let cta_overlap = 1.0 + 0.45 * (effective_residency - 1.0).max(0.0);

        // An SM has four warp schedulers, so a CTA of fewer than four warps
        // cannot keep them all fed no matter how good its tile is. Without
        // this the model ties configs it has no way to tell apart (at
        // M=N=K=256 a 16x32x32 1x1 tile scored identically to 32x32x32 2x1)
        // and selection falls through to candidate-list order, which picked
        // a 32-thread CTA. No effect at or above four warps, so every
        // large-square pick is unchanged by it.
        let scheduler_fill = (candidate.num_warps as f64).min(4.0) / 4.0;

        // Pipeline depth still pays once the residency cliff is already
        // cleared. `cta_overlap` saturates at two resident CTAs, so among
        // configs that all clear it the depth term is what separates them -
        // at M=N=K=256 the 32x32x32 tile fits 9 CTAs at 2 stages and 4 at
        // 4 stages, both past the cliff, and the deeper pipeline measures
        // faster (5.13us vs 5.85us). Depth is ranked BELOW residency rather
        // than above it: at 4096 the 1.45x from a second CTA outweighs the
        // 1.35x from a fourth stage, which is the ordering that was
        // inverted before.
        // A deeper K tile amortises the mainloop's fixed per-iteration cost:
        // the barrier count is one per k-tile, so `k / cta_k` barriers total,
        // and doubling `cta_k` halves them. Small but real, and it is the
        // only thing separating two configs that are otherwise identical to
        // this model - at M=N=K=512, 64x64x64 measured 10.30us against
        // 10.78us for 64x64x32 at the same warp split and stage count.
        let k_amortization = 1.0 + 0.05 * (candidate.cta_k as f64 / 32.0).max(1.0).log2();

        ilp_factor
            * cta_overlap
            * stage_multiplier
            * scheduler_fill
            * k_amortization
            * wave_efficiency
            * smem_penalty
            * 1000.0
    }

    /// Selects the CTA tile layout, warp split and pipeline depth for a GEMM
    /// of shape (M, N, K).
    ///
    /// Resolution order, highest priority first:
    ///
    ///  1. a forced config (`with_forced_config`) - the empirical tuner
    ///     compiling one specific candidate. This must come first or
    ///     measurement recurses into itself.
    ///  2. `Y_CTA_OVERRIDE` - manual override for sweeps.
    ///  3. the in-process cache.
    ///  4. the persisted per-(shape, precision, GPU) measurement in
    ///     `.ysu_hw_profile`, unless the mode is `Remeasure`.
    ///  5. a fresh measurement on the real device, if the mode allows it.
    ///  6. the analytic heuristic.
    ///
    /// `precision` selects the cache namespace. It does not yet change the
    /// candidate search space or the measured kernel: the GEMM codegen path
    /// this tunes (`emit_tensor_core_gemm_kernel`) is F16-in/F32-accumulate
    /// regardless - see `Precision`'s doc comment. Keying the cache by it
    /// anyway means the entries stay correct when that changes, rather than
    /// a future FP8 path silently inheriting F16's tile.
    pub fn autotune(m: u32, n: u32, k: u32, hw_profile: &HardwareProfile, precision: Precision) -> CtaTileConfig {
        Self::autotune_split(m, n, k, 1, hw_profile, precision)
    }

    /// `autotune` for an F16 GEMM split `splits` ways in K
    /// (`PtxEmitter::gemm_split_k`; 1 is unsplit). Measurements are keyed by
    /// the whole K and the split count (`split_tag`), measure the split
    /// kernel with its reduction (`tune_gemm_f16_split`), and draw on
    /// `generate_split_candidates`; borrowing and the analytic model work at
    /// the split's depth.
    pub fn autotune_split(m: u32, n: u32, k: u32, splits: u32, hw_profile: &HardwareProfile, precision: Precision) -> CtaTileConfig {
        Self::resolve_split(m, n, k, splits, hw_profile, precision, None)
    }

    /// `autotune_split` for the split the split-K rule chose
    /// (`split_k_plan`) on its tile `planned`: the override, a measurement of
    /// this shape and split (persisted, or taken under `--autotune`) still
    /// come first, but `planned` replaces the borrowed and the analytic tile.
    /// The rule counted its CTAs on `planned`; the analytic model gives every
    /// split of 512 x 512 the unsplit 64x64 tile, 0.86-0.93x cuBLAS where
    /// 128x128 was 0.98-0.99x (`split_tag`).
    pub fn autotune_split_planned(m: u32, n: u32, k: u32, splits: u32, hw_profile: &HardwareProfile, precision: Precision, planned: &AutotuneCandidate) -> CtaTileConfig {
        Self::resolve_split(m, n, k, splits, hw_profile, precision, Some(planned))
    }

    /// The split of an `m x n x k` F16 GEMM: one measured on this shape where
    /// there is one, else the split-K rule (`split_k_choice`) against the tile
    /// the unsplit kernel would take (`autotune`) and the profile's SM count.
    /// Only where the rule's gate passes (`split_k_gate`): elsewhere no split
    /// was measured faster, and none is measured. A measurement is read from
    /// the profile's `AUTOTUNE_F16_SPLITK_AUTO_<M>x<N>x<K>_<gpu>` entry
    /// (`parse_split_k_entry`; an invalid one is refused, and the rule
    /// decides) unless the tuning mode ignores persisted measurements, and is
    /// taken under `--autotune`: `split_k_measure_candidates`, timed together
    /// by `tune_gemm_f16_split_choice`, a tie with the unsplit kernel going to
    /// it. No candidate of the rule's kind fits every shape:
    /// near the DRAM balance more CTAs an SM win, where the GEMM is compute
    /// bound the larger tile (docs/benchmarks/f16_auto_splitk_2026-10-25/followup/).
    /// A kernel named `<name>_splitk` takes this split.
    pub fn split_k_plan(m: u32, n: u32, k: u32, hw_profile: &HardwareProfile) -> SplitKPlan {
        Self::plan_split_k(m, n, k, hw_profile, true)
    }

    /// `split_k_plan`, but never measuring: what the CLI tells a plain GEMM.
    pub fn split_k_plan_unmeasured(m: u32, n: u32, k: u32, hw_profile: &HardwareProfile) -> SplitKPlan {
        Self::plan_split_k(m, n, k, hw_profile, false)
    }

    fn plan_split_k(m: u32, n: u32, k: u32, hw_profile: &HardwareProfile, may_measure: bool) -> SplitKPlan {
        let unsplit = Self::autotune(m, n, k, hw_profile, Precision::F16);
        let rule = Self::split_k_choice(m, n, k, &unsplit, hw_profile.sm_count);
        if Self::split_k_gate(m, n, k, &unsplit, hw_profile.sm_count).is_err() {
            return rule;
        }
        if let Some(plan) = split_k_measured().lock().ok().and_then(|c| c.get(&(m, n, k)).cloned()) {
            return plan;
        }
        let mode = tuning_mode();
        let key = persist_key(m, n, k, SPLIT_K_AUTO_TAG, hw_profile);
        if mode.may_use_cache() {
            if let Some(value) = load_profile_value(&key) {
                match Self::parse_split_k_entry(&value, m, n, k) {
                    Ok((splits, tile)) => return Self::measured_split(splits, tile, &format!("measured ({})", key), k),
                    Err(why) => eprintln!("[Y autotuner] ignoring {}={}: {}; the split-K rule decides", key, value, why),
                }
            }
        }
        if !(may_measure && mode.may_measure()) {
            return rule;
        }
        let candidates = Self::split_k_measure_candidates(m, n, k, &unsplit);
        let plan = match crate::empirical_autotune::tune_gemm_f16_split_choice(m, n, k, &candidates, hw_profile, true) {
            Ok(ranked) if !ranked.is_empty() => {
                let w = &ranked[0];
                store_profile_value(&key, &Self::split_k_entry_value(w.splits, &w.candidate));
                let tile = (w.splits > 1).then(|| w.candidate.clone());
                Self::measured_split(w.splits, tile, &format!("measured by --autotune among {} kernels", ranked.len()), k)
            }
            Ok(_) => {
                eprintln!("[Y autotuner] measuring the split of M={} N={} K={} ranked nothing; the split-K rule decides", m, n, k);
                rule
            }
            Err(e) => {
                eprintln!("[Y autotuner] could not measure the split of M={} N={} K={} ({}); the split-K rule decides", m, n, k, e);
                rule
            }
        };
        if let Ok(mut c) = split_k_measured().lock() {
            c.insert((m, n, k), plan.clone());
        }
        plan
    }

    fn measured_split(splits: u32, tile: Option<AutotuneCandidate>, source: &str, k: u32) -> SplitKPlan {
        let reason = match &tile {
            Some(t) => format!(
                "{}: {} splits of K={} on {}x{}x{} tiles, {}x{} warps",
                source, splits, k / splits, t.cta_m, t.cta_n, t.cta_k, t.warps_m, t.warps_n
            ),
            None => format!("{}: not split", source),
        };
        SplitKPlan { splits, tile, reason, measured: true }
    }

    /// What `--autotune` measures for a `_splitk` GEMM: the unsplit kernel on
    /// `unsplit`'s tile, then every split count of `SPLIT_K_COUNTS` the shape
    /// allows (`PtxEmitter::gemm_split_k_shape`) at least
    /// `SPLIT_K_MEASURE_MIN_DEPTH` deep, on each tile of `SPLIT_K_TILES` no
    /// larger than M x N: at most ten kernels.
    pub fn split_k_measure_candidates(m: u32, n: u32, k: u32, unsplit: &CtaTileConfig) -> Vec<(u32, AutotuneCandidate)> {
        let mut candidates = vec![(
            1,
            AutotuneCandidate {
                cta_m: unsplit.cta_m,
                cta_n: unsplit.cta_n,
                cta_k: unsplit.cta_k,
                warps_m: unsplit.warps_m,
                warps_n: unsplit.warps_n,
                num_stages: unsplit.num_stages,
                num_warps: unsplit.num_warps,
            },
        )];
        for s in SPLIT_K_COUNTS {
            if crate::ptx_emitter::PtxEmitter::gemm_split_k_shape(m, n, k, s).is_err() || k / s < SPLIT_K_MEASURE_MIN_DEPTH {
                continue;
            }
            for t in SPLIT_K_TILES.iter().filter(|t| t.cta_m <= m && t.cta_n <= n) {
                candidates.push((s, t.clone()));
            }
        }
        candidates
    }

    /// A measured split choice's profile value: `1` unsplit, else
    /// `S,cta_m,cta_n,cta_k,warps_m,warps_n,stages`.
    pub fn split_k_entry_value(splits: u32, tile: &AutotuneCandidate) -> String {
        if splits > 1 {
            format!("{},{},{},{},{},{},{}", splits, tile.cta_m, tile.cta_n, tile.cta_k, tile.warps_m, tile.warps_n, tile.num_stages)
        } else {
            "1".to_string()
        }
    }

    /// Reads `split_k_entry_value`'s form for an `m x n x k` GEMM, refusing
    /// what codegen could not emit: a split count the shape does not allow
    /// (`PtxEmitter::gemm_split_k_shape`), a tile that does not tile
    /// (`validate_warp_tiling`) or cannot be staged at the split's depth
    /// (`empirical_autotune::is_emittable`), no pipeline stage, or anything
    /// else.
    pub fn parse_split_k_entry(value: &str, m: u32, n: u32, k: u32) -> Result<(u32, Option<AutotuneCandidate>), String> {
        let nums = value
            .split(',')
            .map(|p| p.trim().parse::<u32>().map_err(|_| format!("`{}` is not a count", p.trim())))
            .collect::<Result<Vec<u32>, String>>()?;
        match nums[..] {
            [1] => Ok((1, None)),
            [s, cta_m, cta_n, cta_k, warps_m, warps_n, num_stages] if s >= 2 => {
                crate::ptx_emitter::PtxEmitter::gemm_split_k_shape(m, n, k, s)?;
                validate_warp_tiling(cta_m, cta_n, cta_k, warps_m, warps_n, 16, 16, 16)?;
                let tile = AutotuneCandidate { cta_m, cta_n, cta_k, warps_m, warps_n, num_stages, num_warps: warps_m * warps_n };
                if num_stages == 0 || !crate::empirical_autotune::is_emittable(&tile, k / s) {
                    return Err(format!("the tile cannot be staged {} deep", k / s));
                }
                Ok((s, Some(tile)))
            }
            _ => Err("want `1`, or `S,cta_m,cta_n,cta_k,warps_m,warps_n,stages` with S of at least 2".into()),
        }
    }

    /// The split-K rule's gate: the unsplit kernel's `unsplit` tile leaves SMs
    /// without a CTA (fewer CTAs than `sm_count`) and the GEMM is not bound to
    /// DRAM (`SPLIT_K_MIN_INTENSITY`). The unsplit CTAs where it passes, the
    /// reason where it does not.
    fn split_k_gate(m: u32, n: u32, k: u32, unsplit: &CtaTileConfig, sm_count: u32) -> Result<u64, String> {
        let unsplit_ctas = m.div_ceil(unsplit.cta_m.max(1)) as u64 * n.div_ceil(unsplit.cta_n.max(1)) as u64;
        let on = format!("{} CTAs of {}x{} unsplit on {} SMs", unsplit_ctas, unsplit.cta_m, unsplit.cta_n, sm_count);
        if unsplit_ctas >= sm_count as u64 {
            return Err(on);
        }
        let (m64, n64, k64) = (m as u128, n as u128, k as u128);
        let (flops, bytes) = (2 * m64 * n64 * k64, 2 * m64 * k64 + 2 * k64 * n64 + 4 * m64 * n64);
        if flops < SPLIT_K_MIN_INTENSITY as u128 * bytes {
            return Err(format!(
                "{}, but {:.0} flops a byte of A, B and C is below {}: bound to DRAM",
                on, flops as f64 / bytes as f64, SPLIT_K_MIN_INTENSITY
            ));
        }
        Ok(unsplit_ctas)
    }

    /// The split-K rule. Split only when the unsplit kernel's `unsplit` tile
    /// leaves SMs without a CTA (fewer CTAs than `sm_count`) and the GEMM is
    /// not bound to DRAM (`SPLIT_K_MIN_INTENSITY`); then take the split count
    /// S of `SPLIT_K_COUNTS` and the tile of `SPLIT_K_TILES` no larger than
    /// M x N that give the most CTAs with each split at least
    /// `SPLIT_K_MIN_DEPTH` deep, no fewer CTAs than unsplit and no more than
    /// one for each SM; a tie keeps the smaller S, found first. (The intensity
    /// is below min(M, N), so at 96 or more neither is 64 or less: no two of
    /// the tiles then have as many CTAs at one S, and none larger than M x N
    /// could be chosen; the filter keeps that so at a lower threshold.)
    pub fn split_k_choice(m: u32, n: u32, k: u32, unsplit: &CtaTileConfig, sm_count: u32) -> SplitKPlan {
        let ctas = |cta_m: u32, cta_n: u32, s: u32| m.div_ceil(cta_m.max(1)) as u64 * n.div_ceil(cta_n.max(1)) as u64 * s as u64;
        let not_split = |reason: String| SplitKPlan { splits: 1, tile: None, reason, measured: false };
        let unsplit_ctas = match Self::split_k_gate(m, n, k, unsplit, sm_count) {
            Ok(c) => c,
            Err(why) => return not_split(why),
        };
        let on = format!("{} CTAs of {}x{} unsplit on {} SMs", unsplit_ctas, unsplit.cta_m, unsplit.cta_n, sm_count);
        let menu: Vec<&AutotuneCandidate> = SPLIT_K_TILES.iter().filter(|t| t.cta_m <= m && t.cta_n <= n).collect();
        let mut best: Option<(u64, u32, AutotuneCandidate)> = None;
        for s in SPLIT_K_COUNTS {
            if crate::ptx_emitter::PtxEmitter::gemm_split_k_shape(m, n, k, s).is_err() || k / s < SPLIT_K_MIN_DEPTH {
                continue;
            }
            for &t in &menu {
                let c = ctas(t.cta_m, t.cta_n, s);
                if c < unsplit_ctas || c > sm_count as u64 {
                    continue;
                }
                if best.as_ref().is_none_or(|(b, ..)| c > *b) {
                    best = Some((c, s, t.clone()));
                }
            }
        }
        match best {
            Some((c, s, t)) => SplitKPlan {
                reason: format!(
                    "{}; {} splits of K={} on {}x{}x{} tiles, {}x{} warps: {} CTAs",
                    on, s, k / s, t.cta_m, t.cta_n, t.cta_k, t.warps_m, t.warps_n, c
                ),
                splits: s,
                tile: Some(t),
                measured: false,
            },
            None => not_split(format!(
                "{}, but no split of K={} into {:?} parts at least {} deep fills from {} to {} CTAs",
                on, k, SPLIT_K_COUNTS, SPLIT_K_MIN_DEPTH, unsplit_ctas, sm_count
            )),
        }
    }

    fn resolve_split(m: u32, n: u32, k: u32, splits: u32, hw_profile: &HardwareProfile, precision: Precision, planned: Option<&AutotuneCandidate>) -> CtaTileConfig {
        let splits = splits.max(1);
        let k_split = k / splits;
        let tag = split_tag(precision, splits);
        // (1) Pinned to one candidate by the empirical tuner. Deliberately
        // ahead of everything else, including the cache: this call IS the
        // measurement of that candidate.
        if let Some(forced) = forced_config() {
            return forced.to_cta_tile_config();
        }

        // (2) Explicit tile override, for measuring what the search space
        // actually does on hardware instead of trusting `score_candidate`'s
        // model of it. `score_candidate` is a heuristic with no ground
        // truth behind it; this hook lets a benchmark sweep the real
        // candidate set and report measured times, which is the only way to
        // tell whether a scoring change is an improvement or just a
        // different guess.
        //
        //   Y_CTA_OVERRIDE=cta_m,cta_n,cta_k,warps_m,warps_n,num_stages
        if let Ok(spec) = std::env::var("Y_CTA_OVERRIDE") {
            let parts: Vec<u32> = spec
                .split(',')
                .filter_map(|s| s.trim().parse::<u32>().ok())
                .collect();
            if parts.len() == 6 {
                let cand = AutotuneCandidate {
                    cta_m: parts[0],
                    cta_n: parts[1],
                    cta_k: parts[2],
                    warps_m: parts[3],
                    warps_n: parts[4],
                    num_stages: parts[5],
                    num_warps: parts[3] * parts[4],
                };
                return cand.to_cta_tile_config();
            }
            eprintln!(
                "[Y autotuner] ignoring malformed Y_CTA_OVERRIDE='{}' \
                 (want cta_m,cta_n,cta_k,warps_m,warps_n,num_stages)",
                spec
            );
        }

        // (3) In-process cache. A compile unit can hold several kernels of
        // the same shape, and the CLI additionally resolves each `@tile`d
        // kernel once for its own diagnostics before codegen resolves it
        // again - so without this, a measured shape would be measured twice
        // per invocation.
        let key = (m, n, k, splits);
        if let Ok(cache) = get_cache().lock() {
            if let Some((cached_candidate, measured)) = cache.get(&key) {
                if *measured || planned.is_none() {
                    return cached_candidate.to_cta_tile_config();
                }
            }
        }

        let candidates = if splits > 1 {
            Self::generate_split_candidates(m, n, k_split, precision)
        } else {
            Self::generate_candidates(m, n, k, precision)
        };
        let mode = tuning_mode();

        // (4) A previously persisted measurement for this exact (shape,
        // precision, GPU).
        //
        // Read unconditionally, in every mode except `Remeasure` - including
        // plain `Heuristic`. Reading a result someone already measured on
        // this machine costs nothing and is strictly better information than
        // the analytic model; it is *taking* a new measurement that costs
        // seconds of GPU time and therefore has to be opted into. So the
        // workflow is: tune a shape once with `--autotune`, and every
        // ordinary build from then on emits the tuned kernel automatically.
        //
        // The staleness this cannot detect is a change to the GEMM codegen
        // itself, which can move the optimum without changing the key.
        // `--autotune-force` exists for that; hardware changes are covered
        // by deleting `.ysu_hw_profile`, which takes these entries with it.
        // `--no-autotune` (`TuningMode::Analytic`) skips this entirely, for
        // builds that must not depend on what is on this machine's disk.
        if mode.may_use_cache() {
            if let Some(cached) = load_persisted(m, n, k, &tag, hw_profile) {
                println!(
                    "         [Y autotuner] measured tile from {} for M={} N={} K={} ({}): {}x{}x{} {}x{} s{}",
                    PROFILE_PATH,
                    m, n, k, tag,
                    cached.cta_m, cached.cta_n, cached.cta_k,
                    cached.warps_m, cached.warps_n, cached.num_stages
                );
                if let Ok(mut cache) = get_cache().lock() {
                    cache.insert(key, (cached.clone(), true));
                }
                return cached.to_cta_tile_config();
            }
        }

        // (5) Fresh measurement on the real device - not for a planned split,
        // whose tile the split's own measurement chose (`split_k_plan`).
        if mode.may_measure() && planned.is_none() {
            match crate::empirical_autotune::tune_gemm_f16_split(m, n, k, splits, hw_profile, &candidates, true) {
                Ok(measured) if !measured.is_empty() => {
                    let best = measured[0].candidate.clone();
                    store_persisted(m, n, k, &tag, hw_profile, &best);
                    if let Ok(mut cache) = get_cache().lock() {
                        cache.insert(key, (best.clone(), true));
                    }
                    return best.to_cta_tile_config();
                }
                Ok(_) => {
                    eprintln!(
                        "[Y autotuner] measurement produced no ranking for M={} N={} K={}; \
                         falling back to the analytic heuristic",
                        m, n, k
                    );
                }
                Err(e) => {
                    // Not fatal, and deliberately not silent: the resulting
                    // kernel is a guess rather than a measurement, and the
                    // user asked for a measurement.
                    eprintln!(
                        "[Y autotuner] could not measure M={} N={} K={} ({}); \
                         falling back to the analytic heuristic",
                        m, n, k, e
                    );
                }
            }
        }

        // The split-K rule's own tile, in place of a borrowed or modelled one
        // (`autotune_split_planned`). Not cached: a `_splitk<S>` kernel of
        // the same shape in this process still takes steps (6) and (7).
        if let Some(planned) = planned {
            return planned.to_cta_tile_config();
        }

        // (6) A measurement taken at a NEARBY shape, before the analytic
        // model. Ordered here deliberately: this is still real on-device data,
        // just from a neighbouring shape, and tile choice varies slowly with
        // shape - so it beats `score_candidate`, which is a hand-fitted curve.
        // See `load_persisted_nearest` for the bounds that keep this honest.
        if mode.may_use_cache() {
            if let Some((borrowed, (bm, bn, bk))) =
                load_persisted_nearest(m, n, k, k_split, &tag, hw_profile)
            {
                println!(
                    "         [Y autotuner] no measurement for M={} N={} K={} ({}); borrowing the \
                     measured tile from the nearest shape M={} N={} K={}: {}x{}x{} {}x{} s{} \
                     (run with --autotune to measure this shape directly)",
                    m, n, k, tag,
                    bm, bn, bk,
                    borrowed.cta_m, borrowed.cta_n, borrowed.cta_k,
                    borrowed.warps_m, borrowed.warps_n, borrowed.num_stages
                );
                if let Ok(mut cache) = get_cache().lock() {
                    cache.insert(key, (borrowed.clone(), false));
                }
                return borrowed.to_cta_tile_config();
            }
        }

        // (7) Analytic fallback.
        let mut best_candidate = candidates[0].clone();
        let mut best_score = -1.0;

        for cand in &candidates {
            let score = Self::score_candidate(cand, m, n, k_split, hw_profile);
            if score > best_score {
                best_score = score;
                best_candidate = cand.clone();
            }
        }

        if let Ok(mut cache) = get_cache().lock() {
            cache.insert(key, (best_candidate.clone(), false));
        }

        best_candidate.to_cta_tile_config()
    }

    pub fn clear_cache() {
        if let Ok(mut cache) = get_cache().lock() {
            cache.clear();
        }
    }

    /// Rough, formula-based occupancy estimate for a candidate (register/
    /// shared-memory/warp/thread limits vs hardware maxima -> blocks/SM ->
    /// achieved occupancy). Restored as a fresh, disclosed approximation to
    /// satisfy `src/bin/autotune_verify.rs`'s diagnostic printout after this
    /// file was reverted to an older checkpoint (see `Precision`'s doc
    /// comment) - this is NOT a reconstruction of whatever occupancy model
    /// the working tree previously had; `est_regs_per_thread` in particular
    /// is a coarse heuristic (scales with accumulator-fragment count per
    /// warp), not a real `ptxas -v` measurement. Only feeds this bin's own
    /// printed diagnostics, not `score_candidate`/`autotune`'s actual
    /// selection - so it has no bearing on the real GEMM codegen path.
    pub fn estimate_occupancy(candidate: &AutotuneCandidate, hw_profile: &HardwareProfile) -> OccupancyEstimate {
        let max_regs_per_sm = if hw_profile.max_regs_per_sm > 0 { hw_profile.max_regs_per_sm as f64 } else { 65536.0 };
        let max_warps_per_sm = if hw_profile.max_warps_per_sm > 0 { hw_profile.max_warps_per_sm as f64 } else { 48.0 };
        let max_smem_per_sm = if hw_profile.max_smem_per_sm_bytes > 0 { hw_profile.max_smem_per_sm_bytes as f64 } else { 49152.0 };
        let max_threads_per_sm = if hw_profile.max_threads_per_sm > 0 { hw_profile.max_threads_per_sm as f64 } else { 1536.0 };

        let per_warp_m = (candidate.cta_m / candidate.warps_m).max(1);
        let per_warp_n = (candidate.cta_n / candidate.warps_n).max(1);
        let num_frags = ((per_warp_m / 16).max(1) * (per_warp_n / 16).max(1)) as f64;
        // 8 f32 accumulator regs/fragment + fixed per-thread bookkeeping overhead.
        let est_regs_per_thread = 32.0 + num_frags * 8.0 * 1.5;

        let threads_per_block = (candidate.num_warps * 32) as f64;
        let regs_limit_blocks = (max_regs_per_sm / (est_regs_per_thread * threads_per_block)).floor();

        let bytes_per_elem = 2.0; // F16 operand staging
        let smem_bytes_per_block =
            (candidate.cta_m * candidate.cta_k) as f64 * bytes_per_elem * candidate.num_stages as f64
                + (candidate.cta_k * candidate.cta_n) as f64 * bytes_per_elem * candidate.num_stages as f64;
        let smem_limit_blocks = if smem_bytes_per_block > 0.0 { (max_smem_per_sm / smem_bytes_per_block).floor() } else { f64::INFINITY };

        let warps_limit_blocks = (max_warps_per_sm / candidate.num_warps as f64).floor();
        let threads_limit_blocks = (max_threads_per_sm / threads_per_block).floor();

        let achieved_blocks_per_sm = [regs_limit_blocks, smem_limit_blocks, warps_limit_blocks, threads_limit_blocks]
            .into_iter()
            .fold(f64::INFINITY, f64::min)
            .max(0.0);
        let achieved_occupancy = ((achieved_blocks_per_sm * candidate.num_warps as f64) / max_warps_per_sm).min(1.0);

        OccupancyEstimate { est_regs_per_thread, achieved_blocks_per_sm, achieved_occupancy }
    }
}

/// See `Autotuner::estimate_occupancy`'s doc comment.
#[derive(Debug, Clone, Copy)]
pub struct OccupancyEstimate {
    pub est_regs_per_thread: f64,
    pub achieved_blocks_per_sm: f64,
    pub achieved_occupancy: f64,
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autotuner_candidate_generation() {
        let candidates = Autotuner::generate_candidates(1024, 1024, 1024, Precision::F16);
        assert!(!candidates.is_empty());
        for cand in &candidates {
            assert!(cand.num_stages >= 2);
            assert!(cand.num_warps >= 4);
        }
    }

    #[test]
    fn test_autotuner_selection() {
        let hw = HardwareProfile::default();
        let config = Autotuner::autotune(2048, 2048, 2048, &hw, Precision::F16);
        assert!(config.cta_m >= 64);
        assert!(config.num_stages >= 2);
        assert!(config.num_warps >= 4);
    }

    fn hw_named(gpu: &str) -> HardwareProfile {
        HardwareProfile { gpu_name: gpu.to_string(), ..HardwareProfile::default() }
    }

    #[test]
    fn split_candidates_add_the_large_tiles_that_fit() {
        let f16 = Precision::F16;
        let unsplit = Autotuner::generate_candidates(512, 512, 16384, f16);
        let split = Autotuner::generate_split_candidates(512, 512, 16384, f16);
        assert!(unsplit.iter().all(|c| split.contains(c)), "the unsplit bucket is kept");
        let big = |c: &AutotuneCandidate| (c.cta_m, c.cta_n, c.cta_k, c.warps_m, c.warps_n) == (128, 128, 32, 4, 2);
        assert!(!unsplit.iter().any(big) && split.iter().any(big), "the tile 512 x 512 x 65536 in 4 splits measured fastest");
        let mut dedup = split.clone();
        dedup.dedup();
        assert_eq!(dedup.len(), split.len(), "no candidate twice");
        // Never a tile larger than the matrix.
        assert!(Autotuner::generate_split_candidates(512, 128, 4096, f16).iter().all(|c| c.cta_n <= 128));
        // A large square's own bucket already holds them.
        assert_eq!(Autotuner::generate_split_candidates(4096, 4096, 1024, f16), Autotuner::generate_candidates(4096, 4096, 1024, f16));
    }

    #[test]
    fn persist_key_separates_shape_precision_and_gpu() {
        let a = hw_named("NVIDIA GeForce RTX 4070 Ti SUPER");
        let b = hw_named("NVIDIA GeForce RTX 4090");

        // A profile copied between machines must not serve one card's tile
        // choice to a different card.
        assert_ne!(
            persist_key(4096, 4096, 4096, Precision::F16.tag(), &a),
            persist_key(4096, 4096, 4096, Precision::F16.tag(), &b)
        );
        // Precision namespaces are distinct, so a future FP8 path cannot
        // silently inherit F16's measured tile.
        assert_ne!(
            persist_key(4096, 4096, 4096, Precision::F16.tag(), &a),
            persist_key(4096, 4096, 4096, Precision::Fp8.tag(), &a)
        );
        // So are split counts; an unsplit kernel keeps the key it always had.
        assert_ne!(
            persist_key(512, 512, 65536, &split_tag(Precision::F16, 4), &a),
            persist_key(512, 512, 65536, &split_tag(Precision::F16, 1), &a)
        );
        assert_eq!(
            persist_key(512, 512, 65536, &split_tag(Precision::F16, 1), &a),
            "AUTOTUNE_F16_512x512x65536_NVIDIAGeForceRTX4070TiSUPER"
        );
        assert_eq!(
            persist_key(512, 512, 65536, &split_tag(Precision::F16, 4), &a),
            "AUTOTUNE_F16_SPLITK4_512x512x65536_NVIDIAGeForceRTX4070TiSUPER"
        );
        // Shape is part of the key, and non-square shapes stay distinct from
        // their transpose.
        assert_ne!(
            persist_key(1, 4096, 4096, Precision::F16.tag(), &a),
            persist_key(4096, 4096, 1, Precision::F16.tag(), &a)
        );
        // The key must survive as a `KEY=VALUE` line the sentinel's parser
        // can skip over, so it may not contain '='.
        assert!(!persist_key(4096, 4096, 4096, Precision::F16.tag(), &a).contains('='));
    }

    #[test]
    fn persisted_values_round_trip_and_reject_garbage() {
        let cand = AutotuneCandidate {
            cta_m: 128, cta_n: 64, cta_k: 32,
            warps_m: 2, warps_n: 2, num_stages: 3, num_warps: 4,
        };
        let encoded = format!(
            "{},{},{},{},{},{}",
            cand.cta_m, cand.cta_n, cand.cta_k, cand.warps_m, cand.warps_n, cand.num_stages
        );
        assert_eq!(parse_persisted_value(&encoded), Some(cand));

        // A truncated or corrupted entry must read as "no cached result", not
        // as a partially-populated tile the emitter would then try to lower.
        assert_eq!(parse_persisted_value("128,64,32"), None);
        assert_eq!(parse_persisted_value(""), None);
        assert_eq!(parse_persisted_value("not,a,tile,at,all,here"), None);
    }

    #[test]
    fn forced_config_short_circuits_and_always_clears() {
        let hw = HardwareProfile::default();
        let forced = AutotuneCandidate {
            cta_m: 16, cta_n: 32, cta_k: 32,
            warps_m: 1, warps_n: 1, num_stages: 4, num_warps: 1,
        };

        // Inside the scope, tuning returns exactly the pinned candidate -
        // this is what stops the empirical tuner recursing into itself when
        // it compiles a candidate through the real emitter.
        let got = with_forced_config(&forced, || {
            Autotuner::autotune(4096, 4096, 4096, &hw, Precision::F16)
        });
        assert_eq!((got.cta_m, got.cta_n, got.cta_k), (16, 32, 32));

        // ...and outside it, normal selection resumes.
        assert!(forced_config().is_none());

        // A panic in codegen must not leave the process pinned to one tile.
        let _ = std::panic::catch_unwind(|| {
            with_forced_config(&forced, || panic!("codegen exploded"));
        });
        assert!(forced_config().is_none(), "forced config leaked after a panic");
    }
}