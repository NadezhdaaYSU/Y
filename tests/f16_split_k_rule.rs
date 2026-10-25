//! The split-K rule (`Autotuner::split_k_choice`), which a kernel named
//! `<name>_splitk` takes: split only when the unsplit tile leaves SMs without
//! a CTA and the GEMM is not bound to DRAM, on the split count and tile that
//! give the most CTAs, from no fewer than unsplit to one an SM, each split at
//! least `SPLIT_K_MIN_DEPTH` deep; a tie keeps the smaller split count.
//!
//! Every case gives the unsplit tile and the SM count explicitly, so no
//! verdict depends on a profile. Each bound is checked on both sides of it.

use y::autotuner::{validate_warp_tiling, Autotuner, SPLIT_K_COUNTS, SPLIT_K_MIN_DEPTH, SPLIT_K_MIN_INTENSITY, SPLIT_K_TILES};
use y::ptx_emitter::{CtaTileConfig, PtxEmitter};

fn tile(cta_m: u32, cta_n: u32, cta_k: u32, warps_m: u32, warps_n: u32, num_stages: u32) -> CtaTileConfig {
    CtaTileConfig { cta_m, cta_n, cta_k, warps_m, warps_n, mma_m: 16, mma_n: 16, mma_k: 16, num_stages, num_warps: warps_m * warps_n }
}

/// The split count and, when split, the tile as "CMxCN WMxWN sST".
fn choose(m: u32, n: u32, k: u32, unsplit: &CtaTileConfig, sms: u32) -> (u32, String) {
    let plan = Autotuner::split_k_choice(m, n, k, unsplit, sms);
    let t = plan.tile.as_ref().map(|t| format!("{}x{} {}x{} s{}", t.cta_m, t.cta_n, t.warps_m, t.warps_n, t.num_stages)).unwrap_or_default();
    assert_eq!(plan.splits == 1, plan.tile.is_none(), "{m}x{n}x{k}: a split without a tile, or a tile without a split");
    assert!(!plan.reason.is_empty());
    (plan.splits, t)
}

const T64: fn() -> CtaTileConfig = || tile(64, 64, 32, 2, 2, 4);
const T32: fn() -> CtaTileConfig = || tile(32, 32, 32, 2, 1, 4);

#[test]
fn the_rule_splits_where_the_unsplit_grid_leaves_sms_without_a_cta() {
    // The shapes of the sweep the rule splits, on the tiles the measured
    // profile gives them (docs/benchmarks/f16_auto_splitk_2026-10-25/census/).
    // 512 x 512: 64 CTAs unsplit; 64x128 in 2 is 64 too, before 128x128 in 4.
    assert_eq!(choose(512, 512, 16384, &T64(), 66), (2, "64x128 4x2 s4".into()));
    assert_eq!(choose(512, 512, 65536, &T64(), 66), (2, "64x128 4x2 s4".into()));
    // 256 x 256 on 32x32 tiles (64 CTAs): every 2-way split has fewer.
    assert_eq!(choose(256, 256, 16384, &T32(), 66), (4, "64x64 4x2 s4".into()));
}

#[test]
fn the_rule_leaves_full_grids_short_splits_dram_bound_and_decode_gemms_unsplit() {
    // Splits of K = 4096 are at most 2048 deep.
    assert_eq!(choose(512, 512, 4096, &T64(), 66).0, 1);
    // 64 rows: about 62 flops a byte, bound to DRAM.
    for k in [4096, 11008, 14336] {
        let plan = Autotuner::split_k_choice(64, 4096, k, &T64(), 66);
        assert!(plan.splits == 1 && plan.reason.contains("bound to DRAM"), "64x4096x{k}: {plan:?}");
    }
    // Decode: at most M flops a byte.
    let decode = tile(32, 64, 64, 2, 2, 3);
    for (m, n, k) in [(1, 4096, 14336), (8, 4096, 4096), (32, 4096, 14336), (32, 2048, 16384), (8, 1024, 65536)] {
        let plan = Autotuner::split_k_choice(m, n, k, &decode, 66);
        assert!(plan.splits == 1 && plan.reason.contains("bound to DRAM"), "{m}x{n}x{k}: {plan:?}");
    }
    // 66 or more CTAs unsplit.
    assert_eq!(choose(1024, 1024, 16384, &tile(64, 128, 32, 2, 2, 3), 66).0, 1);
    assert_eq!(choose(128, 4096, 14336, &T64(), 66).0, 1);
    assert_eq!(choose(8192, 8192, 8192, &tile(128, 128, 32, 4, 2, 3), 66).0, 1);
    for (m, n, k) in [(1024, 4096, 4096), (2048, 11008, 4096), (256, 14336, 4096), (512, 4096, 14336), (64, 14336, 4096)] {
        assert_eq!(choose(m, n, k, &T64(), 66).0, 1, "{m}x{n}x{k}");
    }
    // No SM count, no split.
    assert_eq!(choose(512, 512, 65536, &T64(), 0).0, 1);
    // Every split shallow enough has fewer CTAs than unsplit: 256 x 256 in
    // 32x32 tiles is 64 CTAs, and 2 splits of 4096 have at most 32.
    assert_eq!(choose(256, 256, 8192, &T32(), 66).0, 1);
}

#[test]
fn each_bound_of_the_rule_on_both_sides() {
    // Unsplit CTAs below the SM count: 64 on 65 SMs splits, on 64 does not.
    assert_eq!(choose(512, 512, 16384, &T64(), 65), (2, "64x128 4x2 s4".into()));
    assert_eq!(choose(512, 512, 16384, &T64(), 64).0, 1);
    // Each split at least SPLIT_K_MIN_DEPTH deep: 256 x 256 splits in 4 at
    // K = 4 x 4096; 32 less, 4 splits are 4088 deep and 2 have too few CTAs.
    assert_eq!(SPLIT_K_MIN_DEPTH, 4096);
    assert_eq!(choose(256, 256, 4 * SPLIT_K_MIN_DEPTH, &T32(), 66), (4, "64x64 4x2 s4".into()));
    assert_eq!(choose(256, 256, 4 * SPLIT_K_MIN_DEPTH - 32, &T32(), 66).0, 1);
    // At most one CTA an SM: on 128 SMs the 128-CTA splits are taken, the
    // smaller S first (64x64 in 2); on 127, the 64-CTA ones.
    assert_eq!(choose(512, 512, 65536, &T64(), 128), (2, "64x64 4x2 s4".into()));
    assert_eq!(choose(512, 512, 65536, &T64(), 127), (2, "64x128 4x2 s4".into()));
    // Most CTAs before the smaller S: 256 x 256 on 64x64 is 16 CTAs; in 2 at
    // most 32, in 4 64.
    assert_eq!(choose(256, 256, 16384, &T64(), 66), (4, "64x64 4x2 s4".into()));
    // Intensity: 62 flops a byte (64 x 4096 x 11008) is below the bound, 126
    // (256 x 256 x 16384) above it.
    assert_eq!(SPLIT_K_MIN_INTENSITY, 96);
    assert_eq!(choose(64, 4096, 11008, &T64(), 66).0, 1);
    assert_eq!(choose(256, 256, 16384, &T32(), 66).0, 4);
}

#[test]
fn the_split_counts_and_the_shape_check() {
    assert_eq!(SPLIT_K_COUNTS, [2, 4, 8]);
    // K = 3 x 8192: 2 splits of 12288.
    assert_eq!(choose(512, 512, 3 * 8192, &T64(), 66), (2, "64x128 4x2 s4".into()));
    // N not a multiple of 4: the reduction sums float4s, so no split.
    assert_eq!(choose(512, 510, 65536, &T64(), 66).0, 1);
    // K = 65540: no split is a multiple of 8 deep.
    assert_eq!(choose(512, 512, 65540, &T64(), 66).0, 1);
}

#[test]
fn every_choice_over_a_grid_of_shapes_keeps_the_rules_bounds() {
    let mut split = 0;
    for m in [64, 96, 112, 128, 192, 256, 384, 512, 1024] {
        for n in [64, 96, 112, 128, 192, 256, 384, 512, 1024] {
            for k in [8192, 16384, 32768, 65536] {
                for (unsplit, sms) in [(T64(), 66), (T32(), 66), (T64(), 128), (tile(128, 128, 32, 4, 2, 3), 66)] {
                    let plan = Autotuner::split_k_choice(m, n, k, &unsplit, sms);
                    let Some(t) = plan.tile else { continue };
                    split += 1;
                    let (s, t1) = (plan.splits, m.div_ceil(unsplit.cta_m) * n.div_ceil(unsplit.cta_n));
                    let ctas = m.div_ceil(t.cta_m) * n.div_ceil(t.cta_n) * s;
                    let at = format!("{m}x{n}x{k} on {}x{}, {sms} SMs: S={s} {}x{}", unsplit.cta_m, unsplit.cta_n, t.cta_m, t.cta_n);
                    assert!(t1 < sms && t1 <= ctas && ctas <= sms, "{at}: {ctas} CTAs from {t1}");
                    assert!(t.cta_m <= m && t.cta_n <= n && SPLIT_K_TILES.contains(&t), "{at}");
                    assert!(SPLIT_K_COUNTS.contains(&s) && k / s >= SPLIT_K_MIN_DEPTH && PtxEmitter::gemm_split_k_shape(m, n, k, s).is_ok(), "{at}");
                    let (m, n, k) = (m as u128, n as u128, k as u128);
                    assert!(2 * m * n * k >= SPLIT_K_MIN_INTENSITY as u128 * (2 * m * k + 2 * k * n + 4 * m * n), "{at}");
                }
            }
        }
    }
    assert!(split > 50, "the grid split only {split} times");
}

#[test]
fn every_split_tile_tiles_and_fits_its_block_sums() {
    for t in SPLIT_K_TILES {
        validate_warp_tiling(t.cta_m, t.cta_n, t.cta_k, t.warps_m, t.warps_n, 16, 16, 16).unwrap();
        assert_eq!(t.num_warps, t.warps_m * t.warps_n, "{t:?}");
        let accumulators = (t.cta_m / t.warps_m) * (t.cta_n / t.warps_n) / 32;
        assert!(PtxEmitter::f16_block_sums_fit(accumulators, t.num_warps * 32), "{t:?}: its block sums do not fit");
    }
    let areas: Vec<u32> = SPLIT_K_TILES.iter().map(|t| t.cta_m * t.cta_n).collect();
    assert!(areas.windows(2).all(|w| w[0] > w[1]), "the tiles are not largest first, each a distinct size");
}
