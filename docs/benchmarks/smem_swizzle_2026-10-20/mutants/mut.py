#!/usr/bin/env python3
"""mut.py <orig> <out> <name>: one mutant of the swizzled GEMM tiles."""
import sys
src = open(sys.argv[1]).read()
name = sys.argv[3]
SYNC = '''        let slc = self.emit_smem_stored_col(&smem, &lr, &lc_chunk, &lc);
        let sidx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", sidx, lr, smem.stride, slc).unwrap();
        let sbyte = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    shl.b32 {}, {}, 1;", sbyte, sidx).unwrap();
        let saddr = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", saddr, smem_base, sbyte).unwrap();'''
CHUNK = SYNC.replace("smem_base, sbyte", "smem_stage_base, sbyte")
HOIST = '''        let slc = self.emit_smem_stored_col(&smem, &lr, &lc_chunk, &lc);
        let sidx = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mad.lo.u32 {}, {}, {}, {};", sidx, lr, smem.stride, slc).unwrap();
        let smem_thread = self.alloc_reg32();'''
unslc = lambda s: s.replace("let slc = self.emit_smem_stored_col(&smem, &lr, &lc_chunk, &lc);", "let slc = lc.clone();")
M = {
 "M1_copies_unswizzled": ("    fn emit_smem_stored_col(&mut self, smem: &SmemTile, row: &str, chunk: &str, col: &str) -> String {\n",
                          "    fn emit_smem_stored_col(&mut self, smem: &SmemTile, row: &str, chunk: &str, col: &str) -> String {\n        if smem.swizzle.is_some() {\n            return col.to_string();\n        }\n"),
 "M2_a_step_added_not_xored": ('writeln!(&mut self.ptx_buffer, "    xor.b32 {}, {}, {};", r, a_xor, kk / 8).unwrap();',
                               'writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", r, a_xor, kk / 8).unwrap();'),
 "M3_b_block_ored_not_xored": ('writeln!(&mut self.ptx_buffer, "    xor.b32 {}, {}, {};", stored, chunk, b_xor).unwrap();',
                               'writeln!(&mut self.ptx_buffer, "    or.b32 {}, {}, {};", stored, chunk, b_xor).unwrap();'),
 "M4_mask_full_width": ("Some(SmemTile { stride: cols, swizzle: Some(((8 / width).trailing_zeros(), width - 1)) })",
                        "Some(SmemTile { stride: cols, swizzle: Some(((8 / width).trailing_zeros(), width)) })"),
 "M5_shift_zero": ("Some(SmemTile { stride: cols, swizzle: Some(((8 / width).trailing_zeros(), width - 1)) })",
                   "Some(SmemTile { stride: cols, swizzle: Some((0, width - 1)) })"),
 "M6_no_period_check": ("        if chunks > 1 && (threads_per_cta / chunks_per_row) % smem.period() != 0 {\n            return None;\n        }\n", ""),
 "M7_lane_half_dropped": ('writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, 3;", half, col_bit_a).unwrap();',
                          'writeln!(&mut self.ptx_buffer, "    shr.u32 {}, {}, 4;", half, col_bit_a).unwrap();'),
 "M8_sync_copy_unswizzled": (SYNC, unslc(SYNC)),
 "M9_chunk_copy_unswizzled": (CHUNK, unslc(CHUNK)),
 "M10_hoisted_copy_unswizzled": (HOIST, unslc(HOIST)),
 "M11_padded_by_default": ('std::env::var("Y_SMEM_SWIZZLE").map_or(true, |v| v != "0")', 'std::env::var("Y_SMEM_SWIZZLE").map_or(false, |v| v != "0")'),
 "M12_b_lane_term_from_a": ("layout.b_xor = self.emit_smem_stored_chunk(&b, row_off, &half);", "layout.b_xor = self.emit_smem_stored_chunk(&a, row_off, &half);"),
 "M13_a_lane_term_from_b": ("layout.a_xor = self.emit_smem_stored_chunk(&a, row_off, &half);", "layout.a_xor = self.emit_smem_stored_chunk(&b, row_off, &half);"),
}
old, new = M[name]
assert src.count(old) == 1, (name, src.count(old))
open(sys.argv[2], "w").write(src.replace(old, new))
