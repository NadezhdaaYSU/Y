#!/usr/bin/env python3
"""mut.py <orig> <out> <name>: one mutant of the rmsnorm_residual emitter."""
import sys
src = open(sys.argv[1]).read()
name = sys.argv[3]
M = {
 "M1_no_tail_predicate": ("""                let pred = if (g + 1) * threads_per_row > vec_groups {""", """                let pred = if false {"""),
 "M2_tail_not_zeroed": ("""                for r in &regs {
                    writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", r).unwrap();
                }
""", ""),
 "M3_weight_offset_by_row": ("""        writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 16;", w_byte_base, group).unwrap();""",
                             """        writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 2;", w_byte_base, elem_base).unwrap();"""),
 "M4_newres_from_x": ("""            let newres_words = self.emit_pack_f16_pairs(&h_vals);
            self.emit_rmsnorm_group_store(&newres_g""", """            let newres_words = x_words.clone();
            self.emit_rmsnorm_group_store(&newres_g"""),
 "M5_guard_wrong_size": ("""        writeln!(&mut self.ptx_buffer, "    setp.ne.u32 {}, {}, {};", wrong_block, ntid, threads_per_row).unwrap();""",
                         """        writeln!(&mut self.ptx_buffer, "    setp.ne.u32 {}, {}, {};", wrong_block, ntid, threads_per_row * 2).unwrap();"""),
 "M6_no_guard": ("""        writeln!(&mut self.ptx_buffer, "    @{} trap;", wrong_block).unwrap();
        let row_offset""", """        let row_offset"""),
 "M7_old_warp_rule": ("""        let num_warps: u32 = if threads_for(1) <= 16 { threads_for(1) } else { threads_for(2).min(32) };""",
                      """        let num_warps: u32 = [8u32, 4, 2, 1].into_iter().find(|&w| vec_groups % (w * WARP) == 0).unwrap_or(1); let _ = threads_for;"""),
 "M8_loop_no_newres": ("""            let newres_words = self.emit_pack_f16_pairs(&h_vals);
            self.emit_rmsnorm_group_store(&newres_g, &offsets[g].0, &newres_words, groups[g].1.as_deref());""",
                       """            let newres_words = self.emit_pack_f16_pairs(&h_vals);
            if unrolled { self.emit_rmsnorm_group_store(&newres_g, &offsets[g].0, &newres_words, groups[g].1.as_deref()); }"""),
 "M9_no_refusal": ("""            if hidden_dim % 8 != 0 {
                // The kernel moves whole 16-byte groups of 8 f16; a row of""", """            if false {
                // The kernel moves whole 16-byte groups of 8 f16; a row of"""),
}
old, new = M[name]
assert src.count(old) == 1, (name, src.count(old))
open(sys.argv[2], "w").write(src.replace(old, new))
