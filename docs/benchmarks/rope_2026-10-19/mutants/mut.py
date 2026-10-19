#!/usr/bin/env python3
"""mut.py <orig> <out> <name>: one mutant of the rope_<head_dim> emitter."""
import sys
src = open(sys.argv[1]).read()
name = sys.argv[3]
M = {
 "M1_no_block_guard": ("""        writeln!(&mut self.ptx_buffer, "    @{} trap;", wrong_block).unwrap();
        // `Rows` is sign-extended""", """        // `Rows` is sign-extended"""),
 "M2_no_negative_guard": ("""        writeln!(&mut self.ptx_buffer, "    @{} trap;", negative).unwrap();\n""", ""),
 "M3_no_short_grid_guard": ("""        writeln!(&mut self.ptx_buffer, "    @{} trap;", short_grid).unwrap();\n""", ""),
 "M4_stores_unpredicated": ("""                n => writeln!(&mut self.ptx_buffer, "    @{} st.global.v{}.u32 [{}], {{{}}};", valid, n, out_addr, out_words.join(",")).unwrap(),""",
                            """                n => writeln!(&mut self.ptx_buffer, "    st.global.v{}.u32 [{}], {{{}}};", n, out_addr, out_words.join(",")).unwrap(),"""),
 "M5_rows_per_cta_halved": ("""        writeln!(&mut self.ptx_buffer, "    mad.wide.u32 {}, {}, {}, {};", row64, cta, rows_per_cta, slot64).unwrap();""",
                            """        writeln!(&mut self.ptx_buffer, "    mad.wide.u32 {}, {}, {}, {};", row64, cta, rows_per_cta / 2, slot64).unwrap();"""),
 "M6_one_warp_a_row": ("""        (head_dim / 4).min(32)""", """        32"""),
 "M7_no_head_dim_refusal": ("""            if head_dim % 64 != 0 {""", """            if false {"""),
 "M8_old_form_unmatched": ("""        if !(kernel.params.len() == 3 || kernel.params.len() == 4)""",
                           """        if kernel.params.len() != 4"""),
 "M9_pair_offset_halved": ("""            writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 4;", pair_byte, pair_i_start).unwrap();""",
                           """            writeln!(&mut self.ptx_buffer, "    mul.wide.u32 {}, {}, 2;", pair_byte, pair_i_start).unwrap();"""),
}
old, new = M[name]
assert src.count(old) == 1, (name, src.count(old))
open(sys.argv[2], "w").write(src.replace(old, new))
