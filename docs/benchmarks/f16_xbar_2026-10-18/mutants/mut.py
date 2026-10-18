#!/usr/bin/env python3
"""mut.py <orig> <out> <name>: write one mutant of the K-loop schedule."""
import sys
src = open(sys.argv[1]).read()
name = sys.argv[3]
M = {
 "M1_wait_one_fewer": ("""                        self.emit_cp_async_wait(n_stages - 2);
                        writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
                        let next_tile_r""", """                        self.emit_cp_async_wait(n_stages - 1);
                        writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
                        let next_tile_r"""),
 "M2_no_mid_barrier": ("""                        self.emit_cp_async_wait(n_stages - 2);
                        writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
                        let next_tile_r""", """                        self.emit_cp_async_wait(n_stages - 2);
                        let next_tile_r"""),
 "M3_loads_above_wait": ("""                        self.emit_cp_async_wait(n_stages - 2);
                        writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
                        let next_tile_r = self.alloc_reg32();""", """                        let next_tile_r = self.alloc_reg32();"""),
 "M4_next_from_k_iter": ("""                        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 1;", next_tile_r, k_iter).unwrap();""",
                         """                        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, 0;", next_tile_r, k_iter).unwrap();"""),
 "M5_no_commit_before_wait": ("""                        if !interleave.is_empty() {
                            self.emit_cp_async_commit();
                        }
                        // Tile `k_iter+1`""", """                        // Tile `k_iter+1`"""),
 "M6_two_stages_too": ("let xbar = n_stages >= 3 && k_substeps", "let xbar = n_stages >= 2 && k_substeps"),
 "M7_no_first_fragments": ("""                self.emit_gemm_fragment_loads(
                    &sets[0].0, &sets[0].1, &warp_col0_local, &warp_row0_local, &smem_pipeline_base, &smem_b_region_base,
                    smem_a_stride, smem_b_stride, &row_off, &col_bit_a, 0,
                );
                Some(sets)""", """                Some(sets)"""),
}
old, new = M[name]
assert src.count(old) == 1, (name, src.count(old))
out = src.replace(old, new)
if name == "M3_loads_above_wait":
    # put the wait and barrier back after the next fragments' loads
    anchor = """                            smem_a_stride, smem_b_stride, &row_off, &col_bit_a, 0,
                        );
                    }
                    let lines: &[String]"""
    assert out.count(anchor) == 1
    out = out.replace(anchor, """                            smem_a_stride, smem_b_stride, &row_off, &col_bit_a, 0,
                        );
                        self.emit_cp_async_wait(n_stages - 2);
                        writeln!(&mut self.ptx_buffer, "    bar.sync 0;").unwrap();
                    }
                    let lines: &[String]""")
open(sys.argv[2], "w").write(out)
