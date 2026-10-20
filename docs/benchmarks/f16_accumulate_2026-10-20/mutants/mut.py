#!/usr/bin/env python3
"""mut.py <orig> <out> <name>: one mutant of the F16 float32 accumulation."""
import sys
src = open(sys.argv[1]).read()
M = {
 "M1_mma_from_accumulator": ('"    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{{}}}, {{{}}}, {{{}}}, {{{z},{z},{z},{z}}};",\n            t.join(","), a.join(","), b.join(",")',
                             '"    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{{}}}, {{{}}}, {{{}}}, {{{}}};",\n            t.join(","), a.join(","), b.join(","), d.join(",")'),
 "M2_no_adds": ('            writeln!(&mut self.ptx_buffer, "    add.rn.f32 {}, {}, {};", r, r, tr).unwrap();\n        }\n    }\n\n    /// Whether an F16 GEMM',
                '            let _ = (r, tr);\n        }\n    }\n\n    /// Whether an F16 GEMM'),
 "M3_threshold_below_32": ("            Err(_) => m <= 32,", "            Err(_) => m < 32,"),
 "M4_f32_knob_ignored": ('            Ok("f32") => true,', '            Ok("f32") => m <= 32,'),
 "M5_bad_knob_accepted": ('            Ok(other) => panic!("Y_F16_ACCUMULATE must be `f32` or `mma`, not {other:?}"),', '            Ok(_) => m <= 32,'),
 "M6_round_toward_zero": ('            writeln!(&mut self.ptx_buffer, "    add.rn.f32 {}, {}, {};", r, r, tr).unwrap();\n        }\n    }\n\n    /// Whether an F16 GEMM',
                          '            writeln!(&mut self.ptx_buffer, "    add.rz.f32 {}, {}, {};", r, r, tr).unwrap();\n        }\n    }\n\n    /// Whether an F16 GEMM'),
}
old, new = M[sys.argv[3]]
assert src.count(old) == 1, (sys.argv[3], src.count(old))
open(sys.argv[2], "w").write(src.replace(old, new))
