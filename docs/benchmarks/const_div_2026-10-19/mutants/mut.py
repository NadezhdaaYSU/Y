#!/usr/bin/env python3
"""mut.py <orig> <out> <name>: one mutant of the constant-division pass."""
import sys
src = open(sys.argv[1]).read()
name = sys.argv[3]
M = {
 "M1_shift_off_by_one": ('format!("shr.{ty} {d}, {n}, {};", c.trailing_zeros())', 'format!("shr.{ty} {d}, {n}, {};", c.trailing_zeros() + 1)'),
 "M2_mask_is_c": ('format!("and.{bits} {d}, {n}, {};", c - 1)', 'format!("and.{bits} {d}, {n}, {};", c)'),
 "M3_no_upper_bound": ("        if m * c as u128 - p <= 1u128 << l {\n            return Some((m as u32, l));", "        if true {\n            return Some((m as u32, l));"),
 "M4_rem_aliasing_allowed": ("        if d == n {\n            return None;\n        }\n", ""),
 "M5_guard_on_first_only": ('Some(instrs.into_iter().map(|i| format!("{indent}{guard}{i}")).collect())',
                            'Some(instrs.into_iter().enumerate().map(|(k, i)| if k == 0 { format!("{indent}{guard}{i}") } else { format!("{indent}{i}") }).collect())'),
 "M6_signed_too": ('        "div.u32" => (true, false),', '        "div.u32" | "div.s32" => (true, false),'),
 "M7_pass_not_applied": ("        self.ptx_buffer = strength_reduce_constant_division(&self.ptx_buffer);\n", ""),
 "M8_u64_mask_b32": ('let (bits, ty) = if wide { ("b64", "u64") } else { ("b32", "u32") };', 'let (bits, ty) = if wide { ("b32", "u64") } else { ("b32", "u32") };'),
 "M9_div_by_one_is_zero": ('format!("mov.{bits} {d}, {n};")', 'format!("mov.{bits} {d}, 0;")'),
 "M10_epilogue_unrolled": ('        writeln!(&mut self.ptx_buffer, "    .pragma \\"nounroll\\";").unwrap();\n', ''),
}
old, new = M[name]
assert src.count(old) == 1, (name, src.count(old))
open(sys.argv[2], "w").write(src.replace(old, new))
