The probe that found the bug (2026-10-26), with the main checkout's compiler at 7d6037f and the
re-tuned profile; shfl_i32.ysu, shfl_f64.ysu, run.py (launches shfl_i32 through y_lang.torch_interop).

Before (7d6037f):
  shfl_i32: shfl.sync.bfly.b32 %f0, %r2, %r3, 0x1f, 0xffffffff;  cvt.rzi.s32.f32 %r4, %f0;  st.global.u32
            assembles (ptxas -arch=sm_89); on the GPU every lane stored 0 (expected its xor-1 neighbour's
            t + 1000: 1001, 1000, 1003, ...).
  shfl_f64: shfl.sync.bfly.b32 %f1, %fd0, ...;  cvt.f64.f32  -> ptxas: "Arguments mismatch for instruction 'shfl'".
  warp_reduce_sum / warp_reduce_max of I32 and U32: add.f32 / max.f32 on the integers' bits, then
            cvt.rzi.{s32,u32}.f32; they assemble. Of I64: ptxas errors.
After (this tree):
  shfl_i32: shfl.sync.bfly.b32 %r5, %r2, %r4, ...;  st.global.u32 %r5 -> 1001, 1000, 1003, 1002, ... on the GPU.
  shfl_f64: mov.b64 {%r3, %r4}, %fd0; two shfl.sync.bfly.b32; mov.b64 %fd1, {%r5, %r6} -> assembles.
