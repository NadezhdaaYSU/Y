// E8 baselines: the arithmetic of lj.ysu, written in CUDA C with the same
// instructions Y emits for it (its fused multiply-adds, and `div.approx.f32`
// for `1.0 / r2`), so that every variant sums the same f32 terms: `lj_terms`
// below must store the bits Y's `lj_terms` stores. Only the accumulation
// differs: f32 atomics, f64 atomics, and unchecked 64-bit fixed point
// (Q32.32, as AMBER's SPFP does it; no range or term checks).
__device__ __forceinline__ float div_approx(float a, float b) {
    float r;
    asm("div.approx.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

#define PAIR                                                                   \
    int p = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (p >= np) return;                                                       \
    int i = pi[p], j = pj[p];                                                  \
    float dx = __fsub_rn(pos[3 * i], pos[3 * j]);                              \
    float dy = __fsub_rn(pos[3 * i + 1], pos[3 * j + 1]);                      \
    float dz = __fsub_rn(pos[3 * i + 2], pos[3 * j + 2]);                      \
    float r2 = __fmaf_rn(dz, dz, __fmaf_rn(dx, dx, __fmul_rn(dy, dy)));         \
    float s2 = div_approx(1.0f, r2);                                           \
    float s6 = __fmul_rn(__fmul_rn(s2, s2), s2);                               \
    float f = __fmul_rn(__fmul_rn(__fmul_rn(24.0f, s6), __fmaf_rn(2.0f, s6, -1.0f)), s2); \
    float t[3] = {__fmul_rn(f, dx), __fmul_rn(f, dy), __fmul_rn(f, dz)};

extern "C" __global__ void lj_terms_cuda(const float* pos, const int* pi, const int* pj, float* T, int np) {
    PAIR
    for (int c = 0; c < 3; c++) T[3 * p + c] = t[c];
}

extern "C" __global__ void lj_f32(const float* pos, const int* pi, const int* pj, float* F, int np) {
    PAIR
    for (int c = 0; c < 3; c++) {
        atomicAdd(&F[3 * i + c], t[c]);
        atomicAdd(&F[3 * j + c], -t[c]);
    }
}

extern "C" __global__ void lj_f64(const float* pos, const int* pi, const int* pj, double* F, int np) {
    PAIR
    for (int c = 0; c < 3; c++) {
        atomicAdd(&F[3 * i + c], (double)t[c]);
        atomicAdd(&F[3 * j + c], -(double)t[c]);
    }
}

extern "C" __global__ void lj_fixed(const float* pos, const int* pi, const int* pj, unsigned long long* F, int np) {
    PAIR
    for (int c = 0; c < 3; c++) {
        long long q = __float2ll_rn(t[c] * 4294967296.0f);
        atomicAdd(&F[3 * i + c], (unsigned long long)q);
        atomicAdd(&F[3 * j + c], (unsigned long long)(-q));
    }
}

// The fixed-point totals back to f32, one thread per total.
extern "C" __global__ void fixed_to_f32(const long long* F, float* out, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) out[k] = (float)((double)F[k] * 2.3283064365386963e-10);
}
