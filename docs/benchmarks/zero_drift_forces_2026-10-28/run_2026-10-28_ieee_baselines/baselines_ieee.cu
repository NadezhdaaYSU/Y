// E8 baselines: the arithmetic of lj.ysu, written in CUDA C and compiled with
// -fmad=false so that no multiply-add is fused (Y fuses none). Only the
// accumulation differs: f32 atomics, f64 atomics, and unchecked 64-bit fixed
// point (Q32.32, as AMBER's SPFP does it; no range or term checks).
#define PAIR                                                                   \
    int p = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (p >= np) return;                                                       \
    int i = pi[p], j = pj[p];                                                  \
    float dx = pos[3 * i] - pos[3 * j];                                        \
    float dy = pos[3 * i + 1] - pos[3 * j + 1];                                \
    float dz = pos[3 * i + 2] - pos[3 * j + 2];                                \
    float r2 = dx * dx + dy * dy + dz * dz;                                    \
    float s2 = 1.0f / r2;                                                      \
    float s6 = s2 * s2 * s2;                                                   \
    float f = 24.0f * s6 * (2.0f * s6 - 1.0f) * s2;                            \
    float t[3] = {f * dx, f * dy, f * dz};

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
