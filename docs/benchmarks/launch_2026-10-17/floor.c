// The driver's own cost of one cuLaunchKernel of Y's vec_add, from C:
// 15 rounds of 200 launches timed on the host clock, a sync between rounds.
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define CK(x) do { CUresult r = (x); if (r) { printf("%s -> %d\n", #x, r); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static int cmp(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); static char ptx[1 << 20]; size_t n = fread(ptx, 1, sizeof ptx - 1, f); ptx[n] = 0;
    CUdevice d; CUcontext c; CUmodule m; CUfunction fn; CUstream s;
    CK(cuInit(0)); CK(cuDeviceGet(&d, 0)); CK(cuDevicePrimaryCtxRetain(&c, d)); CK(cuCtxSetCurrent(c));
    CK(cuModuleLoadData(&m, ptx)); CK(cuModuleGetFunction(&fn, m, "vec_add"));
    CK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
    int N = 100000; CUdeviceptr a, b, o; CK(cuMemAlloc(&a, N * 4)); CK(cuMemAlloc(&b, N * 4)); CK(cuMemAlloc(&o, N * 4));
    void *args[] = {&a, &b, &o, &N};
    unsigned gx = (N / 4 + 127) / 128;
    double r[15];
    for (int round = 0; round < 17; round++) {
        CK(cuCtxSynchronize());
        double t0 = now();
        for (int i = 0; i < 200; i++) CK(cuLaunchKernel(fn, gx, 1, 1, 128, 1, 1, 0, s, args, NULL));
        double t1 = now();
        if (round >= 2) r[round - 2] = (t1 - t0) / 200;
    }
    CK(cuCtxSynchronize());
    qsort(r, 15, sizeof r[0], cmp);
    printf("C cuLaunchKernel: min %.2f median %.2f us per call\n", r[0], r[7]);
    return 0;
}
