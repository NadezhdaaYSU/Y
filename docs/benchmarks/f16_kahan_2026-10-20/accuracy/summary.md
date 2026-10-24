
### 1x4096x4096

| Y | vs | mixed rel_F / max_norm | same | cancellation | wins rel_F / max_norm | farther outputs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Y_f32 | cuBLAS | 1.7-1.9 / 0.71-2.5 | 0.93-0.95 / 0.82-0.93 | 1.8-2.9 / 1.8-2 | 7/9 / 6/9 | 20.7-47.7% |
| Y_f32 | SGEMM | 1.7-1.9 / 1.1-2.2 | 0.93-0.95 / 0.81-0.93 | 1.9-2.9 / 1.3-2.4 | 7/9 / 7/9 | 20.4-49.0% |
| Y_kahan | cuBLAS | 5.6-7.6 / 5.6-12 | 2.8-3 / 5.3-6.6 | 7.3-8.9 / 11-13 | 9/9 / 9/9 | 6.2-20.6% |
| Y_kahan | SGEMM | 5.6-7.5 / 7.3-10 | 2.8-3 / 5.3-6.6 | 7.5-8.7 / 9.1-13 | 9/9 / 9/9 | 6.8-21.5% |

Y_kahan: farther than its f32 twin on 14.3-23.6% of outputs; same-sign rel_F 7.6e-08-8.2e-08 (twin 2.2e-07-2.6e-07), mean_norm 7.3e-08-7.8e-08 (negative)

### 8x4096x4096

| Y | vs | mixed rel_F / max_norm | same | cancellation | wins rel_F / max_norm | farther outputs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Y_f32 | cuBLAS | 22-23 / 9.7-14 | 193-204 / 50 | 18 / 8.2-11 | 9/9 / 9/9 | 0.0-2.3% |
| Y_f32 | SGEMM | 1.1-1.3 / 0.63-0.81 | 0.48-0.5 / 0.45-0.47 | 1.8-2.1 / 0.84-1.6 | 7/9 / 1/9 | 27.1-64.7% |
| Y_kahan | cuBLAS | 73-89 / 78-95 | 584-616 / 328-374 | 55-71 / 54-78 | 9/9 / 9/9 | 0.0-1.1% |
| Y_kahan | SGEMM | 3.8-5.1 / 4.8-5.9 | 1.4-1.5 / 3-3.6 | 6.4-7 / 7.6-8 | 9/9 / 9/9 | 9.1-34.0% |

Y_kahan: farther than its f32 twin on 14.6-23.3% of outputs; same-sign rel_F 7.7e-08-8.1e-08 (twin 2.2e-07-2.6e-07), mean_norm 7.2e-08-7.6e-08 (negative)

### 32x4096x4096

| Y | vs | mixed rel_F / max_norm | same | cancellation | wins rel_F / max_norm | farther outputs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Y_f32 | cuBLAS | 22-23 / 9.7-15 | 194-201 / 45-48 | 18 / 10-12 | 9/9 / 9/9 | 0.0-2.4% |
| Y_f32 | SGEMM | 1.2-1.3 / 0.65-0.83 | 0.5-0.51 / 0.48-0.53 | 1.8-2.2 / 1.4 | 7/9 / 2/9 | 26.0-63.9% |
| Y_kahan | cuBLAS | 73-89 / 72-94 | 586-617 / 329-378 | 55-71 / 60-75 | 9/9 / 9/9 | 0.0-1.2% |
| Y_kahan | SGEMM | 4-5.3 / 3.9-5.9 | 1.5-1.6 / 3.5-4.2 | 6.7-7.1 / 8.4-8.5 | 9/9 / 9/9 | 8.8-31.7% |

Y_kahan: farther than its f32 twin on 14.7-23.2% of outputs; same-sign rel_F 7.7e-08-8.2e-08 (twin 2.3e-07-2.6e-07), mean_norm 7.2e-08-7.7e-08 (negative)

### 512x512x4096

| Y | vs | mixed rel_F / max_norm | same | cancellation | wins rel_F / max_norm | farther outputs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Y_f32 | cuBLAS | 11 / 4.9-8.4 | 95-99 / 21-23 | 2-9.3 / 2.4-4.9 | 9/9 / 9/9 | 0.0-31.9% |
| Y_f32 | SGEMM | 1.6-1.9 / 1-1.5 | 0.96-0.97 / 0.94-0.95 | 2.6-3 / 2.3-2.9 | 7/9 / 7/9 | 20.3-46.0% |
| Y_f32_s2 | cuBLAS | 15-16 / 9.1-12 | 165-168 / 37-42 | 2-9.3 / 2.2-6 | 9/9 / 9/9 | 0.0-28.4% |
| Y_f32_s2 | SGEMM | 2.2-2.6 / 1.9-2.4 | 1.6-1.7 / 1.5-1.9 | 2.6-3 / 2.2-3.6 | 9/9 / 9/9 | 19.6-27.8% |
| Y_kahan | cuBLAS | 39-49 / 37-48 | 289-304 / 170-196 | 9.6-33 / 25-33 | 9/9 / 9/9 | 0.0-9.9% |
| Y_kahan | SGEMM | 5.8-8.3 / 7.6-9.8 | 2.8-3.1 / 7.1-8.8 | 11-12 / 20-24 | 9/9 / 9/9 | 5.6-18.0% |
| Y_kahan_s2 | cuBLAS | 37-46 / 29-42 | 275-289 / 123-146 | 8.3-31 / 18-30 | 9/9 / 9/9 | 0.0-8.6% |
| Y_kahan_s2 | SGEMM | 5.5-7.7 / 6.1-8.8 | 2.7-2.9 / 5.1-6.6 | 10-11 / 17-18 | 9/9 / 9/9 | 5.9-17.8% |

Y_kahan: farther than its f32 twin on 12.5-19.8% of outputs; same-sign rel_F 7.7e-08-8.1e-08 (twin 2.2e-07-2.6e-07), mean_norm 7.2e-08-7.7e-08 (negative)

Y_kahan_s2: farther than its f32 twin on 14.7-22.8% of outputs; same-sign rel_F 8.1e-08-8.6e-08 (twin 1.3e-07-1.5e-07), mean_norm 7.2e-08-7.7e-08 (negative)

### 512x512x65536

| Y | vs | mixed rel_F / max_norm | same | cancellation | wins rel_F / max_norm | farther outputs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Y_f32_s16 | cuBLAS | 42-43 / 36-38 | 852-858 / 222-230 | 2.9-22 / 4-15 | 9/9 / 9/9 | 0.0-21.9% |
| Y_f32_s16 | SGEMM | 2.2-2.6 / 2.2-2.6 | 1.2-1.3 / 1.5-1.7 | 2.7 / 2.7-2.8 | 9/9 / 9/9 | 21.3-34.8% |
| Y_f32_s4 | cuBLAS | 22-23 / 13-18 | 388-404 / 91 | 1.5-12 / 1.4-6.1 | 9/9 / 9/9 | 0.0-35.7% |
| Y_f32_s4 | SGEMM | 1.2-1.4 / 0.74-1.2 | 0.58 / 0.62-0.68 | 1.4-1.5 / 0.94-1.1 | 7/9 / 4/9 | 35.9-60.1% |
| Y_kahan_s16 | cuBLAS | 113-128 / 58-81 | 980-1007 / 262-311 | 12-92 / 14-51 | 9/9 / 9/9 | 0.0-5.4% |
| Y_kahan_s16 | SGEMM | 6-7.8 / 3.5-5.4 | 1.4-1.5 / 1.8-2.3 | 11 / 9.4 | 9/9 / 9/9 | 5.3-31.2% |
| Y_kahan_s4 | cuBLAS | 138-166 / 107-113 | 1102-1178 / 403-490 | 16-115 / 20-71 | 9/9 / 9/9 | 0.0-4.0% |
| Y_kahan_s4 | SGEMM | 7.3-10 / 6.4-7.5 | 1.6-1.8 / 2.7-3.6 | 14 / 13-14 | 9/9 / 9/9 | 4.2-29.2% |

Y_kahan_s16: farther than its f32 twin on 13.8-27.2% of outputs; same-sign rel_F 9.3e-08-1e-07 (twin 1.1e-07-1.2e-07), mean_norm 7.2e-08-7.7e-08 (negative)

Y_kahan_s4: farther than its f32 twin on 5.9-16.7% of outputs; same-sign rel_F 8.3e-08-8.6e-08 (twin 2.3e-07-2.6e-07), mean_norm 7.2e-08-7.7e-08 (negative)
