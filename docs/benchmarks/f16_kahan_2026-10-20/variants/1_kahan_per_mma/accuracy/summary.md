
### 1x4096x4096

| Y | vs | mixed rel_F / max_norm | same | cancellation | wins rel_F / max_norm | farther outputs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Y_f32 | cuBLAS | 1.7-1.9 / 0.71-2.5 | 0.93-0.95 / 0.82-0.93 | 1.8-2.9 / 1.8-2 | 7/9 / 6/9 | 20.7-47.7% |
| Y_f32 | SGEMM | 1.7-1.9 / 1.1-2.2 | 0.93-0.95 / 0.81-0.93 | 1.9-2.9 / 1.3-2.4 | 7/9 / 7/9 | 20.4-49.0% |
| Y_kahan | cuBLAS | 6-8.6 / 5.3-13 | 2.8-3 / 5.4-6.9 | 9.7-12 / 14-17 | 9/9 / 9/9 | 5.0-20.9% |
| Y_kahan | SGEMM | 6-8.6 / 7.3-12 | 2.8-3 / 5.4-6.9 | 10-12 / 12-16 | 9/9 / 9/9 | 5.3-21.8% |

Y_kahan: farther than its f32 twin on 11.2-19.1% of outputs; same-sign rel_F 7.6e-08-8.2e-08 (twin 2.2e-07-2.6e-07), mean_norm 7.3e-08-7.8e-08 (negative)

### 8x4096x4096

| Y | vs | mixed rel_F / max_norm | same | cancellation | wins rel_F / max_norm | farther outputs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Y_f32 | cuBLAS | 22-23 / 9.7-14 | 193-204 / 50 | 18 / 8.2-11 | 9/9 / 9/9 | 0.0-2.3% |
| Y_f32 | SGEMM | 1.1-1.3 / 0.63-0.81 | 0.48-0.5 / 0.45-0.47 | 1.8-2.1 / 0.84-1.6 | 7/9 / 1/9 | 27.1-64.7% |
| Y_kahan | cuBLAS | 79-101 / 81-96 | 586-618 / 348-404 | 70-95 / 75-108 | 9/9 / 9/9 | 0.0-0.9% |
| Y_kahan | SGEMM | 4.1-5.8 / 5-5.9 | 1.4-1.5 / 3.1-3.9 | 8.3-9.3 / 11 | 9/9 / 9/9 | 7.2-34.0% |

Y_kahan: farther than its f32 twin on 11.5-17.4% of outputs; same-sign rel_F 7.7e-08-8.1e-08 (twin 2.2e-07-2.6e-07), mean_norm 7.2e-08-7.6e-08 (negative)

### 32x4096x4096

| Y | vs | mixed rel_F / max_norm | same | cancellation | wins rel_F / max_norm | farther outputs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Y_f32 | cuBLAS | 22-23 / 9.7-15 | 194-201 / 45-48 | 18 / 10-12 | 9/9 / 9/9 | 0.0-2.4% |
| Y_f32 | SGEMM | 1.2-1.3 / 0.65-0.83 | 0.5-0.51 / 0.48-0.53 | 1.8-2.2 / 1.4 | 7/9 / 2/9 | 26.0-63.9% |
| Y_kahan | cuBLAS | 79-101 / 84-104 | 587-618 / 350-402 | 71-94 / 70-108 | 9/9 / 9/9 | 0.0-0.9% |
| Y_kahan | SGEMM | 4.3-6 / 4.8-6.9 | 1.5-1.6 / 3.8-4.5 | 8.7-9.4 / 9.8-12 | 9/9 / 9/9 | 7.2-31.8% |

Y_kahan: farther than its f32 twin on 11.6-17.1% of outputs; same-sign rel_F 7.7e-08-8.1e-08 (twin 2.3e-07-2.6e-07), mean_norm 7.2e-08-7.7e-08 (negative)

### 512x512x4096

| Y | vs | mixed rel_F / max_norm | same | cancellation | wins rel_F / max_norm | farther outputs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Y_f32 | cuBLAS | 11 / 4.9-8.4 | 95-99 / 21-23 | 2-9.3 / 2.4-4.9 | 9/9 / 9/9 | 0.0-31.9% |
| Y_f32 | SGEMM | 1.6-1.9 / 1-1.5 | 0.96-0.97 / 0.94-0.95 | 2.6-3 / 2.3-2.9 | 7/9 / 7/9 | 20.3-46.0% |
| Y_f32_s2 | cuBLAS | 15-16 / 9.1-12 | 165-168 / 37-42 | 2-9.3 / 2.2-6 | 9/9 / 9/9 | 0.0-28.4% |
| Y_f32_s2 | SGEMM | 2.2-2.6 / 1.9-2.4 | 1.6-1.7 / 1.5-1.9 | 2.6-3 / 2.2-3.6 | 9/9 / 9/9 | 19.6-27.8% |
| Y_kahan | cuBLAS | 40-51 / 39-48 | 289-304 / 173-201 | 11-36 / 29-36 | 9/9 / 9/9 | 0.0-8.9% |
| Y_kahan | SGEMM | 5.9-8.6 / 7.8-9.8 | 2.8-3.1 / 7.2-9.1 | 12-13 / 22-28 | 9/9 / 9/9 | 5.3-18.0% |
| Y_kahan_s2 | cuBLAS | 38-47 / 27-40 | 275-289 / 125-152 | 8.9-33 / 15-30 | 9/9 / 9/9 | 0.0-7.7% |
| Y_kahan_s2 | SGEMM | 5.6-7.9 / 5.7-8.8 | 2.7-2.9 / 5.2-6.9 | 11 / 15-18 | 9/9 / 9/9 | 5.5-17.8% |

Y_kahan: farther than its f32 twin on 11.6-17.3% of outputs; same-sign rel_F 7.7e-08-8.1e-08 (twin 2.2e-07-2.6e-07), mean_norm 7.2e-08-7.7e-08 (negative)

Y_kahan_s2: farther than its f32 twin on 13.3-22.8% of outputs; same-sign rel_F 8.1e-08-8.6e-08 (twin 1.3e-07-1.5e-07), mean_norm 7.2e-08-7.7e-08 (negative)

### 512x512x65536

| Y | vs | mixed rel_F / max_norm | same | cancellation | wins rel_F / max_norm | farther outputs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Y_f32_s16 | cuBLAS | 42-43 / 36-38 | 852-858 / 222-230 | 2.9-22 / 4-15 | 9/9 / 9/9 | 0.0-21.9% |
| Y_f32_s16 | SGEMM | 2.2-2.6 / 2.2-2.6 | 1.2-1.3 / 1.5-1.7 | 2.7 / 2.7-2.8 | 9/9 / 9/9 | 21.3-34.8% |
| Y_f32_s4 | cuBLAS | 22-23 / 13-18 | 388-404 / 91 | 1.5-12 / 1.4-6.1 | 9/9 / 9/9 | 0.0-35.7% |
| Y_f32_s4 | SGEMM | 1.2-1.4 / 0.74-1.2 | 0.58 / 0.62-0.68 | 1.4-1.5 / 0.94-1.1 | 7/9 / 4/9 | 35.9-60.1% |
| Y_kahan_s16 | cuBLAS | 114-130 / 70-78 | 979-1007 / 262-311 | 12-95 / 13-60 | 9/9 / 9/9 | 0.0-5.1% |
| Y_kahan_s16 | SGEMM | 6-7.9 / 4.1-5.2 | 1.4-1.5 / 1.8-2.3 | 11-12 / 9.3-11 | 9/9 / 9/9 | 5.1-31.2% |
| Y_kahan_s4 | cuBLAS | 139-169 / 98-112 | 1101-1179 / 392-484 | 16-121 / 18-83 | 9/9 / 9/9 | 0.0-3.8% |
| Y_kahan_s4 | SGEMM | 7.4-10 / 6.1-7.8 | 1.6-1.8 / 2.7-3.6 | 15 / 13-15 | 9/9 / 9/9 | 4.0-29.2% |

Y_kahan_s16: farther than its f32 twin on 13.2-27.2% of outputs; same-sign rel_F 9.3e-08-1e-07 (twin 1.1e-07-1.2e-07), mean_norm 7.2e-08-7.7e-08 (negative)

Y_kahan_s4: farther than its f32 twin on 5.5-16.7% of outputs; same-sign rel_F 8.3e-08-8.6e-08 (twin 2.3e-07-2.6e-07), mean_norm 7.2e-08-7.7e-08 (negative)
