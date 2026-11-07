"""A replica of `f32_gemm_streamk_plan`'s estimate (src/ptx_emitter/f32_simt_gemm.rs):
the tile, the predicted plain / stream-K time ratio, and the grid, per shape,
for 66 SMs and 100 KB of shared memory an SM. Used to pick shapes on both
sides of the bar; the compiler's own header is what the benchmarks read.

usage: plan_model.py MxNxK ..."""
import math, sys
TILES = {"large": (128, 256, 1, [1.0], 3*16*(128+4+256)*4), "mid": (64, 128, 3, [1.0, 1.136], 3*16*(64+4+128)*4), "small": (64, 64, 3, [1.0, 1.077, 1.081], 3*16*(64+4+64)*4)}
def tile(m, n, k, sms=66):
    t = lambda name: math.ceil(m / TILES[name][0]) * math.ceil(n / TILES[name][1])
    large = t("large"); full = math.ceil(large / sms) * sms
    if m >= 128 and large * 10 >= sms * 19 and large * 5 >= full * 4: return "large"
    if t("mid") * 10 >= sms * 9: return "mid"
    return "small"
def ratio(m, n, k, sms=66):
    name = tile(m, n, k); bm, bn, reg_ctas, rates, smem = TILES[name]
    c_max = max(1, min(min(reg_ctas, 102400 // (smem + 1024)), len(rates)))
    time = lambda j: j / rates[j - 1]
    rem = (n % bn != 0 and n % bn <= bn // 2 and not (math.ceil(n / bn) > 50 and not (n % 4 != 0 and n >= 16 and (n % 2 == 0 or bn == 256))))
    rows = (m % bm != 0 and m % bm <= bm // 2 and math.ceil(m / bm) <= 50)
    n_main = n // bn * bn if rem else n; m_main = m // bm * bm if rows else m
    main = math.ceil(m_main / bm) * math.ceil(n_main / bn)
    rbn = {256: 256, 128: 128, 64: 64}[bn]
    row_tiles = math.ceil((m - m_main) / 32) * math.ceil(n_main / rbn) if rows else 0
    rem_bn = 64 if bn == 256 else 32
    rem_tiles = math.ceil(m / (128 if bn == 256 else 64)) * math.ceil((n - n_main) / rem_bn) if rem else 0
    work = main + 0.5 * row_tiles + rem_tiles / 3
    slots = sms * c_max; fullw = math.floor(work / slots); last = work - fullw * slots
    plain = fullw * time(c_max) + (time(min(max(math.ceil(last / sms), 1), c_max)) if last > 1e-9 else 0)
    tiles = math.ceil(m / bm) * math.ceil(n / bn)
    opts = [(tiles / (sms * j) * time(j), sms * j) for j in range(1, c_max + 1) if tiles >= sms * j]
    if not opts: return name, None, None
    sk, p = min(opts)
    return name, plain / sk, p
for s in sys.argv[1:]:
    m, n, k = map(int, s.split("x"))
    print(s, *ratio(m, n, k))
