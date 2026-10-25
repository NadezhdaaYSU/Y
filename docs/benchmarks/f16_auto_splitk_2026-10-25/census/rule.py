"""rule.py TABLE: what candidate split rules choose at each census shape (66 SMs)."""
import re, sys
SMS = 66
unsplit = {}
for line in open(sys.argv[1]):
    f = line.split()
    if len(f) > 3 and f[1] == "1" and "x" in f[0]:
        m, n, k = map(int, f[0].split("x"))
        cm, cn, _ = map(int, f[2].split("x"))
        unsplit[(m, n, k)] = (f"{f[2]} {f[3]}", -(-m // cm) * -(-n // cn))
cdiv = lambda a, b: -(-a // b)
MENU_BIG = [(128, 128, "128x128x32 4x2", 4, 2), (64, 128, "64x128x32 2x4", 2, 4), (64, 64, "64x64x32 4x2", 4, 2)]

def block_fit(cm, cn, wm, wn):
    accs = (cm // wm) * (cn // wn) // 32
    thr = wm * wn * 32
    return 2 * accs + 100 <= min(65536 // thr, 255)

def valid_s(k, kmin):
    return [s for s in (2, 4, 8) if k % s == 0 and (k // s) % 8 == 0 and k // s >= kmin]

def rule_user(m, n, k, kmin=1024):
    """The first guess, made concrete with the record's split tile (128x128 4x2 where it fits)."""
    t1 = unsplit[(m, n, k)][1]
    if t1 >= SMS:
        return None
    best = None
    for s in valid_s(k, kmin):
        for cm, cn, name, wm, wn in MENU_BIG:
            if cm > max(m, 16) or cn > n:
                continue
            ctas = cdiv(m, cm) * cdiv(n, cn) * s
            if ctas <= SMS:
                key = (ctas, cm * cn, -s)
                if best is None or key > best[0]:
                    best = (key, s, name, ctas)
            break  # the first (largest) tile that fits the shape
    return best

def rule_refined(m, n, k, kmin=4096):
    """Trigger T1 < SMs; any menu tile (<= M x N, block sums fit above 32 rows) and S with
    T1 <= CTAs <= SMs, K/S >= kmin; most CTAs, then the larger tile, then the smaller S."""
    t1 = unsplit[(m, n, k)][1]
    if t1 >= SMS:
        return None
    best = None
    menu = MENU_BIG if m > 32 else []
    for s in valid_s(k, kmin):
        for cm, cn, name, wm, wn in menu:
            if cm > m or cn > n or not block_fit(cm, cn, wm, wn):
                continue
            ctas = cdiv(m, cm) * cdiv(n, cn) * s
            if t1 <= ctas <= SMS:
                key = (ctas, cm * cn, -s)
                if best is None or key > best[0]:
                    best = (key, s, name, ctas)
    return best

fmt = lambda b: "unsplit" if b is None else f"S={b[1]} {b[2]} ({b[3]} CTAs)"
print(f"{'shape':18s} {'unsplit tile':16s} T1 | user K/S>=1024 | refined K/S>=1024 | refined K/S>=2048 | refined K/S>=4096")
for (m, n, k), (tile, t1) in sorted(unsplit.items()):
    print(f"{m}x{n}x{k:<10d} {tile:16s} {t1:4d} | {fmt(rule_user(m, n, k)):28s} | {fmt(rule_refined(m, n, k, 1024)):28s} | {fmt(rule_refined(m, n, k, 2048)):28s} | {fmt(rule_refined(m, n, k, 4096))}")
