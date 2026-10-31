#!/usr/bin/env python3
"""Derive Y's t=3 Poseidon constants from the Poseidon reference procedure.

Usage:  python3 tools/gen_poseidon_constants.py           # write the Rust file
        python3 tools/gen_poseidon_constants.py --check   # compare, write nothing

Poseidon's parameters are not chosen by hand. The paper (Grassi, Khovratovich,
Rechberger, Roy, Schofnegger) fixes them: a Grain LFSR seeded with the field,
S-box, width and round counts yields the round constants and then a Cauchy MDS.
This script runs that procedure, so Y's table does not depend on anyone's
transcription of it.

Y evaluates the *optimized* form circomlib uses: one constant per partial round,
the MDS split into a change of basis `P` and sparse matrices `S`. The emitter
fixes the shape of each piece (lane 0 enters the S-box untouched, `S` is
identity below its first row and column), and that shape admits exactly one
solution, which `optimize` computes.

Two independent checks run before anything is written:
  * the reference permutation must reproduce circomlib's published digests,
    which fixes the reading of the procedure (constant order, MDS orientation);
  * the optimized tables, evaluated the way `zk_emitter.rs` evaluates them,
    must agree with the reference permutation on every vector.
"""
import random
import sys

OUT = "src/zk_poseidon_constants.rs"

P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
N_BITS = 254
T = 3
N_ROUNDS_F = 8
N_ROUNDS_P = 57
HALF_F = N_ROUNDS_F // 2

# circomlib's Poseidon(2) digests; the first is circomlibjs's published vector.
DIGESTS = [
    ((1, 2), 7853200120776062878684798364095072458815029376092732009249414926327459813530),
    ((3, 4), 14763215145315200506921711489642608356394854266165572616578112107564877678998),
    ((0, 0), 14744269619966411208579211824598458697587494354926760081771325075741142829156),
    ((7, 0), 10402197090275139279073177788985849389816807868761640028215734431067655199248),
]


def grain_bits():
    """The paper's self-shrinking Grain LFSR, seeded for GF(p), x^5, n, t, R_F, R_P."""
    s = []
    # field = 1 (prime), S-box = 0 (x^alpha)
    for value, width in ((1, 2), (0, 4), (N_BITS, 12), (T, 12), (N_ROUNDS_F, 10), (N_ROUNDS_P, 10)):
        s += [int(c) for c in bin(value)[2:].zfill(width)]
    s += [1] * 30

    def step():
        b = s[62] ^ s[51] ^ s[38] ^ s[23] ^ s[13] ^ s[0]
        s.pop(0)
        s.append(b)
        return b

    for _ in range(160):
        step()
    while True:
        # Bits come in pairs; the second is kept only when the first is 1.
        while step() == 0:
            step()
        yield step()


def take(bits, n):
    return int("".join(str(next(bits)) for _ in range(n)), 2)


def reference_parameters():
    bits = grain_bits()
    rc = []
    for _ in range((N_ROUNDS_F + N_ROUNDS_P) * T):
        x = take(bits, N_BITS)
        while x >= P:  # round constants are rejection-sampled
            x = take(bits, N_BITS)
        rc.append(x)
    # The MDS samples are reduced, not rejected.
    xy = [take(bits, N_BITS) % P for _ in range(2 * T)]
    xs, ys = xy[:T], xy[T:]
    if len(set(xy)) != len(xy) or any((x + y) % P == 0 for x in xs for y in ys):
        sys.exit("the first Cauchy sample is degenerate; the procedure would resample")
    mds = [[inv(xs[i] + ys[j]) for j in range(T)] for i in range(T)]
    return [rc[r * T:(r + 1) * T] for r in range(N_ROUNDS_F + N_ROUNDS_P)], mds


def inv(x):
    return pow(x % P, P - 2, P)


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) % P for j in range(len(b[0]))]
            for i in range(len(a))]


def matvec(a, v):
    return [sum(a[i][k] * v[k] for k in range(len(v))) % P for i in range(len(a))]


def matinv(a):
    n = len(a)
    m = [list(row) + [int(i == j) for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        piv = next(r for r in range(c, n) if m[r][c] % P)
        m[c], m[piv] = m[piv], m[c]
        k = inv(m[c][c])
        m[c] = [x * k % P for x in m[c]]
        for r in range(n):
            if r != c and m[r][c]:
                f = m[r][c]
                m[r] = [(x - f * y) % P for x, y in zip(m[r], m[c])]
    return [row[n:] for row in m]


def transpose(a):
    return [list(r) for r in zip(*a)]


def reference_permutation(inputs, rc, mds):
    """state <- MDS * sbox(state + rc[r]), x^5 on every lane in full rounds, lane 0 in partial."""
    state = [0] + list(inputs)
    for r, c in enumerate(rc):
        state = [(s + k) % P for s, k in zip(state, c)]
        full = r < HALF_F or r >= HALF_F + N_ROUNDS_P
        state = [pow(s, 5, P) if full or i == 0 else s for i, s in enumerate(state)]
        state = matvec(mds, state)
    return state[0]


def optimize(rc, mds):
    """The optimized tables in the layout `zk_emitter.rs::emit_poseidon` reads.

    Partial round k computes A * sbox0(z + rc[k]). Write X_k = Sp_k * D_k with
    D_k = diag(1, X_k[1:,1:]) and Sp_k identity below its first row and column.
    D_k leaves lane 0 alone, so it commutes with the lane-0 S-box and folds into
    the round before: X_{k-1} = D_k * A, starting from X_last = A. The first
    partial round's D lands in the last full round of the first half: P = D * A.
    Constants are pushed back the same way; each round keeps only lane 0's.
    """
    a = mds
    first, last = HALF_F, HALF_F + N_ROUNDS_P - 1
    sp, d = {}, {last + 1: [[int(i == j) for j in range(T)] for i in range(T)]}
    x = a
    for k in range(last, first - 1, -1):
        dk = [[1] + [0] * (T - 1)] + [[0] + x[i][1:] for i in range(1, T)]
        sub_inv = matinv([row[1:] for row in x[1:]])
        s_row = [x[0][0]] + [sum(x[0][1 + m] * sub_inv[m][j] for m in range(T - 1)) % P
                             for j in range(T - 1)]
        sp[k] = [s_row] + [[x[i][0]] + [int(i == j) for j in range(1, T)] for i in range(1, T)]
        d[k] = dk
        x = matmul(dk, a)
    p_applied = x  # D_first * A

    # Lane 0 of the S-box input must match the reference exactly; the other
    # lanes carry an offset g that is zero after the last partial round.
    g = [0] * T
    kappa = {}
    for k in range(last, first - 1, -1):
        h = [(gi + ci) % P for gi, ci in zip(g, matvec(d[k + 1], rc[k + 1]))]
        v = matvec(matinv(sp[k]), h)
        kappa[k] = v[0]
        g = [0] + v[1:]
    a_inv = matinv(a)
    c_star = matvec(a_inv, [(c + e) % P for c, e in zip(rc[first], matvec(matinv(d[first]), g))])

    # A full round's constant is added after the previous S-box and before its
    # mix, so it goes in as A^-1 * rc; only the very first is added raw.
    c = list(rc[0])
    for r in range(1, HALF_F):
        c += matvec(a_inv, rc[r])
    c += c_star
    c += [kappa[k] for k in range(first, last + 1)]
    for r in range(last + 2, N_ROUNDS_F + N_ROUNDS_P):
        c += matvec(a_inv, rc[r])
    s = []
    for k in range(first, last + 1):
        s += sp[k][0] + [sp[k][i][0] for i in range(1, T)]
    # The emitter's `mix` computes out[i] = sum_j mat[j][i] * in[j], so it
    # stores the transpose of the matrix it applies.
    return {"C": c, "M": transpose(a), "P": transpose(p_applied), "S": s}


def optimized_permutation(inputs, t):
    """`emit_poseidon`, line for line, over integers."""
    c, m, p, s = t["C"], t["M"], t["P"], t["S"]

    def mix(state, mat):
        return [sum(mat[j][i] * state[j] for j in range(T)) % P for i in range(T)]

    sbox = lambda v: pow(v, 5, P)
    state = [(v + c[j]) % P for j, v in enumerate([0] + list(inputs))]
    for r in range(HALF_F - 1):
        state = mix([(sbox(v) + c[(r + 1) * T + j]) % P for j, v in enumerate(state)], m)
    state = mix([(sbox(v) + c[HALF_F * T + j]) % P for j, v in enumerate(state)], p)
    width = 2 * T - 1
    for r in range(N_ROUNDS_P):
        inp = list(state)
        inp[0] = (sbox(state[0]) + c[(HALF_F + 1) * T + r]) % P
        out = [sum(inp[i] * s[width * r + i] for i in range(T)) % P]
        out += [(inp[j] + inp[0] * s[width * r + T + j - 1]) % P for j in range(1, T)]
        state = out
    for r in range(HALF_F - 1):
        base = (HALF_F + 1) * T + N_ROUNDS_P + r * T
        state = mix([(sbox(v) + c[base + j]) % P for j, v in enumerate(state)], m)
    return sum(sbox(v) * m[j][0] for j, v in enumerate(state)) % P


def render(t):
    def flat(name, xs):
        rows = ['    "%s",' % hex(x) for x in xs]
        return "\n".join(["pub const POSEIDON_%s_T3: [&str; %d] = [" % (name, len(xs))] + rows + ["];"])

    def matrix(name, xs):
        lines = ["pub const POSEIDON_%s_T3: [[&str; 3]; 3] = [" % name]
        for row in xs:
            lines += ["    ["] + ['        "%s",' % hex(x) for x in row] + ["    ],"]
        return "\n".join(lines + ["];"])

    header = '''//! circomlib-compatible Poseidon parameters for BN254, t=3 (two inputs).
//!
//! GENERATED by `tools/gen_poseidon_constants.py`, which derives them from the
//! Poseidon reference procedure (Grain-LFSR round constants, Cauchy MDS) and
//! checks the result against circomlib's published digests. Do not hand-edit:
//! these are not "some constants", they ARE the hash function. Changing one
//! value yields a hash that matches no other implementation, which silently
//! breaks interop with every circomlib-built Merkle tree and commitment.
//!
//! Layout is circomlib's *optimized* Poseidon (Grassi-Khovratovich-Rechberger-
//! Roy-Schofnegger), where the partial rounds fold the MDS into the sparse `S`
//! matrices. `C` is correspondingly pre-folded and has length
//! `t*R_F + R_P = 81`, not `t*(R_F+R_P)`. `M` is the MDS, `P` the change of
//! basis applied once before the partial rounds.
//!
//! Verified against circomlib's digests in `tests/zk_poseidon_interop.rs` and
//! `tests/circom_frontend.rs`.

'''
    return (header
            + "pub const POSEIDON_T3_ROUNDS_F: usize = %d;\n" % N_ROUNDS_F
            + "pub const POSEIDON_T3_ROUNDS_P: usize = %d;\n\n" % N_ROUNDS_P
            + flat("C", t["C"]) + "\n\n"
            + matrix("M", t["M"]) + "\n\n"
            + matrix("P", t["P"]) + "\n\n"
            + flat("S", t["S"]) + "\n")


def main():
    check = sys.argv[1:] == ["--check"]
    if sys.argv[1:] not in ([], ["--check"]):
        sys.exit(__doc__)

    rc, mds = reference_parameters()
    for inputs, want in DIGESTS:
        got = reference_permutation(inputs, rc, mds)
        if got != want:
            sys.exit("reference Poseidon%s = %d, circomlib publishes %d" % (inputs, got, want))
    tables = optimize(rc, mds)
    rng = random.Random(0)
    vectors = [i for i, _ in DIGESTS] + [(rng.randrange(P), rng.randrange(P)) for _ in range(16)]
    for inputs in vectors:
        if optimized_permutation(inputs, tables) != reference_permutation(inputs, rc, mds):
            sys.exit("optimized tables disagree with the reference permutation on %s" % (inputs,))
    print("reference matches %d published digests; optimized form matches on %d vectors"
          % (len(DIGESTS), len(vectors)))

    text = render(tables)
    if check:
        if open(OUT).read() != text:
            sys.exit("%s differs from the derived constants; rerun without --check" % OUT)
        print("%s is up to date" % OUT)
    else:
        with open(OUT, "w") as f:
            f.write(text)
        print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
