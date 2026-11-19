"""Numerical checks for 'The exact asymptotics of the waiting time for a first-passage pattern'.

1. The recursion (8.2) of Proposition 8.1 for the intensity r_{n,h}.
2. Exact E T for small n (Markov chain on the last n steps), compared with 1/r and
   with the bounds 1 - q <= r E T <= 1 of Lemma 7.1.
3. rho_h from (1.1), rho_2 = 8/(3 pi), and G_1 = 2 from (1.2).
4. Table 1.
Runs in about a minute on a laptop.
"""
import math
import sys

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

NMAX, HMAX = 1024, 32
lg = np.array([math.lgamma(i + 1) for i in range(200000 + 2 * HMAX + 5)])


def pl(l, b):
    """P(S_l = b) for arrays l, b."""
    l = np.asarray(l)
    b = np.asarray(b)
    out = np.zeros(np.broadcast(l, b).shape)
    ok = ((l + b) % 2 == 0) & (np.abs(b) <= l) & (l >= 0)
    ll = np.broadcast_to(l, out.shape)[ok]
    kk = (np.broadcast_to(l + b, out.shape)[ok]) // 2
    out[ok] = np.exp(lg[ll] - lg[kk] - lg[ll - kk] - ll * math.log(2))
    return out


def g(l, b):
    l = np.asarray(l)
    b = np.asarray(b)
    return np.where(b > 0, b / np.maximum(l, 1) * pl(l, b), 0.0)


def intensity(N, H):
    """r[n, h] from the recursion, 0 <= n <= N, 0 <= h <= H."""
    R = np.zeros((N + 1, H + 1))
    ls = np.arange(1, N + 1)
    for h in range(1, H + 1):
        if h > 1:
            Lg, Jg = np.meshgrid(ls, np.arange(1, h), indexing="ij")
            A = g(Lg, Jg) * (pl(Lg, Jg) - pl(Lg, 2 * h - Jg))  # A[l-1, j-1]
        for n in range(h, N + 1, 2):
            p = float(g(n, h))
            s = 0.0
            if h > 1 and n > 1:
                sub = R[n - 1:0:-1, h - 1:0:-1]  # [l-1, j-1] -> R[n-l, h-j]
                s = float(np.sum(A[: n - 1, :] * sub))
            R[n, h] = p * (1 - p) - s
    return R


def exact_ET(n, h):
    """E T and q by the Markov chain on the last n steps (bit i = 1: step +1)."""
    N = 1 << n

    def is_fp(w):
        s, path = 0, [0]
        for i in range(n):
            s += 1 if (w >> i) & 1 else -1
            path.append(s)
        return path[-1] == -h and all(path[-1] < x for x in path[:-1])

    pat = np.array([is_fp(w) for w in range(N)])
    free = np.where(~pat)[0]
    idx = -np.ones(N, dtype=np.int64)
    idx[free] = np.arange(len(free))
    A = lil_matrix((len(free), len(free)))
    for k, w in enumerate(free):
        A[k, k] += 1.0
        for bit in (0, 1):
            w2 = (w >> 1) | (bit << (n - 1))
            if not pat[w2]:
                A[k, idx[w2]] -= 0.5
    x = spsolve(csr_matrix(A), np.ones(len(free)))
    E = np.zeros(N)
    E[free] = x
    ET = n + E.mean()
    # q = P(a hit at one of n consecutive ends) = P(T <= 2n - 1)
    dist = np.where(pat, 0.0, 1.0 / N)  # survivors after the first window (T > n-1 trivially; T = n hit)
    surv = dist.sum()
    for _ in range(n - 1):
        new = np.zeros(N)
        for bit in (0, 1):
            w2 = (np.arange(N) >> 1) | (bit << (n - 1))
            np.add.at(new, w2, 0.5 * dist)
        new[pat] = 0.0
        dist = new
    q = 1.0 - dist.sum()
    return ET, q


def rho(hmax, lmax=200000):
    ls = np.arange(1, lmax + 1)
    rh = [None, 1.0]
    for h in range(2, hmax + 1):
        tot = 0.0
        for j in range(1, h):
            Hj = (h - j) / h * float(np.sum(g(ls, j) * (pl(ls, j) - pl(ls, 2 * h - j))))
            tot += Hj * rh[h - j]
        rh.append(1.0 - tot)
    return rh


def main():
    ok = True
    R = intensity(NMAX, HMAX)

    print("1-2. exact E T against 1/r and Lemma 7.1")
    for n, h in [(6, 2), (8, 2), (9, 3), (10, 4), (12, 4), (13, 3), (14, 4)]:
        ET, q = exact_ET(n, h)
        r = R[n, h]
        good = (1 - q) - 1e-9 <= r * ET <= 1 + 1e-9
        ok &= good
        print(f"  n={n:2d} h={h} E T={ET:.6f} 1/r={1/r:.6f} q={q:.4f} r E T={r*ET:.6f} {'ok' if good else 'FAIL'}")
    # h = 1: r = p(1-p) and p E T = 1
    for n in (5, 9):
        ET, q = exact_ET(n, 1)
        p = float(g(n, 1))
        good = abs(p * ET - 1) < 1e-9 and abs(R[n, 1] - p * (1 - p)) < 1e-15
        ok &= good
        print(f"  h=1 n={n}: p E T={p*ET:.12f} {'ok' if good else 'FAIL'}")

    print("3. constants")
    rh = rho(6)
    good = abs(rh[2] - 8 / (3 * math.pi)) < 1e-6
    ok &= good
    print(f"  rho_2={rh[2]:.8f}  8/(3pi)={8/(3*math.pi):.8f} {'ok' if good else 'FAIL'}")
    print("  rho_h, h=1..6:", " ".join(f"{x:.5f}" for x in rh[1:]))
    print("  r/p at h=2 from the recursion:", " ".join(f"n={n}:{R[n,2]/float(g(n,2)):.6f}" for n in (16, 64, 256, 1024)))
    G1 = 1 + sum(0.5 ** l for l in range(1, 200))
    print(f"  G_1={G1:.12f}; p/r at h=n:", " ".join(f"{float(g(n,n))/R[n,n]:.6f}" for n in (4, 10, 20, 30)))

    print("4. table (lambda = -1)")
    rows = [(10, 4, "38.024010", 38.024010, None), (12, 4, "43.516365", 43.516365, None),
            (100, 10, "416.3", 416.3, (412.7, 419.9)), (1000, 32, "4{,}946.3", 4946.3, (4891.4, 5001.2)),
            (10000, 100, "56{,}062", 56062, (55063, 57061)), (100000, 316, "657{,}063", 657063, (621650, 692475)),
            (1000000, 1000, "7{,}812{,}672", 7812672, (6890114, 8735231))]
    for n, h, Es, E, ci in rows:
        k = (n + h) // 2
        p = h / n * math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2))
        lead = math.log(min(h * h, (n / h) ** 2)) / (2 * math.pi)
        pr = f"{p / R[n, h]:.3f}" if n <= NMAX and h <= HMAX else "--"
        d = f"{p*E-lead:.3f}" if ci is None else f"{p*ci[0]-lead:.2f} to {p*ci[1]-lead:.2f}"
        nn = f"{n:,}".replace(",", "{,}")
        print(f"{nn} & {h} & {Es} & {p*E:.3f} & {pr} & {lead:.3f} & {d}\\\\")
    print("ALL CHECKS PASS" if ok else "SOME CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
