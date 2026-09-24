"""Response- vs token-level PEPO in the regime the experiments actually run in.

Evaluation prompts are unseen, so the count bonus and centering cancel and the
only pessimism is the ensemble minimum. Members are KL-regularized optima of
noisy rewards r_l = r* + e_l, where e_l = sqrt(rho) * shared + sqrt(1 - rho) *
own_l: rho = 0 is the independent-shards hypothesis behind the ensemble,
rho = 1 means every member makes the same mistake (nothing can detect it).
Noise is larger on responses outside the data (sigma_off) than inside
(sigma_on). We report the true regularized value J_beta of each policy.

Part 2 checks the token-level pessimism bound derived in the write-up:
  if A_min := min_l A_l <= A* on the support of pi_tok, then
  J(pi*) - J(pi_tok) <= E_{pi*}[sum_h (A* - A_min)] + beta * KL(pi_tok || pi_prod),
where A_l(s, v) = beta log pi_l(v|s)/pi_ref(v|s) is member l's soft advantage and
A* the true one.
Run: uv run --no-project --with numpy python theory/practical_regime_check.py
"""

from __future__ import annotations

import itertools

import numpy as np


class Tree:
    def __init__(self, V, H):
        self.V, self.H = V, H
        self.seqs = list(itertools.product(range(V), repeat=H))
        self.n = len(self.seqs)
        self.prefixes: list[tuple[int, ...]] = [
            p for h in range(H) for p in itertools.product(range(V), repeat=h)
        ]
        idx = {a: i for i, a in enumerate(self.seqs)}
        self.sub = {
            p + (v,): np.array(
                [idx[a] for a in self.seqs if a[: len(p) + 1] == p + (v,)]
            )
            for p in self.prefixes
            for v in range(V)
        }

    def conditionals(self, pi):
        """(prefix -> (V,) next-token distribution) of a sequence distribution."""
        out = {}
        for p in self.prefixes:
            mass = np.array([pi[self.sub[p + (v,)]].sum() for v in range(self.V)])
            out[p] = mass / mass.sum()
        return out

    def seq_prob(self, cond, a):
        return float(np.prod([cond[a[:h]][a[h]] for h in range(self.H)]))


def gibbs(ref, r, beta):
    w = ref * np.exp((r - r.max()) / beta)
    return w / w.sum()


def J(pi, r, ref, beta):
    mask = pi > 0
    return float(pi @ r) - beta * float(np.sum(pi[mask] * np.log(pi[mask] / ref[mask])))


def policies(tree, pis, ref):
    f = pis.min(axis=0)
    conds = [tree.conditionals(p) for p in pis]
    m = {p: np.min([c[p] for c in conds], axis=0) for p in tree.prefixes}
    tok = np.array(
        [
            np.prod([m[a[:h]][a[h]] / m[a[:h]].sum() for h in range(tree.H)])
            for a in tree.seqs
        ]
    )
    g = np.array([tree.seq_prob(m, a) for a in tree.seqs])
    return {
        "pi_out": f / f.sum(),
        "pi_tok": tok,
        "pi_prod": g / g.sum(),
        "mixture": pis.mean(axis=0),
        "member": pis[0],
    }, conds


def part1():
    beta, L = 0.1, 4
    print("Part 1: true J_beta minus J_beta(member 0 = plain DPO), mean over 300 seeds")
    print(" rho  sigma_off  H   pi_out   pi_tok  pi_prod  mixture")
    for rho in (0.0, 0.5, 1.0):
        for sigma_off in (0.3, 1.0):
            for V, H in ((3, 2), (3, 4)):
                tree = Tree(V, H)
                acc = {k: [] for k in ("pi_out", "pi_tok", "pi_prod", "mixture")}
                for seed in range(300):
                    rng = np.random.default_rng(seed)
                    ref = rng.dirichlet(np.ones(tree.n) * 2)
                    r_true = rng.normal(size=tree.n)
                    covered = rng.random(tree.n) < 0.3
                    sigma = np.where(covered, 0.1, sigma_off)
                    shared = rng.normal(size=tree.n)
                    own = rng.normal(size=(L, tree.n))
                    noise = np.sqrt(rho) * shared + np.sqrt(1 - rho) * own
                    pis = np.array(
                        [gibbs(ref, r_true + sigma * e, beta) for e in noise]
                    )
                    pols, _ = policies(tree, pis, ref)
                    base = J(pols["member"], r_true, ref, beta)
                    for k in acc:
                        acc[k].append(J(pols[k], r_true, ref, beta) - base)
                row = "  ".join(f"{np.mean(acc[k]):+7.3f}" for k in acc)
                print(f" {rho:.1f}   {sigma_off:4.1f}     {H}  {row}")


def part2():
    """Check the advantage-pessimism bound on random instances."""
    rng = np.random.default_rng(0)
    checked = held = 0
    for _ in range(2000):
        V, H, L = (
            int(rng.integers(2, 4)),
            int(rng.integers(1, 4)),
            int(rng.integers(2, 5)),
        )
        beta = float(rng.choice([0.05, 0.3, 1.0]))
        tree = Tree(V, H)
        ref = rng.dirichlet(np.ones(tree.n))
        r_true = rng.normal(size=tree.n)
        # members pessimistic on average: noise with negative drift so the
        # premise A_min <= A* holds on some instances and fails on others
        pis = np.array(
            [gibbs(ref, r_true + rng.normal(-0.5, 1.0, tree.n), beta) for _ in range(L)]
        )
        pols, conds = policies(tree, pis, ref)
        star = gibbs(ref, r_true, beta)
        cstar, cref = tree.conditionals(star), tree.conditionals(ref)
        a_star = {p: beta * np.log(cstar[p] / cref[p]) for p in tree.prefixes}
        a_min = {
            p: np.min([beta * np.log(c[p] / cref[p]) for c in conds], axis=0)
            for p in tree.prefixes
        }
        tok, prod = pols["pi_tok"], pols["pi_prod"]
        premise = all(
            a_min[a[:h]][a[h]] <= a_star[a[:h]][a[h]] + 1e-12
            for i, a in enumerate(tree.seqs)
            if tok[i] > 1e-15
            for h in range(H)
        )
        if not premise:
            continue
        checked += 1
        err = sum(
            star[i] * sum(a_star[a[:h]][a[h]] - a_min[a[:h]][a[h]] for h in range(H))
            for i, a in enumerate(tree.seqs)
        )
        kl_tp = float(np.sum(tok * np.log(tok / prod)))
        gap = J(star, r_true, ref, beta) - J(tok, r_true, ref, beta)
        held += gap <= err + beta * kl_tp + 1e-9
    print(f"\nPart 2: bound held on {held}/{checked} instances satisfying the premise")


if __name__ == "__main__":
    part1()
    part2()
