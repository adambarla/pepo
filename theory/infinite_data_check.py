"""Does token-level PEPO inherit the single-policy guarantee?

Infinite-data limit on one seen prompt: every member recovers the true reward
exactly on the covered responses S (the support of pi_data), and is arbitrary
within [-R, R] on uncovered responses. The count bonus is 0 on covered and
BIG on uncovered responses (token-level: charged at the first uncovered
prefix, as in eq:ctoken). A method with a single-policy guarantee must then
match the best covered response, whatever the members do off S.

Policies (paper definitions, Section 3 and Appendix C):
  pi_out  ∝ min_l pi_l(a) e^{-zeta_l - C(a)/beta}          (Lemma closed_form)
  pi_prod ∝ prod_h min_l pi_l(a_h|x_h) e^{-c_tok/beta}     (rectangular optimum)
  pi_tok  = prod_h normalized min_l pi_l(.|x_h) e^{-c_tok/beta}
Run: uv run --no-project --with numpy python theory/infinite_data_check.py
"""

from __future__ import annotations

import itertools

import numpy as np

BIG = 1e3


def evaluate(V, H, L, beta, R, rng, agree=False):
    seqs = list(itertools.product(range(V), repeat=H))
    n = len(seqs)
    covered = rng.random(n) < 0.5
    covered[rng.integers(n)] = True
    r_true = rng.uniform(-1, 1, size=n)
    r_true[~covered] = -R  # uncovered responses are truly bad
    ref = np.full(n, 1.0 / n)

    r_members = np.tile(r_true, (L, 1))
    rows = 1 if agree else L  # agree: every member has the same off-data rewards
    r_members[:, ~covered] = rng.choice([-R, R], size=(rows, (~covered).sum()))

    pi_l = ref * np.exp((r_members - r_members.max()) / beta)
    pi_l /= pi_l.sum(axis=1, keepdims=True)
    zeta = np.log(pi_l[:, covered] / ref[covered]).mean(axis=1)
    C = np.where(covered, 0.0, BIG)

    # response level
    f = np.min(pi_l * np.exp(-zeta[:, None]), axis=0) * np.exp(-C / beta)
    pi_out = f / f.sum()

    # token level: conditionals from subtree masses, bonus at first uncovered prefix
    idx = {a: i for i, a in enumerate(seqs)}

    def subtree(p):
        return [idx[a] for a in seqs if a[: len(p)] == p]

    def prefix_covered(p):
        return any(covered[i] for i in subtree(p))

    g = np.ones(n)
    tok = np.ones(n)
    for a in seqs:
        i = idx[a]
        for h in range(H):
            s = a[:h]
            mass_s = pi_l[:, subtree(s)].sum(axis=1)
            cond = np.array(
                [pi_l[:, subtree(s + (v,))].sum(axis=1) / mass_s for v in range(V)]
            ).T  # (L, V)
            pen = np.array(
                [
                    BIG if (prefix_covered(s) and not prefix_covered(s + (v,))) else 0.0
                    for v in range(V)
                ]
            )
            m = cond.min(axis=0) * np.exp(-pen / beta)
            g[i] *= m[a[h]]
            tok[i] *= m[a[h]] / m.sum()
    pi_prod = g / g.sum()

    best = r_true[covered].max()
    return {
        name: best - float(pol @ r_true)
        for name, pol in [("pi_out", pi_out), ("pi_prod", pi_prod), ("pi_tok", tok)]
    }


def main():
    rng = np.random.default_rng(0)
    worst = {"pi_out": 0.0, "pi_prod": 0.0, "pi_tok": 0.0}
    frac_bad = {"pi_out": 0, "pi_prod": 0, "pi_tok": 0}
    trials = 3000
    for _ in range(trials):
        sub = evaluate(
            V=int(rng.integers(2, 4)),
            H=int(rng.integers(2, 4)),
            L=int(rng.integers(2, 4)),
            beta=0.05,
            R=3.0,
            rng=rng,
        )
        for k, v in sub.items():
            worst[k] = max(worst[k], v)
            frac_bad[k] += v > 0.1
    print(f"{trials} instances, infinite data on covered responses, beta=0.05, R=3")
    print("suboptimality vs best covered response (true rewards in [-1, 1]):")
    for k in worst:
        print(
            f"  {k:7s}  worst = {worst[k]:.3f}   "
            f"fraction with gap > 0.1 = {frac_bad[k] / trials:.3f}"
        )


if __name__ == "__main__":
    main()
