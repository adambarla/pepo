"""Executable checks for the response-level vs token-level PEPO claims.

Setting: one evaluation prompt (so centering and count penalties cancel),
responses are sequences of H tokens over a vocabulary of size V (EOS can be
modelled as an absorbing token), members pi_l and pi_ref are autoregressive.
Every claim is checked on random instances; the adversarial loop widens the
member disagreement and the regularization strength to look for violations.

Objects (per prompt):
  rho_l(s, v)  = beta * log pi_l(v|s) / pi_ref(v|s)          token log-ratio
  m(v|s)       = min_l pi_l(v|s),   z(s) = sum_v m(v|s)       token agreement
  f(a)         = min_l pi_l(a)                                response-level min
  g(a)         = prod_h m(a_h|x_h)                            product of token mins
  pi_out       = f / Z,  Z = sum f                            response-level PEPO
  pi_prod      = g / F,  F = sum g
  pi_tok(a)    = prod_h m(a_h|x_h) / z(x_h)                   token-level PEPO
  Lambda(a)    = sum_h log 1/z(x_h) >= 0

Run: uv run --with numpy python theory/token_level_checks.py
"""

from __future__ import annotations

import itertools
import math

import numpy as np

TOL = 1e-9


class Instance:
    def __init__(self, V, H, L, scale, beta, rng, iid=False):
        self.V, self.H, self.L, self.beta = V, H, L, beta
        self.prefixes: list[tuple[int, ...]] = [
            p for h in range(H) for p in itertools.product(range(V), repeat=h)
        ]
        self.seqs = list(itertools.product(range(V), repeat=H))

        def dist(logits):
            p = np.exp(logits - logits.max())
            return p / p.sum()

        shared = rng.normal(size=V)
        if iid:  # state-independent members: product measures over tokens
            noise = [scale * rng.normal(size=V) for _ in range(L)]
            self.pi = [{p: dist(shared + eps) for p in self.prefixes} for eps in noise]
        else:
            base = {p: rng.normal(size=V) for p in self.prefixes}
            self.pi = [
                {p: dist(base[p] + scale * rng.normal(size=V)) for p in self.prefixes}
                for _ in range(L)
            ]
        self.ref = {p: dist(rng.normal(size=V)) for p in self.prefixes}
        self.m = {p: np.min([pl[p] for pl in self.pi], axis=0) for p in self.prefixes}
        self.z = {p: self.m[p].sum() for p in self.prefixes}

    def seq_prob(self, table, a):
        return math.prod(table[a[:h]][a[h]] for h in range(self.H))

    def arrays(self):
        pi_l = np.array([[self.seq_prob(t, a) for a in self.seqs] for t in self.pi])
        ref = np.array([self.seq_prob(self.ref, a) for a in self.seqs])
        g = np.array([self.seq_prob(self.m, a) for a in self.seqs])
        lam = np.array(
            [sum(-math.log(self.z[a[:h]]) for h in range(self.H)) for a in self.seqs]
        )
        tok = np.array(
            [
                math.prod(self.m[a[:h]][a[h]] / self.z[a[:h]] for h in range(self.H))
                for a in self.seqs
            ]
        )
        return pi_l, ref, g, lam, tok


def kl(p, q):
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def gibbs(ref, reward, beta):
    w = ref * np.exp((reward - reward.max()) / beta)
    return w / w.sum()


def regularized_value(pi, reward, ref, beta):
    return float(pi @ reward) - beta * kl(pi, ref)


def check(inst, rng):
    """Return a dict of named booleans; every entry must be True."""
    beta = inst.beta
    pi_l, ref, g, lam, tok = inst.arrays()
    f = pi_l.min(axis=0)
    Z, F = f.sum(), g.sum()
    pi_out, pi_prod = f / Z, g / F
    R_l = beta * np.log(pi_l / ref)  # sequence implicit rewards
    R_resp = R_l.min(axis=0)  # min over members of the return
    R_rect = np.array(  # return of the per-token min
        [
            sum(
                beta * math.log(inst.m[a[:h]][a[h]] / inst.ref[a[:h]][a[h]])
                for h in range(inst.H)
            )
            for a in inst.seqs
        ]
    )
    out = {}

    # C1. Closed forms: pi_out and pi_prod are the KL-regularized optima of the
    # response-level and token-rectangular pessimistic rewards.
    out["C1 pi_out = Gibbs(min_l R_l)"] = np.allclose(gibbs(ref, R_resp, beta), pi_out)
    out["C1 pi_prod = Gibbs(sum_h min_l rho_l)"] = np.allclose(
        gibbs(ref, R_rect, beta), pi_prod
    )
    best = regularized_value(pi_prod, R_rect, ref, beta)
    out["C1 pi_prod beats random policies"] = all(
        regularized_value(q / q.sum(), R_rect, ref, beta) <= best + TOL
        for q in rng.dirichlet(np.ones(len(inst.seqs)), size=50)
    )

    # C2. Each member's token log-ratio is its soft Q-function: under per-token
    # reward rho_l, the soft value of every prefix is 0, so Q_l = rho_l, and
    # pi_tok(.|s) is the soft policy improvement step on min_l Q_l.
    for k, table in enumerate(inst.pi):
        V_soft: dict[tuple[int, ...], float] = {}
        for p in reversed(inst.prefixes):  # deepest prefixes first
            cont = [
                V_soft.get(p + (v,), 0.0) if len(p) + 1 < inst.H else 0.0
                for v in range(inst.V)
            ]
            rho = beta * np.log(table[p] / inst.ref[p])
            V_soft[p] = beta * math.log(
                float(np.sum(inst.ref[p] * np.exp((rho + np.array(cont)) / beta)))
            )
        out[f"C2 member {k} soft values are 0"] = max(map(abs, V_soft.values())) < 1e-8
    out["C2 pi_tok = soft-greedy on min_l Q_l"] = all(
        np.allclose(
            gibbs(
                inst.ref[p],
                np.min([beta * np.log(t[p] / inst.ref[p]) for t in inst.pi], axis=0),
                beta,
            ),
            inst.m[p] / inst.z[p],
        )
        for p in inst.prefixes
    )

    # C3. Ordering: sum of mins <= min of sums, hence g <= f and F <= Z;
    # Z is bounded by any pair's sequence Bhattacharyya coefficient, which is
    # bounded by the product over depths of the worst per-token coefficient.
    out["C3 R_rect <= R_resp"] = bool(np.all(R_rect <= R_resp + TOL))
    out["C3 F <= Z"] = F <= Z + TOL
    for i, j in itertools.combinations(range(inst.L), 2):
        bc_seq = float(np.sqrt(pi_l[i] * pi_l[j]).sum())
        bc_path = math.prod(
            max(
                float(np.sqrt(inst.pi[i][p] * inst.pi[j][p]).sum())
                for p in inst.prefixes
                if len(p) == h
            )
            for h in range(inst.H)
        )
        out[f"C3 Z <= BC_seq({i},{j}) <= prod_h max bc"] = (
            Z <= bc_seq + TOL and bc_seq <= bc_path + TOL
        )

    # C4. Token-level identities: pi_tok = g * e^Lambda, F = E_tok[e^-Lambda],
    # KL(pi_tok || pi_prod) = log E_tok[exp(-(Lambda - E Lambda))] <= E Lambda,
    # and the expected rejection-sampling trials 1/Z <= e^{E_tok Lambda}.
    e_lam = float(tok @ lam)
    out["C4 pi_tok = g e^Lambda"] = np.allclose(tok, g * np.exp(lam))
    out["C4 F = E_tok e^-Lambda"] = math.isclose(F, float(tok @ np.exp(-lam)))
    kl_tp = kl(tok, pi_prod)
    out["C4 KL(tok||prod) = log E e^-(Lambda-ELambda)"] = math.isclose(
        kl_tp, math.log(float(tok @ np.exp(-(lam - e_lam)))), abs_tol=1e-9
    )
    out["C4 KL(tok||prod) <= E Lambda"] = kl_tp <= e_lam + TOL
    out["C4 1/Z <= e^{E Lambda}"] = 1 / Z <= math.exp(e_lam) * (1 + 1e-9)
    # Sampling pi_tok and accepting with prob e^-Lambda gives exactly pi_prod.
    acc = tok * np.exp(-lam)
    out["C4 pi_tok thinned by e^-Lambda = pi_prod"] = np.allclose(
        acc / acc.sum(), pi_prod
    )
    return out, dict(Z=Z, F=F, e_lam=e_lam, kl_tp=kl_tp)


def main():
    rng = np.random.default_rng(0)
    failures, n = {}, 0
    for _ in range(400):
        inst = Instance(
            V=int(rng.integers(2, 4)),
            H=int(rng.integers(1, 6)),
            L=int(rng.integers(2, 5)),
            scale=float(rng.choice([0.05, 0.5, 2.0, 5.0])),
            beta=float(rng.choice([0.01, 0.1, 1.0, 10.0])),
            rng=rng,
            iid=bool(rng.integers(2)),
        )
        res, _ = check(inst, rng)
        n += 1
        for k, v in res.items():
            if not v:
                failures[k] = failures.get(k, 0) + 1
    print(f"{n} random instances; failed checks: {failures or 'none'}")

    # C5. State-independent members (product measures): Lambda is constant, so
    # pi_tok = pi_prod exactly (zero myopia cost), while Z <= bc^H decays
    # geometrically and exact response-level sampling needs >= bc^-H trials.
    print("\nC5 iid members, V=3, L=3, beta=0.1:")
    print(" H   1/Z (RS trials)   e^{H C} lower bound   KL(tok||prod)")
    rng = np.random.default_rng(1)
    for H in range(1, 9):
        inst = Instance(3, H, 3, 1.0, 0.1, np.random.default_rng(7), iid=True)
        res, s = check(inst, rng)
        assert all(res.values()), [k for k, v in res.items() if not v]
        p = [t[()] for t in inst.pi]
        # Chernoff information C(p, q) = -min_lam log sum p^lam q^(1-lam);
        # min(a, b) <= a^lam b^(1-lam) tensorizes, so Z <= exp(-H max_ij C_ij).
        lams = np.linspace(0, 1, 2001)
        chern = max(
            -min(math.log(float(np.sum(p[i] ** t * p[j] ** (1 - t)))) for t in lams)
            for i, j in itertools.combinations(range(3), 2)
        )
        assert 1 / s["Z"] >= math.exp(H * chern) * (1 - 1e-9)
        print(
            f"{H:2d}   {1 / s['Z']:15.2f}   {math.exp(H * chern):19.2f}   "
            f"{s['kl_tp']:.2e}"
        )

    # C6. Concentrated disagreement: members agree on the first token and on
    # everything after token 0, but disagree completely after token 1.
    # pi_out and pi_prod avoid branch 1; pi_tok enters it half of the time.
    print("\nC6 fork: agree at step 1, disagree only inside branch 1:")
    inst = Instance(2, 3, 2, 0.0, 0.1, np.random.default_rng(3))
    for t in inst.pi:
        for p in inst.prefixes:
            t[p] = np.array([0.5, 0.5])
    for p in inst.prefixes:
        if len(p) >= 1 and p[0] == 1:
            inst.pi[0][p] = np.array([0.98, 0.02])
            inst.pi[1][p] = np.array([0.02, 0.98])
    inst.m = {p: np.min([t[p] for t in inst.pi], axis=0) for p in inst.prefixes}
    inst.z = {p: inst.m[p].sum() for p in inst.prefixes}
    res, s = check(inst, rng)
    assert all(res.values()), [k for k, v in res.items() if not v]
    pi_l, _, g, _, tok = inst.arrays()
    f = pi_l.min(axis=0)
    branch1 = np.array([a[0] == 1 for a in inst.seqs])
    for name, pol in [
        ("pi_out", f / f.sum()),
        ("pi_prod", g / g.sum()),
        ("pi_tok", tok),
    ]:
        print(f"  mass on branch 1 under {name:7s}: {pol[branch1].sum():.3f}")
    print(f"  KL(tok||prod) = {s['kl_tp']:.3f}  (E Lambda = {s['e_lam']:.3f})")


if __name__ == "__main__":
    main()
