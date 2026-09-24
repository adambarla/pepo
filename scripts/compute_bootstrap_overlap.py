import numpy as np

N = 61135
L = 4
SEED = 42
k = N // L  # 15283

# --- Exact replica of the code's logic for bootstrap_replace (replace=True) ---
print("=== bootstrap_replace (replace=True) — exact code logic ===")
rng = np.random.default_rng(SEED)

# Each submodel: rng.choice(n, size=k, replace=True) — keeps duplicates
choices = [rng.choice(N, size=k, replace=True) for _ in range(L)]

# Instance-level pairwise overlap: how many of the k*k pairs share the same index?
for i in range(L):
    for j in range(i + 1, L):
        # Count exact index matches between the two (k-length) arrays
        # Sum over all i in choice_i of count of same index in choice_j
        matches = sum(np.sum(choices[j] == idx) for idx in choices[i])
        pct = matches / k * 100
        print(f"  Instance overlap submodel {i} & {j}: {matches} / {k} ({pct:.1f}%)")

# Unique-index overlap
for i in range(L):
    for j in range(i + 1, L):
        set_i = set(choices[i].tolist())
        set_j = set(choices[j].tolist())
        overlap = len(set_i & set_j)
        avg_unique = (len(set_i) + len(set_j)) / 2
        print(
            f"  Unique-index overlap submodel {i} & {j}: {overlap} "
            f"(avg unique per submodel: {avg_unique:.0f})"
        )

unique_sizes = [len(set(c.tolist())) for c in choices]
union = set()
for c in choices:
    union.update(c.tolist())
print(f"  Unique samples per submodel: {unique_sizes}")
print(
    f"  Union size (unique indices):  {len(union)} / {N} ({len(union) / N * 100:.1f}%)"
)
print()

# --- For comparison: bootstrap (replace=False) — exact code logic ---
print("=== bootstrap (replace=False) — exact code logic ===")
rng = np.random.default_rng(SEED)
choices_det = [rng.choice(N, size=k, replace=False) for _ in range(L)]

for i in range(L):
    for j in range(i + 1, L):
        set_i = set(choices_det[i].tolist())
        set_j = set(choices_det[j].tolist())
        overlap = len(set_i & set_j)
        print(f"  Overlap submodel {i} & {j}: {overlap} ({overlap / k * 100:.1f}%)")

union_det = set()
for c in choices_det:
    union_det.update(c.tolist())
print(f"  Union size: {len(union_det)} / {N} ({len(union_det) / N * 100:.1f}%)")
