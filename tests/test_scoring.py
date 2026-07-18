import torch

from moep.scoring import criteria_matrix, keep_sets, select


def _synthetic_stats():
    """Expert 3 dominant on every signal; expert 0 never routed; expert 5 rarely routed
    but with huge activations (separates mean-based from frequency-weighted criteria).
    Per-expert gate quality g and norm scale m are strictly distinct to avoid topk ties."""
    L = 2
    counts = torch.tensor([[0, 10, 20, 500, 30, 2, 40, 50]] * L, dtype=torch.float32)
    g = torch.tensor([0.0, 0.30, 0.31, 0.60, 0.32, 0.33, 0.34, 0.35])
    m = torch.tensor([0.0, 1.00, 1.10, 2.00, 1.20, 1.30, 1.35, 1.40])
    stats = {
        "counts": counts,
        "gate_sum": counts * g,
        "prob_mass": counts * 0.9 + 1.0,
        "norm_sum": counts * m,
        "norm_sq_sum": counts * m * m,
        "reap_sum": counts * g * m,
        "total_tokens": torch.tensor(652.0),
    }
    stats["norm_sum"][:, 5] = 400.0        # 2 tokens x norm 200
    stats["norm_sq_sum"][:, 5] = 80000.0
    stats["reap_sum"][:, 5] = 120.0
    return stats


def test_dominant_expert_always_kept_and_dead_dropped():
    stats = _synthetic_stats()
    for crit in ("freq", "gate_mass", "reap", "reap_flow", "man", "msan"):
        keeps, info = select(stats, crit, keep_ratio=0.5)
        for row in keeps:
            assert 3 in row, crit
            assert 0 not in row, crit
        assert info["keep_per_layer"] == 4


def test_mean_vs_frequency_disagree_on_rare_loud_expert():
    stats = _synthetic_stats()
    mats = criteria_matrix(stats)
    keep_mean = keep_sets(mats["man"], 4)[0]
    keep_freq = keep_sets(mats["freq"], 4)[0]
    assert 5 in keep_mean       # mean norm 200 dominates
    assert 5 not in keep_freq   # 2 routed tokens of 652 is negligible frequency


def test_contrastive_prefers_domain_specialists():
    a, b = _synthetic_stats(), _synthetic_stats()
    b["reap_sum"][:, 6] = 0.001  # expert 6 matters for domain A, dead in reference B
    keeps, _ = select(a, "contrastive", keep_ratio=0.25, reference_stats=b)
    for row in keeps:
        assert 6 in row


def test_guard_protects_reference_experts_and_fills_by_reap():
    a, b = _synthetic_stats(), _synthetic_stats()
    # expert 1 carries big general (reference) gate mass but is weak on the primary
    # profile: exactly the expert reap-only pruning discards and guard must protect.
    b["gate_sum"][:, 1] = 500.0
    a["reap_sum"][:, 1] = 0.001
    keeps_reap, _ = select(a, "reap", keep_ratio=0.5)
    keeps_guard, info = select(a, "guard2", keep_ratio=0.5, reference_stats=b)
    mats_reap = criteria_matrix(a)["reap"]
    for li, (row_reap, row_guard) in enumerate(zip(keeps_reap, keeps_guard)):
        assert 1 not in row_reap        # reap alone drops it
        assert 1 in row_guard           # guard keeps it
        assert len(row_guard) == info["keep_per_layer"] == 4
        assert row_guard == sorted(set(row_guard))
        # non-guarded slots are filled in reap order: everything kept beyond the guard
        # must outscore everything dropped
        dropped = set(range(8)) - set(row_guard)
        for e in set(row_guard) - {1, 3}:  # ref top-2 = {1, 3}
            assert all(mats_reap[li, e] > mats_reap[li, d] for d in dropped)


def test_guard_larger_than_budget_rejected():
    a, b = _synthetic_stats(), _synthetic_stats()
    try:
        select(a, "guard4", keep_ratio=0.25, reference_stats=b)  # keep_n = 2 < 4
        raise AssertionError("guard_n > keep_n must raise")
    except ValueError:
        pass


def test_guard_requires_reference_stats():
    try:
        select(_synthetic_stats(), "guard2", keep_ratio=0.5)
        raise AssertionError("guard without reference stats must raise")
    except ValueError:
        pass


def test_keep_sets_sorted_unique():
    scores = torch.rand(3, 16)
    for row in keep_sets(scores, 5):
        assert row == sorted(set(row)) and len(row) == 5


def test_random_criterion_seeded_and_valid():
    stats = _synthetic_stats()
    k1, i1 = select(stats, "random", keep_ratio=0.5, seed=7)
    k2, _ = select(stats, "random", keep_ratio=0.5, seed=7)
    k3, _ = select(stats, "random", keep_ratio=0.5, seed=8)
    assert k1 == k2 and k1 != k3 and i1["seed"] == 7
    for row in k1:
        assert row == sorted(set(row))
