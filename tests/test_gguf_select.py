import json

import pytest

from moep.gguf_select import load_stats, select, split_half


def _half(counts, gate, reap, tokens):
    n = len(counts)
    return {"tokens": tokens, "counts": counts, "gate_sum": gate, "norm_sum": [1.0] * n,
            "norm_sq_sum": [1.0] * n, "reap_sum": reap}


@pytest.fixture
def stats_file(tmp_path):
    # layer 1: 8 experts, REAP favours 7,6,5,...; layer 3: 4 experts, routing mass favours 0
    l1 = _half([1, 2, 3, 4, 5, 6, 7, 8], [.1] * 8, [1, 4, 9, 16, 25, 36, 49, 64], 10)
    l3 = _half([4, 3, 2, 0], [.9, .5, .3, 0.0], [1, 1, 1, 0], 10)
    d = {"layers": {"1": {"n_expert": 8, "halves": [l1, l1]}, "3": {"n_expert": 4, "halves": [l3, l3]}}}
    p = tmp_path / "stats.json"
    p.write_text(json.dumps(d))
    return p


def test_keep_counts_scale_with_each_layer(stats_file):
    keep, info = select(load_stats(stats_file), "reap", 0.5, n_used=1)
    assert keep[1] == [4, 5, 6, 7]                 # mean g*||f|| = reap_sum / counts grows with index
    assert len(keep[3]) == 2 and 3 not in keep[3]  # never-routed expert scores 0 and goes first
    assert info["experts_kept"] == 6 and info["experts_total"] == 12


def test_never_keeps_fewer_than_routed(stats_file):
    keep, _ = select(load_stats(stats_file), "gate_mass", 0.25, n_used={1: 3, 3: 3})
    assert len(keep[1]) == 3 and len(keep[3]) == 3
    assert keep[3] == [0, 1, 2]


def test_identical_halves_are_perfectly_reliable(stats_file):
    sh = split_half(load_stats(stats_file), "gate_mass", 0.5, n_used=1)
    assert sh["jaccard_min"] == 1.0
    assert sh["spearman"][3] == pytest.approx(1.0)
    assert sh["spearman"][1] is None and sh["spearman_undefined_layers"] == 1  # constant gate_mass
