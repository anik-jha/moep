import json

import pytest
import torch

from moep.attribution import (ItemAccumulator, attach_item_hooks, bootstrap_diff,
                              build_item_sets, measure_item, pruned_mask)


def _write_transitions(root, tag, dataset, t1_by_task):
    d = root / "evalplus" / tag / f"{dataset}-repair"
    d.mkdir(parents=True)
    per = {tid: {"t1": s, "t2": None} for tid, s in t1_by_task.items()}
    (d / "transitions.json").write_text(json.dumps({"per_task": per}))


def test_build_item_sets(tmp_path):
    for ds in ("humaneval", "mbpp"):
        _write_transitions(tmp_path, "base_q8", ds,
                           {"t/0": "pass", "t/1": "pass", "t/2": "fail"})
        _write_transitions(tmp_path, "reap25_q8", ds,
                           {"t/0": "fail", "t/1": "pass", "t/2": "fail"})
    sets = build_item_sets(tmp_path)
    for ds in ("humaneval", "mbpp"):
        # base-fail items belong to neither group regardless of reap25's outcome
        assert sets[ds] == {"fail": ["t/0"], "ctrl": ["t/1"]}


def test_pruned_mask(tmp_path):
    keep = {"info": {"num_experts": 8}, "layer_ids": [0, 1],
            "keep": {"0": [0, 1, 2, 3], "1": [4, 5, 6, 7]}}
    p = tmp_path / "keep.json"
    p.write_text(json.dumps(keep))
    mask = pruned_mask(p)
    assert mask.shape == (2, 8)
    assert mask[0].tolist() == [False] * 4 + [True] * 4
    assert mask[1].tolist() == [True] * 4 + [False] * 4
    assert int(mask.sum()) == 8


def test_measure_item_fractions(tiny_ckpt, tiny_adapter, sample_ids):
    _, model = tiny_ckpt
    ids = sample_ids[0]

    # all experts pruned -> fraction exactly 1; none pruned -> exactly 0
    for fill, expect in ((True, 1.0), (False, 0.0)):
        mask = torch.full((2, 8), fill, dtype=torch.bool)
        acc, handles = attach_item_hooks(model, tiny_adapter, mask)
        try:
            fr = measure_item(model, acc, ids, start=8)
        finally:
            for h in handles:
                h.remove()
        assert fr["pooled"] == pytest.approx(expect)
        assert all(x == pytest.approx(expect) for x in fr["per_layer"])
    # routed mass was actually accumulated over the windowed positions
    assert float(acc.total.sum()) > 0

    # a proper subset pruned -> strictly interior fraction, and window start matters
    torch.manual_seed(0)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :4] = True
    acc, handles = attach_item_hooks(model, tiny_adapter, mask)
    try:
        full = measure_item(model, acc, ids, start=1)
        tail = measure_item(model, acc, ids, start=ids.shape[0] - 4)
    finally:
        for h in handles:
            h.remove()
    for fr in (full, tail):
        assert 0.0 < fr["pooled"] < 1.0
    assert full["pooled"] != pytest.approx(tail["pooled"])


def test_measure_item_rejects_bad_window(tiny_ckpt, tiny_adapter, sample_ids):
    _, model = tiny_ckpt
    acc = ItemAccumulator(2)
    with pytest.raises(AssertionError):
        measure_item(model, acc, sample_ids[0], start=0)
    with pytest.raises(AssertionError):
        measure_item(model, acc, sample_ids[0], start=sample_ids.shape[1])


def test_bootstrap_diff_direction():
    fail = [0.5, 0.6, 0.55, 0.62, 0.58]
    ctrl = [0.30, 0.32, 0.31, 0.29, 0.33]
    s = bootstrap_diff(fail, ctrl, b=2000)
    assert s["diff"] == pytest.approx(sum(fail) / 5 - sum(ctrl) / 5)
    assert s["ci95"][0] > 0 and s["p_le_0"] < 0.01
    same = bootstrap_diff(ctrl, ctrl, b=2000)
    assert same["ci95"][0] < 0 < same["ci95"][1]
