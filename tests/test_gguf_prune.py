import numpy as np
import pytest

gp = pytest.importorskip("moep.gguf_prune")
gguf = gp.gguf

ARCH, E, USED, EMBD, FF = "testmoe", 4, 2, 32, 6


def _write_model(path, extra_tensor=None):
    rng = np.random.default_rng(0)
    w = gguf.GGUFWriter(str(path), ARCH)
    w.add_uint32(f"{ARCH}.expert_count", E)
    w.add_uint32(f"{ARCH}.expert_used_count", USED)
    w.add_string("general.name", "tiny")
    w.add_array(f"{ARCH}.feed_forward_length", [FF, FF])
    for il in (0, 1):
        w.add_tensor(f"blk.{il}.ffn_gate_inp.weight", rng.standard_normal((E, EMBD), dtype=np.float32))
        w.add_tensor(f"blk.{il}.exp_probs_b.bias", rng.standard_normal(E, dtype=np.float32))
        up = gguf.quants.quantize(rng.standard_normal((E, FF, EMBD), dtype=np.float32), gguf.GGMLQuantizationType.Q8_0)
        w.add_tensor(f"blk.{il}.ffn_up_exps.weight", up, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        w.add_tensor(f"blk.{il}.ffn_down_exps.weight", rng.standard_normal((E, EMBD, FF), dtype=np.float32))
        w.add_tensor(f"blk.{il}.ffn_gate_inp_shexp.weight", rng.standard_normal(EMBD, dtype=np.float32))
        w.add_tensor(f"blk.{il}.attn_norm.weight", rng.standard_normal(EMBD, dtype=np.float32))
    if extra_tensor:
        w.add_tensor(extra_tensor, rng.standard_normal((E, 3), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _tensors(path):
    return {t.name: t for t in gguf.GGUFReader(str(path)).tensors}


def test_keep_all_is_byte_identical(tmp_path):
    src, dst = tmp_path / "src.gguf", tmp_path / "dst.gguf"
    _write_model(src)
    gp.prune_gguf(src, dst, {0: list(range(E)), 1: list(range(E))})
    a, b = _tensors(src), _tensors(dst)
    assert a.keys() == b.keys()
    for name in a:
        assert np.array_equal(a[name].data, b[name].data) and a[name].tensor_type == b[name].tensor_type
    assert gguf.GGUFReader(str(dst)).fields[f"{ARCH}.expert_count"].contents() == E


def test_kept_rows_are_byte_identical_and_other_layers_untouched(tmp_path):
    src, dst = tmp_path / "src.gguf", tmp_path / "dst.gguf"
    _write_model(src)
    summary = gp.prune_gguf(src, dst, {0: [2, 0]})
    a, b = _tensors(src), _tensors(dst)
    for name in ("ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_up_exps.weight", "ffn_down_exps.weight"):
        assert np.array_equal(b[f"blk.0.{name}"].data, a[f"blk.0.{name}"].data[[0, 2]])
        assert np.array_equal(b[f"blk.1.{name}"].data, a[f"blk.1.{name}"].data)
    assert np.array_equal(b["blk.0.ffn_gate_inp_shexp.weight"].data, a["blk.0.ffn_gate_inp_shexp.weight"].data)
    assert b["blk.0.ffn_up_exps.weight"].tensor_type == gguf.GGMLQuantizationType.Q8_0
    # layer 1 still holds E experts, so the model-wide count must not shrink below it
    assert summary["expert_count"] == E
    assert gp.verify_pruned(src, dst, {0: [0, 2]})["identical"]


def test_uniform_prune_sets_expert_count(tmp_path):
    src, dst = tmp_path / "src.gguf", tmp_path / "dst.gguf"
    _write_model(src)
    gp.prune_gguf(src, dst, {0: [1, 3], 1: [0, 1]})
    assert gguf.GGUFReader(str(dst)).fields[f"{ARCH}.expert_count"].contents() == 2


def test_refuses_fewer_experts_than_routed(tmp_path):
    src = tmp_path / "src.gguf"
    _write_model(src)
    with pytest.raises(ValueError, match="routes to"):
        gp.prune_gguf(src, tmp_path / "dst.gguf", {0: [1]})


def test_refuses_unknown_per_expert_tensor(tmp_path):
    src = tmp_path / "src.gguf"
    _write_model(src, extra_tensor="blk.0.ffn_mystery_exps.weight")
    with pytest.raises(ValueError, match="looks per-expert"):
        gp.prune_gguf(src, tmp_path / "dst.gguf", {0: [0, 1]})


def test_verify_catches_wrong_selection(tmp_path):
    src, dst = tmp_path / "src.gguf", tmp_path / "dst.gguf"
    _write_model(src)
    gp.prune_gguf(src, dst, {0: [0, 1]})
    with pytest.raises(AssertionError):
        gp.verify_pruned(src, dst, {0: [0, 2]})
