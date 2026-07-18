from pathlib import Path

from moep.codeval import _select_samples_file


def test_select_samples_file_prefers_sanitized_over_raw(tmp_path):
    root = tmp_path / "evalplus" / "base_q8" / "humaneval"
    d = root / "humaneval"
    d.mkdir(parents=True)
    # Write raw first so directory-order glob would surface it before the sanitized file
    # on filesystems that don't sort readdir() output.
    (d / "base_q8_openai_temp_0.0.raw.jsonl").write_text('{"task_id": "HumanEval/0", "solution": "```python\\ndef f(): pass\\n```"}\n')
    (d / "base_q8_openai_temp_0.0.jsonl").write_text('{"task_id": "HumanEval/0", "solution": "def f(): pass"}\n')

    picked = _select_samples_file(root, "humaneval")

    assert picked is not None
    assert picked.name == "base_q8_openai_temp_0.0.jsonl"


def test_select_samples_file_none_when_only_raw_present(tmp_path):
    root = tmp_path / "evalplus" / "base_q8" / "humaneval"
    d = root / "humaneval"
    d.mkdir(parents=True)
    (d / "base_q8_openai_temp_0.0.raw.jsonl").write_text('{"task_id": "HumanEval/0", "solution": "x"}\n')

    assert _select_samples_file(root, "humaneval") is None
