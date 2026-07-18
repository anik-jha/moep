import json

from moep.repair_eval import (INSTRUCTION_PREFIX, build_feedback, build_repair_messages,
                              failed_task_ids, load_eval_results, merge_solutions,
                              transitions, turn1_user_message)


def test_turn1_user_message_matches_evalplus_openai_backend():
    # Must reproduce evalplus 0.3.1 provider/openai.py:24 verbatim: the assistant's
    # failed solution has to sit in the exact context it was generated in.
    msg = turn1_user_message("def f(x):\n    ...\n")
    assert msg == INSTRUCTION_PREFIX + "\n```python\ndef f(x):\n    ...\n```"


def test_build_feedback_modes():
    fb, mode = build_feedback("f([1, 2])", "[2, 1]")
    assert mode == "input_expected"
    assert "`f([1, 2])`" in fb and "`[2, 1]`" in fb

    fb, mode = build_feedback("f([1, 2])", None)
    assert mode == "input_only"
    assert "`f([1, 2])`" in fb and "[2, 1]" not in fb

    fb, mode = build_feedback(None, None)
    assert mode == "generic"
    assert "failed its unit tests" in fb
    for f in (fb,):
        assert "markdown code block" in f  # every mode demands a parseable answer


def test_build_repair_messages_structure():
    msgs = build_repair_messages("def f():\n    ...", "def f():\n    return 1", "feedback!")
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"].startswith(INSTRUCTION_PREFIX)
    assert msgs[1]["content"] == "```python\ndef f():\n    return 1\n```"
    assert msgs[2]["content"] == "feedback!"


def test_load_eval_results_and_failed_ids(tmp_path):
    p = tmp_path / "r_eval_results.json"
    p.write_text(json.dumps({"eval": {
        "T/0": [{"plus_status": "pass"}],
        "T/1": [{"plus_status": "fail"}],
        "T/2": [{"base_status": "fail"}],   # no plus_status recorded => not a pass
        "T/3": [],                          # no samples => dropped entirely
    }}))
    res = load_eval_results(p)
    assert set(res) == {"T/0", "T/1", "T/2"}
    assert failed_task_ids(res) == ["T/1", "T/2"]


def test_merge_solutions_replaces_only_failed_with_usable_turn2():
    rows = [{"task_id": "T/0", "solution": "pass0"},
            {"task_id": "T/1", "solution": "fail1"},
            {"task_id": "T/2", "solution": "fail2"},
            {"task_id": "T/3", "solution": "fail3"}]
    merged, no_attempt = merge_solutions(
        rows, failed_ids={"T/1", "T/2", "T/3"},
        turn2_solutions={"T/1": "fixed1", "T/2": "   ", "T/0": "never-used"})
    by_id = {r["task_id"]: r["solution"] for r in merged}
    assert by_id == {"T/0": "pass0",   # turn-1 pass untouched even if turn2 present
                     "T/1": "fixed1",  # replaced
                     "T/2": "fail2",   # whitespace-only turn2 -> keep turn-1
                     "T/3": "fail3"}   # no turn2 at all -> keep turn-1
    assert no_attempt == 2
    assert len(merged) == len(rows)


def test_transitions_counts():
    t1 = {"T/0": {"plus_status": "pass"}, "T/1": {"plus_status": "pass"},
          "T/2": {"plus_status": "fail"}, "T/3": {"plus_status": "fail"}}
    t2 = {"T/0": {"plus_status": "pass"}, "T/1": {"plus_status": "fail"},
          "T/2": {"plus_status": "pass"}, "T/3": {"plus_status": "fail"}}
    tr = transitions(t1, t2)
    assert (tr["kept_pass"], tr["regressed"], tr["repaired"], tr["still_fail"]) == (1, 1, 1, 1)
    assert tr["per_task"]["T/2"] == {"t1": "fail", "t2": "pass"}
