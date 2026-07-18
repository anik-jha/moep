"""tokenize_rows must yield flat id lists across transformers versions (5.x changed
apply_chat_template to return a BatchEncoding by default), merge consecutive roles,
and skip unrenderable rows instead of raising."""

from pathlib import Path

import pytest

from moep.data import tokenize_rows, tokenize_rows_plain

TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "models/qwen3.6-35b-a3b"


@pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer_config.json").exists(),
                    reason="real tokenizer not downloaded")
def test_tokenize_rows_real_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    rows = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
        {"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"},
                      {"role": "assistant", "content": "c"}]},
    ]
    out = tokenize_rows(tok, rows, log=lambda *_: None)
    assert len(out) == 2
    for ids in out:
        assert isinstance(ids, list) and ids and all(isinstance(i, int) for i in ids)


@pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer_config.json").exists(),
                    reason="real tokenizer not downloaded")
def test_tokenize_rows_plain_skips_template():
    """Plain rendering must carry the message content and none of the chat-template
    control tokens (the gemma4 template inflates teacher-forced ppl ~30x, so ppl for
    template-sensitive families is measured on plain text)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    rows = [{"messages": [{"role": "user", "content": "print hello world"},
                          {"role": "assistant", "content": "print('hello world')"}]}]
    plain = tokenize_rows_plain(tok, rows)
    templated = tokenize_rows(tok, rows, log=lambda *_: None)
    assert len(plain) == 1 and plain[0]
    assert len(plain[0]) < len(templated[0])  # no template scaffolding
    text = tok.decode(plain[0])
    assert "print hello world" in text and "print(\'hello world\')" in text
    assert "<|im_start|>" not in text and "start_of_turn" not in text
