"""Check observed-prefix scoring, deterministic sampling and the fixed advance screen."""

import importlib
import importlib.util
from collections import Counter

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
)


def api():
    assert importlib.util.find_spec("active_ocr.query_intervention"), (
        "query intervention not implemented"
    )
    return importlib.import_module("active_ocr.query_intervention")


def test_first_difference_preserves_exact_prefix_and_eos():
    fn = api().first_difference
    assert fn([3, 2], [3, 4, 2], {2}) == {
        "prefix_token_ids": [3],
        "contrastive_token_id": 2,
        "text_only_token_id": 4,
        "answer_position": 1,
        "eos_difference": True,
    }
    with pytest.raises(ValueError, match="differ"):
        fn([3, 2], [3, 2], {2})


def test_selection_is_balanced_deterministic_page_capped_and_reserves_eos():
    fn = api().select_cases
    candidates = [
        {
            "key": f"validation:p{i % 8}/{group}-{stratum}-{i}",
            "page_id": f"validation:p{i % 8}",
            "group": group,
            "stratum": stratum,
            "eos_difference": stratum == "later" and i == 0,
        }
        for group in ("improved", "harmed")
        for stratum in ("first", "later")
        for i in range(8)
    ]
    chosen = fn(candidates, 8)
    assert chosen == fn(list(reversed(candidates)), 8)
    assert Counter((r["group"], r["stratum"]) for r in chosen) == {
        (g, s): 4 for g in ("improved", "harmed") for s in ("first", "later")
    }
    assert max(Counter((r["group"], r["page_id"]) for r in chosen).values()) <= 2
    assert all(
        any(r["group"] == g and r["eos_difference"] for r in chosen) for g in ("improved", "harmed")
    )
    with pytest.raises(ValueError, match="balanced"):
        fn(candidates[:3], 8)


def test_query_effects_rank_the_composite_and_preserve_masked_scores():
    fn = api().decision
    processors = LogitsProcessorList([RepetitionPenaltyLogitsProcessor(1.1)])
    ids = torch.tensor([[0]])
    original = {
        "paired": torch.tensor([[2.0, 1.0, 0.0, -2.0]]),
        "mismatched": torch.tensor([[1.0, 0.0, 3.0, -2.0]]),
        "text-only": torch.tensor([[0.0, 2.0, 1.0, -1.0]]),
    }
    donor = original | {"paired": torch.tensor([[1.0, 2.0, 0.0, -2.0]])}
    before = fn(original, {k: ids for k in original}, {k: processors for k in original}, 0, 1)
    after = fn(donor, {k: ids for k in donor}, {k: processors for k in donor}, 0, 1)
    effects = api().query_effects(before, after)
    assert effects["G"] == 4 and effects["rank_G"] == 2
    assert before["contrastive"]["top_token_id"] == 0
    assert before["text-only"]["top_token_id"] == 1
    masked = original | {"paired": torch.tensor([[3.0, -10.0, 0.0, -2.0]])}
    row = fn(masked, {k: ids for k in masked}, {k: processors for k in masked}, 0, 1)
    assert row["contrastive"]["candidates"][1]["processed_score"] is None
    assert not row["contrastive"]["candidates"][1]["admissible"]


def test_prefix_readout_replays_native_cached_history_and_final_projection():
    fn = api().prefix_state
    torch.manual_seed(824)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=24,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            eos_token_id=None,
            pad_token_id=0,
        )
    ).eval()
    ids = torch.tensor([[1, 7, 7, 12]])
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    settings = {"max_new_tokens": 5, "repetition_penalty": 1.1, "do_sample": False}
    with torch.inference_mode():
        native = model.generate(**batch, **settings)[0, 4:].tolist()
        state = fn(model, batch, native[:2], settings)
    scores = state["processors"](state["ids"], state["logits"].clone())
    assert int(scores.argmax()) == native[2]
    assert state["geometry"]["cache_tokens"] == 6
    assert state["geometry"]["final_max_abs_error"] <= 0.001
    assert state["ids"][0, 4:].tolist() == native[:2]


def test_advance_screen_rejects_harmful_group_or_rank_only_confound():
    fn = api().screen
    rows = [
        {
            "group": g,
            "stratum": s,
            "G": 1.0 if g == "improved" else -1.0,
            "rank_G": 2 if g == "improved" else -2,
            "eos_difference": False,
            "both_admissible": True,
        }
        for g in ("improved", "harmed")
        for s in ("first", "later")
        for _ in range(4)
    ]
    assert fn(rows)["passed"]
    assert not fn([r | {"G": 3.0} if r["group"] == "harmed" else r for r in rows])["passed"]
    assert not fn([r | {"rank_G": -1} for r in rows])["passed"]
