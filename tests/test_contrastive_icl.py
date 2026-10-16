"""Check native equivalence, independent caches and the declared plausibility rule."""

import importlib
import importlib.util
import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor


def api():
    assert importlib.util.find_spec("active_ocr.contrastive_icl"), "decoder not implemented"
    return importlib.import_module("active_ocr.contrastive_icl")


def tiny_model():
    torch.manual_seed(824)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
        )
    ).eval()


def batch(ids):
    ids = torch.tensor([ids])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_native_equivalence_includes_repetition_history_and_eos():
    decoder = api()
    model = tiny_model()
    positive, negative = batch([1, 7, 7, 12]), batch([1, 9, 4])
    settings = {"max_new_tokens": 6, "repetition_penalty": 1.1, "do_sample": False}
    with torch.inference_mode():
        native = model.generate(**positive, **settings)
        zero, _ = decoder.contrastive_generate(model, positive, negative, settings, alpha=0)
        identical, _ = decoder.contrastive_generate(model, positive, positive, settings)
        assert torch.equal(native, zero) and torch.equal(native, identical)
        model.generation_config.eos_token_id = int(native[0, positive["input_ids"].shape[1]])
        native = model.generate(**positive, **settings)
        zero, _ = decoder.contrastive_generate(model, positive, negative, settings, alpha=0)
    assert torch.equal(native, zero)
    assert native.shape[1] == positive["input_ids"].shape[1] + 1


def test_cached_two_branch_loop_matches_independent_uncached_reference():
    decoder = api()
    model = tiny_model()
    positive, negative = batch([1, 7, 7, 12]), batch([1, 9, 4])
    settings = {"max_new_tokens": 6, "repetition_penalty": 1.1, "do_sample": False}
    p, n = positive["input_ids"], negative["input_ids"]
    penalty = RepetitionPenaltyLogitsProcessor(1.1)
    with torch.inference_mode():
        for _ in range(6):
            zp = model(p, attention_mask=torch.ones_like(p), use_cache=False).logits[:, -1]
            zn = model(n, attention_mask=torch.ones_like(n), use_cache=False).logits[:, -1]
            eligible = penalty(p, zp.clone())
            mask = eligible - eligible.max(dim=-1, keepdim=True).values >= math.log(0.1)
            scores = penalty(p, zp + (zp - zn)).masked_fill(~mask, -torch.inf)
            token = scores.argmax(dim=-1, keepdim=True)
            p, n = torch.cat((p, token), -1), torch.cat((n, token), -1)
        actual, trace = decoder.contrastive_generate(model, positive, negative, settings)
    assert torch.equal(actual, p)
    assert len(trace) == 6
    assert all(row["positive_cache_tokens"] == 4 + i for i, row in enumerate(trace))
    assert all(row["negative_cache_tokens"] == 3 + i for i, row in enumerate(trace))
    assert all(row["shared_prefix_tokens"] == i for i, row in enumerate(trace))


def test_plausibility_uses_processed_positive_logits_and_rejects_nonfinite_input():
    decoder = api()
    ids = torch.tensor([[0, 0]])
    positive, negative = torch.tensor([[1.0, 0.9]]), torch.tensor([[-100.0, 100.0]])
    penalty = RepetitionPenaltyLogitsProcessor(2.0)
    scores = decoder.contrastive_scores(positive, negative, ids, penalty, alpha=1, beta=0.9)
    assert scores.argmax(-1).item() == 1 and torch.isneginf(scores[0, 0])
    zero = decoder.contrastive_scores(positive, negative, ids, penalty, alpha=0, beta=0.9)
    assert torch.equal(zero, penalty(ids, positive.clone()))
    with pytest.raises(ValueError, match="finite"):
        decoder.contrastive_scores(positive, negative * torch.nan, ids, penalty)


def test_flat_line_batch_is_grouped_without_losing_region_order():
    decoder = api()
    group = getattr(decoder, "group_queries", None)
    assert callable(group), "flat pilot batches need page grouping"
    box = {"x": 0, "y": 0, "width": 1, "height": 1}
    page = {"page_id": "validation:a", "image_sha256": "0" * 64, "width": 1, "height": 1}
    flat = [page | {"regions": [{"id": r, "box": box}]} for r in ("a", "b")]
    flat.append(page | {"page_id": "validation:b", "regions": [{"id": "c", "box": box}]})
    grouped = group(flat)
    decoder._check_records(grouped, labelled=False)
    assert [p["page_id"] for p in grouped] == ["validation:a", "validation:b"]
    assert [[r["id"] for r in p["regions"]] for p in grouped] == [["a", "b"], ["c"]]
    assert [len(p["regions"]) for p in flat] == [1, 1, 1]


def test_reference_reuse_checks_model_slice_adapter_and_supports():
    import copy

    check = getattr(api(), "check_reference", None)
    assert callable(check), "reference reuse needs identity and ordered coverage validation"
    config = {"models": {"qwen": {"repository": "qwen", "revision": "pinned"}}}
    queries = [{"page_id": "validation:a", "regions": [{"id": "line1"}]}]
    contexts = {"validation:a/line1": ["train:b/line2"]}
    receipt = {
        "source_run": "source", "model_repository": "qwen", "model_revision": "pinned",
        "adapter_sha256": None,
        "predictions": [{"page_id": "validation:a", "regions": [
            {"id": "line1", "demo_ids": ["train:b/line2"], "text": "output"}
        ]}],
    }
    assert list(check(receipt, queries, config, "source", contexts)) == ["validation:a/line1"]
    for field, value in [("source_run", "other"), ("model_revision", "other"),
                         ("adapter_sha256", "unexpected")]:
        bad = copy.deepcopy(receipt)
        bad[field] = value
        with pytest.raises(ValueError):
            check(bad, queries, config, "source", contexts)
    for changes in [[], [{"page_id": "validation:a", "regions": [{"id": "wrong"}]}]]:
        bad = receipt | {"predictions": changes}
        with pytest.raises(ValueError):
            check(bad, queries, config, "source", contexts)
    bad = copy.deepcopy(receipt)
    bad["predictions"][0]["regions"][0]["demo_ids"] = ["train:wrong/line2"]
    with pytest.raises(ValueError):
        check(bad, queries, config, "source", contexts)
