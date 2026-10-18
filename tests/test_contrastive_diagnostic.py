import importlib

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
)


def test_readouts_distinguish_help_harm_mask_and_terminal_event():
    api = importlib.import_module("active_ocr.contrastive_diagnostic")
    ids = torch.tensor([[1, 1]])
    p = torch.tensor([[0.0, 3.0, 2.0, -10.0, -11.0]])
    m = torch.tensor([[0.0, 6.0, -1.0, -10.0, -11.0]])
    processors = LogitsProcessorList([RepetitionPenaltyLogitsProcessor(1.1)])
    r = api.token_readouts(p, m, ids, processors, 2, [3, 4])
    assert r["paired"]["top"] == 1
    assert r["contrastive"]["top"] == 2
    assert r["contrastive"]["rank"] == 1
    assert r["paired"]["rank"] == 2
    assert r["contrastive"]["nll"] < r["paired"]["nll"]
    assert 0 < r["paired"]["eos_mass"] < 1
    harm = api.token_readouts(
        p, torch.tensor([[0.0, 8.0, -1.0, -10.0, -11.0]]), ids, processors, 1, [3, 4]
    )
    assert harm["paired"]["correct"] and not harm["contrastive"]["correct"]
    excluded = api.token_readouts(p, m, ids, processors, 3, [3, 4])
    assert excluded["contrastive"]["nll"] is None
    assert excluded["contrastive"]["excluded"]
    assert excluded["paired_masked"]["excluded"]
    terminal = api.token_readouts(p, m, ids, processors, 3, [1, 3], terminal=True)
    assert terminal["paired"]["correct"]  # any native EOS is accepted
    assert terminal["paired"]["eos_mass"] > 0.5


def test_forced_stream_scores_before_appending_current_target():
    api = importlib.import_module("active_ocr.contrastive_diagnostic")
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
    settings = {"max_new_tokens": 64, "repetition_penalty": 1.1, "do_sample": False}
    target = [3, 4, 5]
    with torch.inference_mode():
        rows = list(api.forced_states(model, batch, target, settings))
        for i, row in enumerate(rows):
            assert row["ids"][0].tolist() == ids[0].tolist() + target[:i]
            assert row["cache_tokens"] == ids.shape[1] + i
            expected = api.prefix_state(model, batch, target[:i], settings)
            assert torch.equal(row["logits"], expected["logits"])
        other = list(api.forced_states(model, batch, [3, 4, 9], settings))
    assert torch.equal(rows[-1]["logits"], other[-1]["logits"])
