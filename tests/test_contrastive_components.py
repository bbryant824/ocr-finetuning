import importlib

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from active_ocr.contrastive_icl import contrastive_generate


def test_component_choice_isolates_stop_and_text_with_native_eos_and_ties():
    api = importlib.import_module("active_ocr.contrastive_components")
    p = torch.tensor([[1.0, 5.0, 4.0, -1.0, -2.0]])
    c = torch.tensor([[6.0, 3.0, 4.0, -torch.inf, -2.0]])
    eos = [0, 4]
    assert api.component_choice(p, c, eos, "contrast_stopping")[0] == 0
    assert api.component_choice(p, c, eos, "contrast_text")[0] == 2
    for mode, expected in [("paired", 1), ("contrastive", 0)]:
        assert api.component_choice(p, c, eos, mode)[0] == expected
    p = torch.tensor([[1.0, 3.0, 3.0, -1.0, 5.0]])
    c = torch.tensor([[1.0, 3.0, 4.0, -torch.inf, 2.0]])
    assert api.component_choice(p, c, eos, "contrast_text")[0] == 4
    assert api.component_choice(p, c, eos, "contrast_stopping")[0] == 1
    c = torch.tensor([[2.0, -torch.inf, -torch.inf, -torch.inf, 3.0]])
    p = torch.tensor([[1.0, -1.0, -2.0, -3.0, 4.0]])
    for mode in api.POLICIES:
        assert api.component_choice(p, c, eos, mode)[0] == 4


def test_generation_endpoints_reproduce_native_and_original_contrast():
    api = importlib.import_module("active_ocr.contrastive_components")
    torch.manual_seed(824)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=24,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            eos_token_id=[2, 3],
            pad_token_id=0,
        )
    ).eval()
    p = {"input_ids": torch.tensor([[1, 7, 7, 12]])}
    n = {"input_ids": torch.tensor([[1, 9, 7, 12]])}
    for batch in (p, n):
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    settings = {"max_new_tokens": 6, "do_sample": False, "repetition_penalty": 1.1}
    with torch.inference_mode():
        native = model.generate(**p, **settings)[0, 4:].tolist()
        original, _ = contrastive_generate(model, p, n, settings)
        paired, _, _ = api.component_generate(model, p, n, settings, "paired")
        contrast, trace, _ = api.component_generate(model, p, n, settings, "contrastive")
    assert paired == native
    assert contrast == original[0, 4:].tolist()
    assert [r["selected"] for r in trace] == contrast
    assert all(r["positive_cache_tokens"] == 4 + i for i, r in enumerate(trace))
