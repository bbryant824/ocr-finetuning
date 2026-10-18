"""Reference construction must preserve the intended pair sets and query boundary."""

import importlib
import importlib.util
from collections import Counter
from types import SimpleNamespace

import torch


def api():
    assert importlib.util.find_spec("active_ocr.contrastive_references"), "reference runner missing"
    return importlib.import_module("active_ocr.contrastive_references")


def test_references_preserve_wrong_and_correct_pair_multisets():
    m = api()
    ids = list("abcdefgh")
    original = list(zip(ids[1:] + ids[:1], ids, strict=True))
    wrong = m.reference_pairs(ids, "wrong_reordered")
    correct = m.reference_pairs(ids, "correct_reordered")
    assert Counter(wrong) == Counter(original) and wrong != original
    assert wrong == list(zip(ids, ids[-1:] + ids[:-1], strict=True))
    assert Counter(correct) == Counter(zip(ids, ids, strict=True))
    assert correct == list(zip(ids[1:] + ids[:1], ids[1:] + ids[:1], strict=True))
    assert m.reference_pairs(ids, "query_noise") == list(zip(ids, ids, strict=True))


def test_diffusion_changes_only_query_and_keeps_temporal_duplicates():
    m = api()
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(patch_size=2, temporal_patch_size=2)
    )
    pixels = torch.arange(5 * 3 * 2 * 2 * 2, dtype=torch.float32).reshape(5, 3, 2, 2, 2)
    pixels[:, :, 1] = pixels[:, :, 0]
    pixels = pixels.flatten(1)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 1, 3], [1, 1, 2]]),
        "pixel_values": pixels,
    }
    frozen = pixels.clone()
    before_rng = torch.get_rng_state().clone()
    changed, meta = m.query_noise(batch, "query-a", processor)
    repeated, _ = m.query_noise(batch, "query-a", processor)
    other, _ = m.query_noise(batch, "query-b", processor)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert torch.equal(pixels, frozen) and torch.equal(changed["pixel_values"][:3], pixels[:3])
    assert torch.equal(changed["pixel_values"], repeated["pixel_values"])
    assert not torch.equal(changed["pixel_values"][3:], other["pixel_values"][3:])
    for k in ("input_ids", "attention_mask", "image_grid_thw"):
        assert torch.equal(batch[k], changed[k])
    shaped = changed["pixel_values"][3:].reshape(2, 3, 2, 2, 2)
    assert torch.equal(shaped[:, :, 0], shaped[:, :, 1])
    betas = torch.sigmoid(torch.linspace(-6, 6, 1000)) * (0.005 - 1e-5) + 1e-5
    a = torch.cumprod(1 - betas, 0)[500]
    generator = torch.Generator().manual_seed(meta["seed"])
    noise = torch.randn((2, 3, 1, 2, 2), generator=generator).expand(-1, -1, 2, -1, -1)
    expected = a.sqrt() * pixels[3:] + (1 - a).sqrt() * noise.reshape(2, -1)
    assert torch.equal(changed["pixel_values"][3:], expected)
    assert meta["alpha_bar"] == float(a) and meta["query_patch_rows"] == 2


def test_shot_extension_uses_nested_frozen_supports():
    m = api()
    recipe = {
        "contexts": {"query": list("abcdefgh")},
        "spec": {"shot_counts": {"zero_query_noise": 0, "two_ordinary": 2, "two_query_noise": 2}},
    }
    assert m.support_ids(recipe, "zero_query_noise", "query") == []
    assert m.support_ids(recipe, "two_ordinary", "query") == ["a", "b"]
    assert m.support_ids(recipe, "two_query_noise", "query") == ["a", "b"]
    recipe["spec"] = {}
    assert m.support_ids(recipe, "query_noise", "query") == list("abcdefgh")


def test_cached_query_noise_reuses_exact_input_and_rejects_other_query(tmp_path, monkeypatch):
    import pytest

    m = api()
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(patch_size=2, temporal_patch_size=2)
    )
    pixels = (
        torch.arange(24, dtype=torch.float32)
        .reshape(2, 3, 1, 2, 2)
        .expand(-1, -1, 2, -1, -1)
        .reshape(2, -1)
    )
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "image_grid_thw": torch.tensor([[1, 1, 2]]),
        "pixel_values": pixels,
    }
    first, meta = m.cached_query_noise(batch, "validation/q", processor, tmp_path)

    def unexpected(*args):
        raise AssertionError("cached query noise must not be regenerated")

    monkeypatch.setattr(m, "query_noise", unexpected)
    with_support = dict(
        batch,
        image_grid_thw=torch.tensor([[1, 1, 2], [1, 1, 2]]),
        pixel_values=torch.cat([pixels + 99, pixels]),
    )
    repeated, other_meta = m.cached_query_noise(with_support, "validation/q", processor, tmp_path)
    assert torch.equal(first["pixel_values"], repeated["pixel_values"][-2:])
    assert torch.equal(repeated["pixel_values"][:2], pixels + 99)
    assert meta["cache_sha256"] == other_meta["cache_sha256"]
    with pytest.raises(AssertionError):
        m.cached_query_noise(
            dict(batch, pixel_values=pixels + 1), "validation/q", processor, tmp_path
        )


def test_full_shot_extension_keeps_eight_supports():
    m = api()
    recipe = {
        "contexts": {"query": list("abcdefgh")},
        "spec": {"shot_counts": {"eight_query_noise": 8}},
    }
    assert m.support_ids(recipe, "eight_query_noise", "query") == list("abcdefgh")
