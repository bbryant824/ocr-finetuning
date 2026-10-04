"""Guard the predeclared decoder decision against cap-only apparent improvements."""

import runpy

import pytest


@pytest.mark.parametrize(
    "candidate_page,candidate_raw,chosen",
    [(0.6, 0.6, 1.0), (0.6, 0.9, 1.1), (0.8, 0.8, 1.1)],
)
def test_decoder_requires_improvement_in_both_score_views(candidate_page, candidate_raw, chosen):
    decide = runpy.run_path("experiments/exp015_decoding.py")["decide"]
    results = {
        penalty: {
            str(shots): {"metrics": {"cer": page}, "raw_line": {"cer": raw}} for shots in (0, 8, 24)
        }
        for penalty, page, raw in [("1.1", 0.8, 0.8), ("1.0", candidate_page, candidate_raw)]
    }
    assert decide(results)["repetition_penalty"] == chosen
