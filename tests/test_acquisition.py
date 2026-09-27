"""A greedy diversity batch must update distances after each selected page."""

from active_ocr.acquisition_pipeline import kcenter_greedy


def test_kcenter_recomputes_coverage_between_choices():
    embeddings = {
        "seed": (1.0, 0.0),
        "opposite": (-1.0, 0.0),
        "near-opposite": (-0.8, 0.6),
        "uncovered": (0.0, -1.0),
    }
    assert kcenter_greedy(
        embeddings,
        labelled=["seed"],
        candidates=["opposite", "near-opposite", "uncovered"],
        count=2,
    ) == ("opposite", "uncovered")


def test_kcenter_starts_at_centroid_and_breaks_ties_by_id():
    embeddings = {"b": (1.0, 0.0), "a": (1.0, 0.0), "far": (-1.0, 0.0)}
    assert kcenter_greedy(embeddings, [], list(embeddings), 2) == ("a", "far")
