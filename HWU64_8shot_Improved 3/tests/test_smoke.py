"""Offline smoke tests for core HWU64 selection and metrics."""
import numpy as np
from src.pipeline import query_features, choose_demos, metric_row, switch_analysis, paired_comparisons


def test_feature_vector():
    candidates = [
        {"similarity": 0.9, "label_id": 0},
        {"similarity": 0.8, "label_id": 0},
        {"similarity": 0.7, "label_id": 1},
        {"similarity": 0.6, "label_id": 1},
    ]
    f = query_features(candidates)
    assert f["max_similarity"] == 0.9
    assert abs(f["mean_similarity"] - 0.75) < 1e-9
    assert f["entropy_bits"] == 1.0
    assert f["top_label_vote"] == 0.5


def test_selection_strategies_return_k():
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(10, 6)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    q = vectors[0]
    sims = vectors @ q
    candidates = [{"pool_index": i, "similarity": float(sims[i]), "example_id": f"id_{i:02d}"}
                  for i in range(10)]
    for strategy in ("similar", "diverse", "balanced"):
        chosen = choose_demos(strategy, candidates, vectors, 8, 0.5)
        assert len(chosen) == 8
        assert len({x["example_id"] for x in chosen}) == 8


def test_metric_and_switch():
    labels = ["a", "b", "c"]
    result = metric_row(labels, ["a", "x", "c"], ["a", "b", "c"])
    assert result["correct"] == 2
    assert result["invalid_predictions"] == 1
    s = switch_analysis(labels, ["a", "b", "x"], ["a", "x", "c"], ["similar", "diverse", "balanced"])
    assert s["beneficial"] == 1 and s["harmful"] == 1


if __name__ == "__main__":
    test_feature_vector()
    test_selection_strategies_return_k()
    test_metric_and_switch()
    print("3 offline smoke tests passed")


def test_stratified_hwuintent_splits():
    import pandas as pd
    from src.pipeline import make_splits

    labels = [f"intent_{i}" for i in range(64)]
    rows = []
    for label_id, label_name in enumerate(labels):
        for j in range(20):
            rows.append({"example_id": f"train_{label_id}_{j}", "text": f"utterance {label_id} {j}",
                         "label": label_id, "label_name": label_name, "source_split": "train"})
    train = pd.DataFrame(rows)
    test_rows = []
    for label_id, label_name in enumerate(labels):
        for j in range(3):
            test_rows.append({"example_id": f"test_{label_id}_{j}", "text": f"test {label_id} {j}",
                              "label": label_id, "label_name": label_name, "source_split": "test"})
    splits = make_splits(train, pd.DataFrame(test_rows), 42)
    assert [len(splits[k]) for k in ("demonstration_pool", "meta_training", "validation", "test")] == [896, 256, 128, 192]
    assert all(splits[k]["label"].nunique() == 64 for k in splits)
    ids = [set(x["example_id"]) for x in splits.values()]
    assert all(not ids[i] & ids[j] for i in range(3) for j in range(i + 1, 4))
