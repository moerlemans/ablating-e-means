import pytest
import torch

import emeans.results as results
from emeans.metrics import dead_clusters, imbalance, mse


def test_mse_is_exact_fp64():
    x = torch.tensor([[0.0, 0.0], [2.0, 0.0], [10.0, 1.0]])
    c = torch.tensor([[1.0, 0.0], [10.0, 0.0]])
    assert mse(x, c) == pytest.approx((1 + 1 + 1) / 3)


def test_dead_clusters_and_imbalance():
    idx = torch.tensor([0] * 12 + [1] * 12 + [2] * 3)
    assert dead_clusters(idx, 4) == 2  # cluster 2 has 3 points, cluster 3 none
    p = torch.tensor([12, 12, 3, 0]) / 27
    assert imbalance(idx, 4) == pytest.approx(float(4 * (p * p).sum()))
    assert imbalance(torch.arange(4), 4) == pytest.approx(1.0)


def test_results_csv_resumes_and_rejects_other_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS_DIR", tmp_path)
    a = results.ResultsCSV("t", ["method", "seed"])
    a.append([{"method": "m", "seed": 0, "mse": 1.0}])
    b = results.ResultsCSV("t", ["method", "seed"])  # a restart
    assert b.has({"method": "m", "seed": 0}) and not b.has({"method": "m", "seed": 1})
    b.append([{"method": "m", "seed": 1, "mse": 2.0}])
    with pytest.raises(ValueError):
        b.append([{"method": "m", "seed": 2, "dead": 0, "mse": 2.0}])
    assert (tmp_path / "t.csv").read_text().splitlines() == ["method,seed,mse", "m,0,1.0", "m,1,2.0"]
