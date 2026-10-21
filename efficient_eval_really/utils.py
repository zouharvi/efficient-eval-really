import numpy as np
import scipy.stats

from efficient_eval_really.methods import ModelScores, ModelScoresSubset


def meta_evaluate_scores_subset(model_scores: ModelScoresSubset, model_scores_all: ModelScores) -> dict[str, float]:
    """Meta-evaluate subset evaluation scores against ground-truth full-benchmark scores.

    Computes:
      - corr_tau: Kendall tau correlation of model scores.
      - corr_wtau: Weighted Kendall tau focusing on top-ranked models.
      - model_mae: Average absolute error in estimating each model's benchmark score:
          mean_{m in M} |mean(subset_m) - mean(all_m)|
      - item_mae: Average absolute error across individual item predictions:
          mean_{m in M} mean_{x in X} |pred(m, x) - true(m, x)|
    """
    models = list(model_scores.keys())
    model_avg = [float(np.mean(model_scores[model])) for model in models]
    model_avg_true = [float(np.mean(model_scores_all[model])) for model in models]
    model_mae = float(np.mean(np.abs(np.array(model_avg) - np.array(model_avg_true))))

    # item_mae: per-item error across models
    item_errors = []
    for model, m_avg in zip(models, model_avg):
        if len(model_scores[model]) == len(model_scores_all[model]):
            item_errors.append(np.mean(np.abs(np.array(model_scores[model]) - np.array(model_scores_all[model]))))
        else:
            item_errors.append(np.mean(np.abs(m_avg - np.array(model_scores_all[model]))))
    item_mae = float(np.mean(item_errors))

    return {
        "corr_tau": scipy.stats.kendalltau(model_avg, model_avg_true).correlation,  # type: ignore
        "corr_wtau": scipy.stats.weightedtau(model_avg, model_avg_true, weigher=lambda rank: 1 / (rank + 1)).correlation,  # type: ignore
        "model_mae": model_mae,
        "item_mae": item_mae,
    }
