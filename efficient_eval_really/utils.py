import numpy as np
from efficient_eval_really.methods import ModelScores, ModelScoresSubset
import scipy.stats

def meta_evaluate_scores_subset(model_scores: ModelScoresSubset, model_scores_all: ModelScores) -> float:
    # for now just simple spearman ranking
    models = list(model_scores.keys())
    model_avg = [float(np.mean(model_scores[model])) for model in models]
    model_avg_true = [float(np.mean(model_scores_all[model])) for model in models]
    return {
        "corr_spearman": scipy.stats.spearmanr(model_avg, model_avg_true).correlation # type: ignore
    }