import numpy as np

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget
from efficient_eval_really.methods.irt_chatbots import _fit_2pl_irt


def atlas_budgets(data: Data, budgets: Budgets, seed: int = 0, **kwargs) -> ModelScoresAtBudget:
    """ATLAS: Adaptive Testing for Language Models via Item Response Theory (CAT).

    Algorithm from Li et al. (2025):
    'ATLAS: Adaptive Testing for Language Models'

    Fits a 2-Parameter Logistic (2PL) IRT model on proxy metrics to obtain item
    discrimination (a) and difficulty (b). Then, for each model, dynamically selects
    evaluation items that maximize Fisher Information at the model's currently estimated
    latent ability (theta), iteratively refining theta as responses are observed.
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    S = np.array([[item["scores_metrics"][m][metric] for m in models] for item in data], dtype=float)
    mean_val = float(np.mean(S))
    Y = (S >= mean_val).astype(float).T  # (n_models, n_items)

    a, b, _ = _fit_2pl_irt(Y)
    a = np.clip(np.nan_to_num(a, nan=1.0), 0.1, 5.0)
    b = np.clip(np.nan_to_num(b, nan=0.0), -5.0, 5.0)

    results: ModelScoresAtBudget = []
    for budget in budgets:
        k = max(1, min(budget // n_models, n_items))

        if k >= n_items:
            results.append({m: [float(item["scores"][m]) for item in data] for m in models})
            continue

        model_scores = {}
        for m in models:
            theta = 0.0
            administered: list[int] = []
            available = set(range(n_items))

            for _ in range(k):
                cand = np.array(list(available), dtype=int)
                # Fisher information for candidate items: a^2 * p * (1 - p)
                logits = a[cand] * (theta - b[cand])
                p = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
                fisher = (a[cand] ** 2) * p * (1.0 - p)

                best_cand_idx = int(cand[np.argmax(fisher)])
                administered.append(best_cand_idx)
                available.remove(best_cand_idx)

                # Newton-Raphson update of ability theta based on observed scores
                admin_arr = np.array(administered, dtype=int)
                l_logits = a[admin_arr] * (theta - b[admin_arr])
                l_p = 1.0 / (1.0 + np.exp(-np.clip(l_logits, -20.0, 20.0)))
                l_fisher = (a[admin_arr] ** 2) * l_p * (1.0 - l_p)
                observed_y = np.array([data[idx]["scores"][m] for idx in admin_arr], dtype=float)
                grad = np.sum(a[admin_arr] * (observed_y - l_p))
                hess = np.sum(l_fisher) + 0.1
                theta = float(np.clip(theta + grad / hess, -4.0, 4.0))

            model_scores[m] = [float(data[idx]["scores"][m]) for idx in administered]

        results.append(model_scores)

    return results
