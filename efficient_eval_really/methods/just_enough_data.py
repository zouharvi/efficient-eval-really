import numpy as np

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def just_enough_data_budgets(
    data: Data,
    budgets: Budgets,
    batch_size: int = 5,
    z_crit: float = 1.96,
    seed: int = 42,
) -> ModelScoresAtBudget:
    """Group Sequential Testing with Adaptive Early Stopping.

    Algorithm from ICLR (2026):
    'Stop Guessing When to Stop Testing: Efficient Model Evaluation with Just Enough Data'

    Evaluates models in batches of items. At each interim stage, checks pairwise
    statistical separation using confidence bounds. Models that achieve confident
    separation are frozen early, dynamically allocating remaining budget to
    distinguish models with close performance.
    """
    models = list(data[0]["scores"].keys())
    n_items = len(data)

    results: ModelScoresAtBudget = []
    for budget in budgets:
        model_scores = {m: [] for m in models}
        item_indices = {m: 0 for m in models}
        active_models = set(models)
        cost = 0.0

        # Initial warm-up batch for all models
        init_size = min(max(5, batch_size), n_items)
        for m in models:
            for i in range(init_size):
                model_scores[m].append(float(data[i]["scores"][m]))
                cost += data[i]["cost"]
            item_indices[m] = init_size

        while active_models and cost + len(active_models) * batch_size <= budget:
            means = {m: float(np.mean(model_scores[m])) for m in models}
            stds = {m: float(np.std(model_scores[m], ddof=1)) if len(model_scores[m]) > 1 else 0.1 for m in models}
            se = {m: stds[m] / np.sqrt(len(model_scores[m])) for m in models}

            # Freeze models that are statistically separated from all others or exhausted
            frozen = set()
            for m in list(active_models):
                separated = True
                for other in models:
                    if m == other:
                        continue
                    diff = abs(means[m] - means[other])
                    pooled_se = float(np.sqrt(se[m] ** 2 + se[other] ** 2))
                    if pooled_se > 1e-8 and diff < z_crit * pooled_se:
                        separated = False
                        break
                if separated or item_indices[m] >= n_items:
                    frozen.add(m)

            if len(frozen) == len(active_models):
                break

            active_models.difference_update(frozen)
            if not active_models:
                break

            # Evaluate next batch for active models
            for m in list(active_models):
                curr = item_indices[m]
                next_items = min(batch_size, n_items - curr)
                if next_items <= 0 or cost + next_items > budget:
                    active_models.remove(m)
                    continue
                for i in range(curr, curr + next_items):
                    model_scores[m].append(float(data[i]["scores"][m]))
                    cost += data[i]["cost"]
                item_indices[m] += next_items

        results.append(model_scores)

    return results
