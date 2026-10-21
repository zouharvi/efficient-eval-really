import math

import numpy as np

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def syrs_bandit_budgets(data: Data, budgets: Budgets, seed: int = 42) -> ModelScoresAtBudget:
    """Synchronized Successive Rejects (SySRs).

    Algorithm 1 from Lyu et al. (2026):
    'Cutting LLM Evaluation Costs with SySRs: A Bandit Algorithm that Provably Exploits Model Similarity'

    Splits the evaluation budget across K - 1 phases, evaluating remaining active
    models on synchronized batches of queries drawn uniformly without replacement,
    eliminating the worst arm at each phase.
    """
    rng = np.random.default_rng(seed)
    models = list(data[0]["scores"].keys())
    n_models = len(models)
    n_items = len(data)

    results: ModelScoresAtBudget = []
    log_K = 0.5 + sum(1.0 / i for i in range(2, n_models + 1)) if n_models > 1 else 1.0

    for budget in budgets:
        if n_models <= 1:
            k_items = min(budget, n_items)
            results.append({m: [data[i]["scores"][m] for i in range(k_items)] for m in models})
            continue

        active_models = list(models)
        evaluated_items = {m: [] for m in models}
        item_pool = list(range(n_items))
        rng.shuffle(item_pool)
        pool_idx = 0
        n_prev = 0

        for k in range(1, n_models):
            # Formula: nk = ceil((1 / log_K) * (budget - n_models) / (n_models + 1 - k))
            if budget > n_models:
                nk = math.ceil((1.0 / log_K) * (budget - n_models) / (n_models + 1 - k))
            else:
                nk = 1
            nk = max(n_prev + 1, min(n_items, nk))
            delta_n = nk - n_prev

            if pool_idx + delta_n > n_items:
                delta_n = max(0, n_items - pool_idx)

            phase_items = item_pool[pool_idx : pool_idx + delta_n]
            pool_idx += delta_n
            n_prev = nk

            for m in active_models:
                evaluated_items[m].extend(phase_items)

            if len(active_models) > 1 and phase_items:
                means = {m: float(np.mean([data[idx]["scores"][m] for idx in evaluated_items[m]])) for m in active_models}
                worst_m = min(active_models, key=lambda m: means[m])
                active_models.remove(worst_m)

        model_scores = {}
        for m in models:
            if evaluated_items[m]:
                model_scores[m] = [float(data[idx]["scores"][m]) for idx in evaluated_items[m]]
            else:
                model_scores[m] = [float(data[0]["scores"][m])]
        results.append(model_scores)

    return results
