import numpy as np

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def efficient_benchmarking_apis_budgets(
    data: Data,
    budgets: Budgets,
    seed: int = 42,
) -> ModelScoresAtBudget:
    """Thompson Sampling Bandit for Efficient Benchmarking.

    Algorithm from (EACL 2017):
    'Efficient Benchmarking of NLP APIs using Multi-armed Bandits'

    Maintains a Gaussian posterior distribution over the expected evaluation performance
    of each candidate system/model. At each decision step, samples from the model posteriors
    (Thompson Sampling) to balance exploration and exploitation, dynamically allocating more
    evaluation budget to top-performing candidates.
    """
    rng = np.random.default_rng(seed)
    models = list(data[0]["scores"].keys())
    n_items = len(data)

    results: ModelScoresAtBudget = []
    for budget in budgets:
        model_scores = {m: [] for m in models}
        item_indices = {m: 0 for m in models}
        cost = 0.0

        # Initial warm-up: 2 items per model
        warmup = min(2, n_items)
        for m in models:
            for i in range(warmup):
                model_scores[m].append(float(data[i]["scores"][m]))
                cost += data[i]["cost"]
            item_indices[m] = warmup

        while cost < budget:
            available_models = [m for m in models if item_indices[m] < n_items]
            if not available_models:
                break

            samples = {}
            for m in available_models:
                scores = model_scores[m]
                mean = np.mean(scores)
                std = np.std(scores, ddof=1) if len(scores) > 1 else 0.2
                sem = max(1e-4, std / np.sqrt(len(scores)))
                samples[m] = rng.normal(loc=mean, scale=sem)

            best_model = max(available_models, key=lambda m: samples[m])
            idx = item_indices[best_model]
            model_scores[best_model].append(float(data[idx]["scores"][best_model]))
            item_indices[best_model] += 1
            cost += data[idx]["cost"]

        results.append(model_scores)

    return results
