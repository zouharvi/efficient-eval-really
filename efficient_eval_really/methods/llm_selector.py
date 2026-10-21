import numpy as np

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def llm_selector_budgets(
    data: Data,
    budgets: Budgets,
    temperature: float = 0.05,
) -> ModelScoresAtBudget:
    """LLM Selector: Active Model Selection for LLMs via Expected Entropy Minimization.

    Paper: "Active Model Selection for Large Language Models" (ICLR 2026 / arXiv:2510.09418)

    Selects queries adaptively to minimize the expected conditional entropy of the
    best model identity. Queries with high score variance among top candidate models
    are prioritized.
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    S = np.array(
        [[item["scores_metrics"][m][metric] for m in models] for item in data],
        dtype=float,
    ).T  # (n_models, n_items)
    costs = np.array([item["cost"] for item in data], dtype=float)

    # Initial query selection: item with maximum overall variance
    selected_items: list[int] = []
    available = set(range(n_items))

    first_item = int(np.argmax(np.var(S, axis=0) / (costs + 1e-6)))
    selected_items.append(first_item)
    available.remove(first_item)

    current_cost = float(costs[first_item] * n_models)
    results_map = {}
    sorted_budgets = sorted(set(budgets))

    for b in sorted_budgets:
        while current_cost < b and available:
            # Current mean estimates for each model on selected items
            S_obs = S[:, selected_items]  # (n_models, n_selected)
            mu = np.mean(S_obs, axis=1)

            # Posterior probability of each model being the top model
            scaled_mu = (mu - np.max(mu)) / max(1e-4, temperature)
            exp_mu = np.exp(np.clip(scaled_mu, -50, 0))
            p_best = exp_mu / np.sum(exp_mu)

            # Score candidates by expected entropy reduction (variance under p_best)
            cand_list = list(available)
            S_cand = S[:, cand_list]  # (n_models, n_cand)
            cand_mean = np.sum(p_best[:, None] * S_cand, axis=0)
            cand_var = np.sum(p_best[:, None] * (S_cand - cand_mean) ** 2, axis=0)

            # Acquisition score normalized by cost
            scores = cand_var / (costs[cand_list] + 1e-6)
            best_cand_idx = int(np.argmax(scores))
            picked_item = cand_list[best_cand_idx]

            selected_items.append(picked_item)
            available.remove(picked_item)
            current_cost += float(costs[picked_item] * n_models)

        results_map[b] = {
            m: [data[i]["scores"][m] for i in selected_items] for m in models
        }

    return [results_map[b] for b in budgets]


def select_llm_budgets(
    data: Data,
    budgets: Budgets,
    temperature: float = 0.05,
) -> ModelScoresAtBudget:
    """SELECT-LLM: Large Language Model Selection with Limited Annotations.

    Paper: "Large Language Model Selection with Limited Annotations" (arXiv:2605.24981)

    Actively selects queries that maximize the expected information gain across
    candidate models, weighted by posterior model belief and pairwise discrepancies.
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    S = np.array(
        [[item["scores_metrics"][m][metric] for m in models] for item in data],
        dtype=float,
    ).T  # (n_models, n_items)
    costs = np.array([item["cost"] for item in data], dtype=float)

    selected_items: list[int] = []
    available = set(range(n_items))

    first_item = int(np.argmax(np.var(S, axis=0) / (costs + 1e-6)))
    selected_items.append(first_item)
    available.remove(first_item)

    current_cost = float(costs[first_item] * n_models)
    results_map = {}
    sorted_budgets = sorted(set(budgets))

    for b in sorted_budgets:
        while current_cost < b and available:
            S_obs = S[:, selected_items]
            mu = np.mean(S_obs, axis=1)

            scaled_mu = (mu - np.max(mu)) / max(1e-4, temperature)
            exp_mu = np.exp(np.clip(scaled_mu, -50, 0))
            p = exp_mu / np.sum(exp_mu)

            # Pairwise discrepancy weighted by p_i * p_j
            cand_list = list(available)
            S_cand = S[:, cand_list]  # (n_models, n_cand)

            # Vectorized pairwise absolute difference:
            # D_cand[c] = sum_{i < j} p[i] p[j] |S[i, c] - S[j, c]|
            diffs = np.abs(S_cand[:, None, :] - S_cand[None, :, :])  # (m, m, c)
            weights = p[:, None, None] * p[None, :, None]  # (m, m, 1)
            disc = np.sum(weights * diffs, axis=(0, 1))

            scores = disc / (costs[cand_list] + 1e-6)
            best_cand_idx = int(np.argmax(scores))
            picked_item = cand_list[best_cand_idx]

            selected_items.append(picked_item)
            available.remove(picked_item)
            current_cost += float(costs[picked_item] * n_models)

        results_map[b] = {
            m: [data[i]["scores"][m] for i in selected_items] for m in models
        }

    return [results_map[b] for b in budgets]
