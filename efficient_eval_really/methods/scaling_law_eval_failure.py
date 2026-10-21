import numpy as np
import scipy.special

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def scaling_law_eval_failure_budgets(
    data: Data,
    budgets: Budgets,
    n_iter: int = 25,
    seed: int = 0,
    **kwargs,
) -> ModelScoresAtBudget:
    """The Scaling Law of Evaluation Failure: IRT Recovery under Sparsity.

    Paper: "The Scaling Law of Evaluation Failure: Why Simple Averaging Collapses
    Under Data Sparsity and Item Difficulty Gaps, and How Item Response Theory
    Recovers Ground Truth Across Domains" (Sun et al., 2026)

    When data is sparse and items vary in difficulty, naive averages fail.
    This method evaluates models under data sparsity and fits a joint IRT model
    (alternating MLE/ALS for abilities theta and difficulties b) to recover
    calibrated, debiased abilities across all items.
    """
    rng = np.random.default_rng(seed)
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    S = np.array(
        [[item["scores_metrics"][m][metric] for m in models] for item in data],
        dtype=float,
    ).T  # (n_models, n_items)
    costs = np.array([item["cost"] for item in data], dtype=float)

    observed = np.zeros((n_models, n_items), dtype=bool)
    current_cost = 0.0

    # Ensure each model starts with at least 1 random observation
    for m in range(n_models):
        j_init = int(rng.integers(n_items))
        observed[m, j_init] = True
        current_cost += float(costs[j_init])

    results_map = {}
    sorted_budgets = sorted(set(budgets))

    for b in sorted_budgets:
        # Incrementally acquire observations up to budget b
        while current_cost < b:
            unobs = np.where(~observed)
            if len(unobs[0]) == 0:
                break
            idx = rng.integers(len(unobs[0]))
            m_pick = unobs[0][idx]
            j_pick = unobs[1][idx]
            observed[m_pick, j_pick] = True
            current_cost += float(costs[j_pick])

        # If fully observed, return ground truth
        if np.all(observed):
            results_map[b] = {m: [item["scores"][m] for item in data] for m in models}
            continue

        # Fit 1PL Rasch IRT: P(Y_mj = 1) = sigmoid(theta_m - b_j)
        # Initialize abilities and item difficulties
        theta = np.zeros(n_models, dtype=float)
        b_diff = np.zeros(n_items, dtype=float)

        # Baseline empirical item difficulty
        for j in range(n_items):
            col_obs = observed[:, j]
            if np.any(col_obs):
                p_mean = np.clip(np.mean(S[col_obs, j]), 0.05, 0.95)
                b_diff[j] = -scipy.special.logit(p_mean)

        # Alternating updates for theta and b
        for _ in range(n_iter):
            # Update theta_m
            for m in range(n_models):
                row_obs = observed[m]
                if not np.any(row_obs):
                    continue
                # Gradient step on log-likelihood: sum (y_mj - p_mj)
                logits = theta[m] - b_diff[row_obs]
                p = scipy.special.expit(logits)
                grad = np.sum(S[m, row_obs] - p)
                hess = -np.sum(p * (1.0 - p)) - 0.1
                theta[m] -= grad / min(-1e-4, hess)
                theta[m] = float(np.clip(theta[m], -5.0, 5.0))

            # Update b_j
            for j in range(n_items):
                col_obs = observed[:, j]
                if not np.any(col_obs):
                    continue
                logits = theta[col_obs] - b_diff[j]
                p = scipy.special.expit(logits)
                grad = np.sum(p - S[col_obs, j])
                hess = -np.sum(p * (1.0 - p)) - 0.1
                b_diff[j] -= grad / min(-1e-4, hess)
                b_diff[j] = float(np.clip(b_diff[j], -5.0, 5.0))

        # Reconstructed predictions for each model across all items
        model_scores = {}
        for m_idx, m in enumerate(models):
            logits = theta[m_idx] - b_diff
            p_pred = scipy.special.expit(logits)
            # Retain true observed scores where available
            p_pred[observed[m_idx]] = S[m_idx, observed[m_idx]]
            model_scores[m] = list(np.clip(p_pred, 0.0, 1.0))

        results_map[b] = model_scores

    return [results_map[b] for b in budgets]
