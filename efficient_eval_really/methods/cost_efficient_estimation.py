import numpy as np
import scipy.special

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def cost_efficient_estimation_budgets(
    data: Data,
    budgets: Budgets,
    d: int = 5,
    seed: int = 0,
    **kwargs,
) -> ModelScoresAtBudget:
    """Cost-Efficient Estimation of General Abilities Across Benchmarks.

    Algorithm based on:
    'Cost-Efficient Estimation of General Abilities Across Benchmarks'

    Combines a multidimensional latent factor model with adaptive item selection driven
    by Optimal Experimental Design (OED). Specifically:
    1. Extracts latent item loadings K and difficulties delta via SVD/PCA on proxy metrics.
    2. Builds the target information matrix M.
    3. For each model, adaptively routes items sequentially by maximizing V-optimality
       discounted by item evaluation cost (s_x / cost_x).
    4. Updates model ability theta and posterior covariance C via Woodbury correction.
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    S = np.array([[item["scores_metrics"][m][metric] for m in models] for item in data], dtype=float)
    costs = np.array([item["cost"] for item in data], dtype=float)

    item_means = np.clip(np.mean(S, axis=1), 0.01, 0.99)
    delta = -scipy.special.logit(item_means)
    alpha = np.ones(n_items, dtype=float)

    # Latent factor loadings K in R^(n_items, dim)
    dim = max(1, min(d, n_models, n_items))
    S_centered = S - item_means[:, None]
    U, s, _ = np.linalg.svd(S_centered, full_matrices=False)
    K = U[:, :dim] * (s[:dim] / np.sqrt(max(1, n_models)))

    # Target matrix M = sum_t alpha_t^2 K_t K_t^T
    K_scaled = K * alpha[:, None]
    M = K_scaled.T @ K_scaled

    sorted_budgets = sorted(set(budgets))
    results_map = {}

    for budget in sorted_budgets:
        k = max(1, min(budget // n_models, n_items))
        if k >= n_items:
            results_map[budget] = {m: [item["scores"][m] for item in data] for m in models}
            continue

        model_scores = {}
        for m in models:
            theta = np.zeros(dim, dtype=float)
            C = np.eye(dim, dtype=float)
            administered: list[int] = []
            available = set(range(n_items))

            for _ in range(k):
                cand = np.array(list(available), dtype=int)
                K_cand = K[cand]
                d_cand = delta[cand]
                a_cand = alpha[cand]
                cost_cand = costs[cand]

                logits = a_cand * (K_cand @ theta - d_cand)
                p = scipy.special.expit(np.clip(logits, -20.0, 20.0))
                w = np.maximum(1e-6, p * (1.0 - p))

                c_mat = K_cand @ C
                c_M_c = np.sum(c_mat * (c_mat @ M), axis=1)
                K_c = np.sum(K_cand * c_mat, axis=1)
                denom = 1.0 + w * (a_cand**2) * K_c

                v_opt = (w * (a_cand**2) * c_M_c) / np.maximum(1e-12, denom)
                scores_cand = v_opt / np.maximum(1e-6, cost_cand)

                best_idx_in_cand = int(np.argmax(scores_cand))
                best_item = cand[best_idx_in_cand]

                administered.append(best_item)
                available.remove(best_item)

                # Woodbury / Sherman-Morrison update of covariance C
                c_x = c_mat[best_idx_in_cand]
                w_x = w[best_idx_in_cand]
                a_x = a_cand[best_idx_in_cand]
                denom_x = denom[best_idx_in_cand]
                C = C - (w_x * (a_x**2) / denom_x) * np.outer(c_x, c_x)

                # Update theta
                y_obs = float(data[best_item]["scores"][m])
                p_x = p[best_idx_in_cand]
                K_x = K[best_item]
                theta = theta + C @ (a_x * (y_obs - p_x) * K_x)

            model_scores[m] = [data[i]["scores"][m] for i in administered]

        results_map[budget] = model_scores

    return [results_map[b] for b in budgets]
