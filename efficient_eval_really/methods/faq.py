import numpy as np
import scipy.special

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def faq_budgets(
    data: Data,
    budgets: Budgets,
    k_factors: int = 5,
    tau_mix: float = 0.1,
    rho: float = 0.5,
) -> ModelScoresAtBudget:
    """Factorized Active Querying (FAQ) for LLM evaluation.

    Algorithm from Wu et al. (Stanford):
    'FAQ: Efficient Evaluation of LLM Performance with Statistical Guarantees'

    1. Factor model: extracts item factors V and model prior from historical/proxy evaluations.
    2. Hybrid sampling policy: combines variance reduction (oracle item difficulty variance)
       and active learning (uncertainty reduction on model capability vector u) with tempering.
    3. Sequentially updates the model factor posterior using Laplace online updates.
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    S = np.array([[item["scores_metrics"][m][metric] for m in models] for item in data], dtype=float)

    # Factorize proxy responses into item factors V and model factors U
    k = max(1, min(k_factors, n_models, n_items))
    item_means = np.clip(np.mean(S, axis=1), 0.01, 0.99)
    S_centered = S - item_means[:, None]

    U_svd, s_svd, Vt_svd = np.linalg.svd(S_centered, full_matrices=False)
    V = U_svd[:, :k] * np.sqrt(s_svd[:k])
    U = (Vt_svd[:k].T) * np.sqrt(s_svd[:k])

    u_prior = np.mean(U, axis=0)
    cov_u = np.cov(U, rowvar=False) if n_models > 1 else np.eye(k)
    cov_u = np.atleast_2d(cov_u) + 0.1 * np.eye(k)

    results_map = {}
    sorted_budgets = sorted(set(budgets))

    for budget in sorted_budgets:
        n_b = max(1, min(budget // n_models, n_items))
        if n_b >= n_items:
            results_map[budget] = {m: [item["scores"][m] for item in data] for m in models}
            continue

        model_scores = {}
        for m in models:
            u = u_prior.copy()
            cov = cov_u.copy()

            administered: list[int] = []
            available = set(range(n_items))

            for t in range(n_b):
                logits = V @ u
                p_hat = scipy.special.expit(np.clip(logits, -20.0, 20.0))
                var_hat = np.maximum(1e-6, p_hat * (1.0 - p_hat))

                # Variance reduction score
                s_o = np.sqrt(var_hat)

                # Active learning score
                v_cov_v = np.maximum(0.0, np.sum(V * (V @ cov), axis=1))
                s_a = np.sqrt(v_cov_v) * var_hat

                # Normalize scores
                h_o = s_o / max(1e-12, float(np.sum(s_o)))
                h_a = s_a / max(1e-12, float(np.sum(s_a)))

                # Hybrid mixing schedule
                alpha_t = max(0.0, 1.0 - (t + 1) / (rho * n_b))
                h_cat = ((1.0 - alpha_t) * h_o + alpha_t * h_a) ** 0.5
                sum_h = max(1e-12, float(np.sum(h_cat)))
                q_t = (tau_mix / n_items) + (1.0 - tau_mix) * (h_cat / sum_h)

                cand_list = list(available)
                q_cand = q_t[cand_list]
                best_item = cand_list[int(np.argmax(q_cand))]

                administered.append(best_item)
                available.remove(best_item)

                z_obs = float(data[best_item]["scores"][m])
                p_obs = p_hat[best_item]
                w_obs = var_hat[best_item]
                v_item = V[best_item]

                # Online Laplace update via Sherman-Morrison
                c_vec = cov @ v_item
                denom = 1.0 + w_obs * float(np.dot(v_item, c_vec))
                cov = cov - (w_obs / max(1e-12, denom)) * np.outer(c_vec, c_vec)
                u = u + cov @ (v_item * (z_obs - p_obs))

            model_scores[m] = [data[i]["scores"][m] for i in administered]

        results_map[budget] = model_scores

    return [results_map[b] for b in budgets]
