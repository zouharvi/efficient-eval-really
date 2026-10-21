import numpy as np
import scipy.special

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def fluid_benchmarking_budgets(
    data: Data,
    budgets: Budgets,
    seed: int = 0,
    **kwargs,
) -> ModelScoresAtBudget:
    """Fluid Language Model Benchmarking: Computerized Adaptive Testing via 2PL IRT.

    Paper: "Fluid Language Model Benchmarking" (Li et al., AllenAI, 2025)

    Dynamically routes evaluation items to each model by matching item difficulty
    and discrimination to the model's evolving latent ability theta.
    Maximizes Fisher Information at each step:
        I_j(theta) = a_j^2 * P_j(theta) * (1 - P_j(theta))
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

    # Estimate 2PL item parameters (a_j: discrimination, b_j: difficulty)
    item_means = np.clip(np.mean(S, axis=0), 0.05, 0.95)
    # Logit difficulty: lower mean = harder = higher b
    b_params = -scipy.special.logit(item_means)
    # Discrimination: item-rest correlation or std
    item_stds = np.std(S, axis=0)
    a_params = np.clip(item_stds * 4.0, 0.5, 3.0)

    results_map = {}
    sorted_budgets = sorted(set(budgets))

    for b in sorted_budgets:
        n_per_model = max(1, min(int(b // (n_models * np.mean(costs))), n_items))
        if n_per_model >= n_items:
            results_map[b] = {m: [item["scores"][m] for item in data] for m in models}
            continue

        model_scores = {}
        for m_idx, m in enumerate(models):
            # Start at neutral ability
            theta = 0.0
            visited: list[int] = []
            unvisited = set(range(n_items))

            for _ in range(n_per_model):
                if not unvisited:
                    break

                cand_arr = np.array(list(unvisited))
                # 2PL probability: P(correct | theta)
                logits = a_params[cand_arr] * (theta - b_params[cand_arr])
                p_theta = scipy.special.expit(logits)
                # Fisher information
                fisher_info = (a_params[cand_arr] ** 2) * p_theta * (1.0 - p_theta)
                # Cost-adjusted information
                eff_info = fisher_info / (costs[cand_arr] + 1e-6)

                best_c = int(np.argmax(eff_info))
                picked_item = cand_arr[best_c]

                visited.append(picked_item)
                unvisited.remove(picked_item)

                # Online Bayesian update on theta:
                # theta_new = theta + step * a_j * (y_j - P_j(theta))
                y_obs = S[m_idx, picked_item]
                step = 1.0 / (1.0 + 0.1 * len(visited))
                theta += step * a_params[picked_item] * (y_obs - p_theta[best_c])
                theta = float(np.clip(theta, -4.0, 4.0))

            model_scores[m] = [data[j]["scores"][m] for j in visited]

        results_map[b] = model_scores

    return [results_map[b] for b in budgets]
