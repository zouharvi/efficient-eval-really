import numpy as np
import scipy.special

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def adaptive_arena_budgets(
    data: Data,
    budgets: Budgets,
    seed: int = 0,
    lambda_reg: float = 1.0,
    **kwargs,
) -> ModelScoresAtBudget:
    """Adaptive-Arena: Fewer Battles, More Gain via A-optimality on Bradley-Terry.

    Algorithm based on Liu et al. (ICLR 2026):
    'Fewer Battles, More Gain: An Information-Efficient Framework for Arena-based LLM Evaluation'

    1. Maintains online Bradley-Terry latent ability ratings r for each model.
    2. Dynamically selects model pairs to battle that maximize the A-optimality criterion
       (trace reduction of the Bradley-Terry Fisher information matrix inverse C = I_S^(-1)).
    3. Fair budget accounting & 0-cost battle harvesting (Option A):
       Evaluating a model on a prompt for the first time incurs 1 budget unit. Whenever a model
       is evaluated on a prompt, it automatically battles against all other models already
       evaluated on that prompt for 0 additional generation cost.
    4. Updates ratings and information matrix online via Sherman-Morrison rank-1 updates.
    """
    np.random.default_rng(seed)
    models = list(data[0]["scores"].keys())
    n_models = len(models)
    n_items = len(data)

    scores_mat = np.array([[item["scores"][m] for m in models] for item in data], dtype=float)
    costs_arr = np.array([item["cost"] for item in data], dtype=float)
    max_possible_cost = float(np.sum(costs_arr) * n_models)

    r = np.zeros(n_models, dtype=float)
    c_diag = (1.0 / lambda_reg) * np.ones(n_models, dtype=float)

    pairs = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            pairs.append((i, j))
    pairs = np.array(pairs, dtype=int)
    p_i = pairs[:, 0]
    p_j = pairs[:, 1]
    n_pairs = len(pairs)

    wins = np.zeros(n_models, dtype=float)
    matches = np.zeros(n_models, dtype=float)
    evaluated_mask = np.zeros((n_items, n_models), dtype=bool)
    model_scores = [[] for _ in range(n_models)]
    current_cost = 0.0

    sorted_budgets = sorted(set(budgets))
    results_map = {}
    budget_idx = 0
    item_counter = 0

    BATCH_SIZE = max(32, min(128, n_models // 2))

    def evaluate_model_on_item(m: int, item: int):
        nonlocal current_cost
        if evaluated_mask[item, m]:
            return
        current_cost += costs_arr[item]
        model_scores[m].append(scores_mat[item, m])

        s_m = scores_mat[item, m]
        others = np.where(evaluated_mask[item])[0]
        if len(others) > 0:
            s_others = scores_mat[item, others]
            w_m = np.sum(s_m > s_others) + 0.5 * np.sum(s_m == s_others)
            wins[m] += w_m
            matches[m] += len(others)

            w_others = (s_others > s_m).astype(float) + 0.5 * (s_others == s_m).astype(float)
            wins[others] += w_others
            matches[others] += 1.0

        evaluated_mask[item, m] = True

    while budget_idx < len(sorted_budgets):
        target_budget = sorted_budgets[budget_idx]

        if current_cost >= target_budget or current_cost >= max_possible_cost:
            rates = np.where(matches > 0, wins / matches, 0.5)
            results_map[target_budget] = {m: list(model_scores[i]) if len(model_scores[i]) > 0 else [float(rates[i])] for i, m in enumerate(models)}
            budget_idx += 1
            continue

        if n_pairs == 0:
            break

        logits = r[p_i] - r[p_j]
        prob = scipy.special.expit(logits)
        w = prob * (1.0 - prob)

        c_diff = c_diag[p_i] + c_diag[p_j]
        c2_diff = (c_diag[p_i] ** 2) + (c_diag[p_j] ** 2)

        denom = 1.0 + w * c_diff
        trace_red = (w * c2_diff) / np.maximum(1e-12, denom)

        batch_sz = min(BATCH_SIZE, max(1, int(target_budget - current_cost) // 2))
        top_pairs = np.argsort(trace_red)[-batch_sz:]

        cost_before = current_cost
        for p_idx in top_pairs:
            if current_cost >= target_budget or current_cost >= max_possible_cost:
                break

            m1, m2 = pairs[p_idx]
            uneval = np.where(~(evaluated_mask[:, m1] & evaluated_mask[:, m2]))[0]
            if len(uneval) == 0:
                item_idx = item_counter % n_items
                item_counter += 1
            else:
                item_idx = uneval[0]

            evaluate_model_on_item(m1, item_idx)
            if current_cost < target_budget:
                evaluate_model_on_item(m2, item_idx)

            s1 = scores_mat[item_idx, m1]
            s2 = scores_mat[item_idx, m2]

            y_12 = 1.0 if s1 > s2 else (0.0 if s2 > s1 else 0.5)

            lr = 0.5 / np.sqrt(1.0 + matches[m1] + matches[m2])
            grad = y_12 - prob[p_idx]
            r[m1] += lr * grad
            r[m2] -= lr * grad

            w_val = w[p_idx]
            denom_val = denom[p_idx]
            c_factor = w_val / max(1e-12, denom_val)
            c_diag[m1] = max(1e-4, c_diag[m1] - c_factor * (c_diag[m1] ** 2))
            c_diag[m2] = max(1e-4, c_diag[m2] - c_factor * (c_diag[m2] ** 2))

        if current_cost == cost_before:
            while budget_idx < len(sorted_budgets):
                tb = sorted_budgets[budget_idx]
                rates = np.where(matches > 0, wins / matches, 0.5)
                results_map[tb] = {m: list(model_scores[i]) if len(model_scores[i]) > 0 else [float(rates[i])] for i, m in enumerate(models)}
                budget_idx += 1
            break

        r -= np.mean(r)

    rates = np.where(matches > 0, wins / matches, 0.5)
    for b in sorted_budgets:
        if b not in results_map:
            results_map[b] = {m: list(model_scores[i]) if len(model_scores[i]) > 0 else [float(rates[i])] for i, m in enumerate(models)}

    return [results_map[b] for b in budgets]
