import numpy as np
import scipy.special

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def adaptive_arena_budgets(
    data: Data,
    budgets: Budgets,
    seed: int = 42,
    lambda_reg: float = 1.0,
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
    rng = np.random.default_rng(seed)
    models = list(data[0]["scores"].keys())
    n_models = len(models)
    n_items = len(data)

    scores_mat = np.array([[item["scores"][m] for m in models] for item in data], dtype=float)
    costs_arr = np.array([item["cost"] for item in data], dtype=float)

    r = np.zeros(n_models, dtype=float)
    C = (1.0 / lambda_reg) * np.eye(n_models, dtype=float)

    pairs = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            pairs.append((i, j))
    pairs = np.array(pairs, dtype=int)
    n_pairs = len(pairs)

    wins = np.zeros(n_models, dtype=float)
    matches = np.zeros(n_models, dtype=float)
    evaluated_per_item = [set() for _ in range(n_items)]
    model_scores = [[] for _ in range(n_models)]
    current_cost = 0.0

    sorted_budgets = sorted(set(budgets))
    results_map = {}
    budget_idx = 0
    item_counter = 0

    BATCH_SIZE = min(16, max(1, n_models // 10))

    def evaluate_model_on_item(m: int, item: int):
        nonlocal current_cost
        if m in evaluated_per_item[item]:
            return
        current_cost += costs_arr[item]
        model_scores[m].append(scores_mat[item, m])

        s_m = scores_mat[item, m]
        for other in evaluated_per_item[item]:
            s_other = scores_mat[item, other]
            matches[m] += 1
            matches[other] += 1
            if s_m > s_other:
                wins[m] += 1.0
            elif s_other > s_m:
                wins[other] += 1.0
            else:
                wins[m] += 0.5
                wins[other] += 0.5

        evaluated_per_item[item].add(m)

    while budget_idx < len(sorted_budgets):
        target_budget = sorted_budgets[budget_idx]

        if current_cost >= target_budget:
            rates = np.where(matches > 0, wins / matches, 0.5)
            results_map[target_budget] = {
                m: list(model_scores[i]) if len(model_scores[i]) > 0 else [float(rates[i])]
                for i, m in enumerate(models)
            }
            budget_idx += 1
            continue

        if n_pairs == 0:
            break

        C2 = C @ C
        p_i = pairs[:, 0]
        p_j = pairs[:, 1]

        logits = r[p_i] - r[p_j]
        prob = scipy.special.expit(logits)
        w = prob * (1.0 - prob)

        c_diff = C[p_i, p_i] + C[p_j, p_j] - 2.0 * C[p_i, p_j]
        c2_diff = C2[p_i, p_i] + C2[p_j, p_j] - 2.0 * C2[p_i, p_j]

        denom = 1.0 + w * c_diff
        trace_red = (w * c2_diff) / np.maximum(1e-12, denom)

        top_pairs = np.argsort(trace_red)[-BATCH_SIZE:]

        for p_idx in top_pairs:
            if current_cost >= target_budget:
                break

            m1, m2 = pairs[p_idx]
            item_idx = item_counter % n_items
            item_counter += 1

            # Option A: Fair budget accounting + free co-evaluation battles
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

            v_diff = np.zeros(n_models)
            v_diff[m1] = 1.0
            v_diff[m2] = -1.0
            c_vec = C @ v_diff
            w_val = w[p_idx]
            denom_val = denom[p_idx]
            C -= (w_val / max(1e-12, denom_val)) * np.outer(c_vec, c_vec)

        r -= np.mean(r)

    rates = np.where(matches > 0, wins / matches, 0.5)
    for b in sorted_budgets:
        if b not in results_map:
            results_map[b] = {
                m: list(model_scores[i]) if len(model_scores[i]) > 0 else [float(rates[i])]
                for i, m in enumerate(models)
            }

    return [results_map[b] for b in budgets]
