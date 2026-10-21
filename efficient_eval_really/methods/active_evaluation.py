import numpy as np

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def active_evaluation_budgets(
    data: Data,
    budgets: Budgets,
    exploration_c: float = 1.0,
    seed: int = 0,
    **kwargs,
) -> ModelScoresAtBudget:
    """Active Evaluation: Efficient Evaluation with Successive Elimination (Beat-the-Mean).

    Paper: "Active Evaluation: Efficient NLG Evaluation with Few Pairwise Comparisons"
    (Khosla et al., ACL/arXiv:2203.06063)

    Maintains empirical performance estimates and confidence intervals for each system.
    Sub-optimal systems are progressively eliminated using a Beat-the-Mean style
    criterion, focusing the evaluation budget on distinguishing the top-performing models.
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

    # Fast state tracking with running sums, counts, and boolean mask
    sums = np.zeros(n_models, dtype=float)
    counts = np.zeros(n_models, dtype=int)
    evaluated_mask = np.zeros((n_models, n_items), dtype=bool)
    evaluated: list[list[int]] = [[] for _ in range(n_models)]
    current_cost = 0.0

    # Initial warm-up: 1 item evaluated across all models
    initial_item = int(np.argmax(np.var(S, axis=0) / (costs + 1e-6)))
    for m in range(n_models):
        evaluated[m].append(initial_item)
        evaluated_mask[m, initial_item] = True
        sums[m] += S[m, initial_item]
        counts[m] += 1
        current_cost += float(costs[initial_item])

    results_map = {}
    sorted_budgets = sorted(set(budgets))
    BATCH_SIZE = max(10, min(100, n_models // 2))

    for b in sorted_budgets:
        while current_cost < b:
            # Vectorized mean and confidence bound
            means = sums / np.maximum(1, counts)
            radius = exploration_c * np.sqrt(np.log(n_models + 1) / np.maximum(1, counts))

            ucb = means + radius
            lcb = means - radius

            # Active set: models not eliminated by the current best LCB
            max_lcb = np.max(lcb)
            active_models = np.where(ucb >= max_lcb)[0]
            if len(active_models) == 0:
                active_models = np.arange(n_models)

            batch_limit = min(BATCH_SIZE, int(b - current_cost) + 1)
            added = 0
            for _ in range(batch_limit):
                # Pick active model with smallest count
                m_pick = active_models[np.argmin(counts[active_models])]
                if counts[m_pick] >= n_items:
                    avail_active = [m for m in active_models if counts[m] < n_items]
                    if not avail_active:
                        break
                    m_pick = avail_active[0]

                unvisited = np.where(~evaluated_mask[m_pick])[0]
                if len(unvisited) == 0:
                    break

                if len(active_models) > 1 and len(unvisited) > 1:
                    cand_slice = unvisited[:50]
                    var_cand = np.var(S[active_models][:, cand_slice], axis=0)
                    j_pick = cand_slice[np.argmax(var_cand / (costs[cand_slice] + 1e-6))]
                else:
                    j_pick = unvisited[0]

                evaluated_mask[m_pick, j_pick] = True
                evaluated[m_pick].append(j_pick)
                sums[m_pick] += S[m_pick, j_pick]
                counts[m_pick] += 1
                current_cost += float(costs[j_pick])
                added += 1

            if added == 0:
                break

        # Construct results
        model_scores = {m: [data[j]["scores"][m] for j in evaluated[i]] for i, m in enumerate(models)}
        results_map[b] = model_scores

    return [results_map[b] for b in budgets]
