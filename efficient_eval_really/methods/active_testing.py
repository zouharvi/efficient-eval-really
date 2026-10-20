import numpy as np

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def active_testing_budgets(data: Data, budgets: Budgets, seed: int = 42) -> ModelScoresAtBudget:
    """Active Testing with Level-Unit Residual Estimator (LURE).

    Algorithm from Kossen et al. (ICML 2021):
    'Active Testing: Sample-Efficient Model Evaluation'

    Uses surrogate metrics to form acquisition proposals q(x) proportional to
    cross-model variance/uncertainty, samples items sequentially without replacement,
    and applies LURE weights to eliminate selection bias and minimize estimation variance.
    """
    rng = np.random.default_rng(seed)
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    # Proxy scores matrix: (n_items, n_models)
    S_metric = np.array([[item["scores_metrics"][m][metric] for m in models] for item in data], dtype=float)

    # Surrogate uncertainty across models
    uncertainty = np.std(S_metric, axis=1)
    uncertainty = np.nan_to_num(uncertainty, nan=0.0)
    max_unc = float(np.max(uncertainty)) if len(uncertainty) > 0 else 1.0
    eps = 1e-4 * (max_unc if max_unc > 0 else 1.0)
    base_weights = uncertainty + eps

    results: ModelScoresAtBudget = []
    for budget in budgets:
        k = max(1, min(budget // n_models, n_items))

        if k >= n_items:
            results.append({m: [float(item["scores"][m]) for item in data] for m in models})
            continue

        remaining_indices = list(range(n_items))
        q = base_weights.copy()

        sampled_indices = []
        v_weights = []

        for m_step in range(1, k + 1):
            q_rem = q[remaining_indices]
            q_sum = q_rem.sum()
            if q_sum > 0:
                p = q_rem / q_sum
            else:
                p = np.ones(len(remaining_indices)) / len(remaining_indices)

            pick_idx_in_rem = int(rng.choice(len(remaining_indices), p=p))
            item_idx = remaining_indices[pick_idx_in_rem]
            prob_chosen = float(p[pick_idx_in_rem])

            # LURE weight formula (Eq. 4 in Kossen et al.):
            # v_m = 1 + ((N - M) / (N - m)) * ( 1 / ((N - m + 1) * p) - 1 )
            denom = (n_items - m_step + 1) * prob_chosen
            if denom > 1e-12:
                v_m = 1.0 + ((n_items - k) / max(1, n_items - m_step)) * (1.0 / denom - 1.0)
            else:
                v_m = 1.0

            sampled_indices.append(item_idx)
            v_weights.append(v_m)
            remaining_indices.pop(pick_idx_in_rem)

        # Scale evaluated scores by LURE weights so np.mean equals R_LURE
        results.append({
            m: [float(v * data[idx]["scores"][m]) for v, idx in zip(v_weights, sampled_indices)]
            for m in models
        })

    return results
