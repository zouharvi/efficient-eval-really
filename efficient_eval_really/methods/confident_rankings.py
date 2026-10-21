import numpy as np
from scipy.special import ndtr

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget
from efficient_eval_really.methods.irt_chatbots import _fit_2pl_irt


def confident_rankings_budgets(
    data: Data,
    budgets: Budgets,
    confidence: float = 0.95,
    n_init: int = 5,
    seed: int = 0,
    **kwargs,
) -> ModelScoresAtBudget:
    """Continuous Computerized Adaptive Testing with Confident Multi-Model Ranking.

    Algorithm from Balkir et al. (2026):
    'Confident Rankings with Fewer Items: Adaptive LLM Evaluation with Continuous Scores'

    Uses continuous-outcome IRT item calibration (difficulty b, discrimination a)
    followed by a two-phase adaptive procedure:
    1. Warm-up: Administer n_init items with Maximum Fisher Information (MFI) to each model.
    2. Adaptive allocation: Greedily identify adjacent model pairs that fail confidence threshold
       gamma, selecting models with the highest marginal variance reduction per item (SE^2 / (n + 1)).
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    S = np.array([[item["scores_metrics"][m][metric] for m in models] for item in data], dtype=float)
    mean_val = float(np.mean(S))
    Y = (S >= mean_val).astype(float).T

    a, b, _ = _fit_2pl_irt(Y)
    a = np.clip(np.nan_to_num(a, nan=1.0), 0.1, 5.0)
    b = np.clip(np.nan_to_num(b, nan=0.0), -5.0, 5.0)

    # Pre-extract scores and costs as 2D numpy arrays for fast indexing: (n_items, n_models)
    scores_mat = np.array([[item["scores"][m] for m in models] for item in data], dtype=float)
    costs_arr = np.array([item["cost"] for item in data], dtype=float)

    # State tracking
    model_scores = [[] for _ in range(n_models)]
    used_mask = np.zeros((n_models, n_items), dtype=bool)
    counts = np.zeros(n_models, dtype=int)
    sums = np.zeros(n_models, dtype=float)
    sum_sqs = np.zeros(n_models, dtype=float)
    thetas = np.zeros(n_models, dtype=float)
    se = np.ones(n_models, dtype=float)
    cost = 0.0

    # Phase 1: Warm-up
    warmup_k = min(max(2, n_init), n_items)
    p_0 = 1.0 / (1.0 + np.exp(a * b))
    fisher_0 = (a**2) * p_0 * (1.0 - p_0)
    top_warmup = np.argsort(fisher_0)[::-1][:warmup_k]

    for m_idx in range(n_models):
        for idx in top_warmup:
            sc = scores_mat[idx, m_idx]
            model_scores[m_idx].append(sc)
            used_mask[m_idx, idx] = True
            counts[m_idx] += 1
            sums[m_idx] += sc
            sum_sqs[m_idx] += sc * sc
            cost += costs_arr[idx]

        thetas[m_idx] = sums[m_idx] / counts[m_idx]
        if counts[m_idx] > 1:
            var = max(1e-8, (sum_sqs[m_idx] - (sums[m_idx] ** 2) / counts[m_idx]) / (counts[m_idx] - 1))
            se[m_idx] = np.sqrt(var / counts[m_idx])
        else:
            se[m_idx] = 0.5

    # Incremental simulation across sorted budgets
    results_map = {}
    sorted_budgets = sorted(set(budgets))

    for budget in sorted_budgets:
        while cost < budget:
            order = np.argsort(-thetas)

            # Vectorized adjacent pair uncertainty check using C-level standard normal CDF
            diffs = thetas[order[:-1]] - thetas[order[1:]]
            denoms = np.sqrt(se[order[:-1]] ** 2 + se[order[1:]] ** 2)
            probs = ndtr(diffs / np.maximum(1e-8, denoms))
            uncertain = (probs > 1.0 - confidence) & (probs < confidence)

            if not np.any(uncertain):
                break

            # Models involved in uncertain pairs that still have available items
            unc_pairs = np.where(uncertain)[0]
            cand_left = order[unc_pairs]
            cand_right = order[unc_pairs + 1]
            candidates = np.unique(np.concatenate([cand_left, cand_right]))
            candidates = [m for m in candidates if counts[m] < n_items]

            if not candidates:
                break

            cand_arr = np.array(candidates, dtype=int)
            marginal_var = (se[cand_arr] ** 2) / (counts[cand_arr] + 1.0)
            best_cands = cand_arr[np.argsort(-marginal_var)]
            batch_limit = min(max(10, min(100, n_models // 2)), int(budget - cost) + 1)
            added = 0

            for target_m in best_cands[:batch_limit]:
                if cost >= budget:
                    break
                av_items = np.where(~used_mask[target_m])[0]
                if len(av_items) == 0:
                    continue

                # Fisher information selection
                a_av = a[av_items]
                b_av = b[av_items]
                logits = a_av * (thetas[target_m] - b_av)
                p = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
                fisher = (a_av**2) * p * (1.0 - p)
                best_item = av_items[np.argmax(fisher)]

                # Administer item
                sc = scores_mat[best_item, target_m]
                model_scores[target_m].append(sc)
                used_mask[target_m, best_item] = True
                counts[target_m] += 1
                sums[target_m] += sc
                sum_sqs[target_m] += sc * sc
                cost += costs_arr[best_item]
                added += 1

                thetas[target_m] = sums[target_m] / counts[target_m]
                if counts[target_m] > 1:
                    var = max(1e-8, (sum_sqs[target_m] - (sums[target_m] ** 2) / counts[target_m]) / (counts[target_m] - 1))
                    se[target_m] = np.sqrt(var / counts[target_m])
                else:
                    se[target_m] = 0.5

            if added == 0:
                break

        # Snapshot for this budget level
        results_map[budget] = {models[i]: list(model_scores[i]) for i in range(n_models)}

    return [results_map[b] for b in budgets]
