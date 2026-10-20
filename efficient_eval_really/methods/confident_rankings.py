import numpy as np
from scipy import stats

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget
from efficient_eval_really.methods.irt_chatbots import _fit_2pl_irt


def confident_rankings_budgets(
    data: Data,
    budgets: Budgets,
    confidence: float = 0.95,
    n_init: int = 5,
    seed: int = 42,
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
    n_items = len(data)

    S = np.array([[item["scores_metrics"][m][metric] for m in models] for item in data], dtype=float)
    mean_val = float(np.mean(S))
    Y = (S >= mean_val).astype(float).T

    a, b, _ = _fit_2pl_irt(Y)
    a = np.clip(np.nan_to_num(a, nan=1.0), 0.1, 5.0)
    b = np.clip(np.nan_to_num(b, nan=0.0), -5.0, 5.0)

    results: ModelScoresAtBudget = []
    for budget in budgets:
        model_scores = {m: [] for m in models}
        used_items = {m: set() for m in models}
        thetas = {m: 0.0 for m in models}
        se = {m: 1.0 for m in models}
        cost = 0.0

        # Phase 1: Warm-up
        warmup_k = min(max(2, n_init), n_items)
        p_0 = 1.0 / (1.0 + np.exp(a * b))
        fisher_0 = (a ** 2) * p_0 * (1.0 - p_0)
        top_warmup = np.argsort(fisher_0)[::-1][:warmup_k]

        for m in models:
            for idx in top_warmup:
                score = float(data[idx]["scores"][m])
                model_scores[m].append(score)
                used_items[m].add(int(idx))
                cost += data[idx]["cost"]
            thetas[m] = float(np.mean(model_scores[m]))
            se[m] = (
                float(np.std(model_scores[m], ddof=1) / np.sqrt(len(model_scores[m])))
                if len(model_scores[m]) > 1
                else 0.5
            )

        # Phase 2: Adaptive selection
        while cost < budget:
            sorted_models = sorted(models, key=lambda m: thetas[m], reverse=True)
            uncertain_pairs = []
            for i in range(len(sorted_models) - 1):
                m_a, m_b = sorted_models[i], sorted_models[i + 1]
                if len(used_items[m_a]) >= n_items and len(used_items[m_b]) >= n_items:
                    continue
                diff = thetas[m_a] - thetas[m_b]
                denom = np.sqrt(se[m_a] ** 2 + se[m_b] ** 2)
                p_a_gt_b = stats.norm.cdf(diff / max(1e-8, denom))
                if 1.0 - confidence < p_a_gt_b < confidence:
                    uncertain_pairs.append((m_a, m_b))

            if not uncertain_pairs:
                break

            candidates = set()
            for m_a, m_b in uncertain_pairs:
                if len(used_items[m_a]) < n_items:
                    candidates.add(m_a)
                if len(used_items[m_b]) < n_items:
                    candidates.add(m_b)

            if not candidates:
                break

            target_model = max(
                candidates,
                key=lambda m: (se[m] ** 2) / (len(model_scores[m]) + 1.0),
            )

            available = [idx for idx in range(n_items) if idx not in used_items[target_model]]
            if not available:
                break
            av_arr = np.array(available, dtype=int)
            logits = a[av_arr] * (thetas[target_model] - b[av_arr])
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
            fisher = (a[av_arr] ** 2) * p * (1.0 - p)
            best_item = int(av_arr[np.argmax(fisher)])

            score = float(data[best_item]["scores"][target_model])
            model_scores[target_model].append(score)
            used_items[target_model].add(best_item)
            cost += data[best_item]["cost"]

            thetas[target_model] = float(np.mean(model_scores[target_model]))
            se[target_model] = (
                float(np.std(model_scores[target_model], ddof=1) / np.sqrt(len(model_scores[target_model])))
                if len(model_scores[target_model]) > 1
                else 0.5
            )

        results.append(model_scores)

    return results
