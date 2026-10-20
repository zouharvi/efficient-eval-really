import numpy as np
from sklearn.decomposition import TruncatedSVD

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def valid_best_model_budgets(
    data: Data,
    budgets: Budgets,
    rank: int = 2,
    tau: int = 25,
    eta: float = 1.0,
    seed: int = 42,
) -> ModelScoresAtBudget:
    """Prediction-Powered Unbiased Low-Rank Sequential Evaluation (PULSE).

    Algorithm from Tolochinsky, Tenzer & Romano (2026):
    'Valid Best-Model Identification for LLM Evaluation via Low-Rank Factorization'

    Combines Multi-Armed Bandit arm selection with low-rank matrix factorization (SVD)
    to impute unobserved model-item evaluations. Explores arm-instance pairs maximizing
    an optimistic upper bound of observed performance plus imputed control variates.
    """
    rng = np.random.default_rng(seed)
    models = list(data[0]["scores"].keys())
    n_models = len(models)
    n_items = len(data)

    results: ModelScoresAtBudget = []
    for budget in budgets:
        observed = np.zeros((n_models, n_items), dtype=bool)
        scores_matrix = np.zeros((n_models, n_items), dtype=float)
        cost = 0.0

        # Stage 1: Warm-up phase
        n_warmup = min(max(2, int(0.05 * n_items)), n_items)
        for m_idx in range(n_models):
            items_w = rng.choice(n_items, size=n_warmup, replace=False)
            for j in items_w:
                scores_matrix[m_idx, j] = float(data[j]["scores"][models[m_idx]])
                observed[m_idx, j] = True
                cost += data[j]["cost"]

        actual_rank = min(rank, n_models, n_warmup)
        step = 0
        S_hat = scores_matrix.copy()
        R = np.ones((n_models, n_items), dtype=float)

        # Stage 2: Adaptive low-rank bandit selection
        while cost < budget:
            # Periodically re-fit low-rank matrix completion
            if step % tau == 0:
                M = scores_matrix.copy()
                for i in range(n_models):
                    row_obs = observed[i]
                    if np.any(row_obs):
                        M[i, ~row_obs] = np.mean(M[i, row_obs])
                    else:
                        M[i, ~row_obs] = 0.5

                if actual_rank >= 1 and n_models > 1:
                    svd = TruncatedSVD(n_components=actual_rank, random_state=seed)
                    S_hat = svd.inverse_transform(svd.fit_transform(M))
                else:
                    S_hat = M

                R = np.abs(M - S_hat) + (1.0 - observed.astype(float))

            # Upper bound for each model
            B = []
            for i in range(n_models):
                row_val = np.where(observed[i], scores_matrix[i], S_hat[i] + eta * R[i])
                B.append(float(np.mean(row_val)))

            unobs_models = [i for i in range(n_models) if not np.all(observed[i])]
            if not unobs_models:
                break
            best_m_idx = max(unobs_models, key=lambda i: B[i])

            unobs_items = np.where(~observed[best_m_idx])[0]
            if len(unobs_items) == 0:
                break
            best_item = int(unobs_items[np.argmax(R[best_m_idx, unobs_items])])

            scores_matrix[best_m_idx, best_item] = float(data[best_item]["scores"][models[best_m_idx]])
            observed[best_m_idx, best_item] = True
            cost += data[best_item]["cost"]
            step += 1

        model_scores = {}
        for m_idx, m in enumerate(models):
            obs_indices = np.where(observed[m_idx])[0]
            if len(obs_indices) > 0:
                model_scores[m] = [float(scores_matrix[m_idx, idx]) for idx in obs_indices]
            else:
                model_scores[m] = [float(data[0]["scores"][m])]

        results.append(model_scores)

    return results
