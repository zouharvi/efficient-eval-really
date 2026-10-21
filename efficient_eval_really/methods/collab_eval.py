import numpy as np
from sklearn.decomposition import TruncatedSVD

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def collab_eval_budgets(
    data: Data,
    budgets: Budgets,
    rank: int = 2,
    seed: int = 0,
    **kwargs,
) -> ModelScoresAtBudget:
    """CollabEval: Collaborative Model Evaluation via Matrix Completion + CrossPPI.

    Paper: "CollabEval: Statistically Efficient Collaborative Model Evaluation via Matrix Completion"
    (Zhang et al., 2026)

    Reframes model evaluation as matrix completion across models and items.
    At each budget level:
    1. Sequentially gathers observations across (model, item) pairs.
    2. Imputes unobserved performance using low-rank matrix decomposition (TruncatedSVD).
    3. Uses CrossPPI (Cross-Prediction-Powered Inference) as unbiased control variates:
       imputed predictions are debiased using the mean observed residual.
    """
    rng = np.random.default_rng(seed)
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    scores_mat = np.array(
        [[item["scores_metrics"][m][metric] for m in models] for item in data],
        dtype=float,
    ).T  # (n_models, n_items)
    costs_arr = np.array([item["cost"] for item in data], dtype=float)

    actual_rank = max(1, min(rank, n_models, n_items))
    observed = np.zeros((n_models, n_items), dtype=bool)
    current_cost = 0.0

    # Initial observations: seed at least 1 item per model
    n_init = max(1, min(2, n_items))
    for m in range(n_models):
        init_items = rng.choice(n_items, size=n_init, replace=False)
        for j in init_items:
            if not observed[m, j]:
                observed[m, j] = True
                current_cost += costs_arr[j]

    results_map = {}
    sorted_budgets = sorted(set(budgets))

    for b in sorted_budgets:
        while current_cost < b:
            unobs = np.where(~observed)
            if len(unobs[0]) == 0:
                break
            idx = rng.integers(len(unobs[0]))
            m_pick = unobs[0][idx]
            j_pick = unobs[1][idx]
            observed[m_pick, j_pick] = True
            current_cost += costs_arr[j_pick]

        # Impute via Matrix Completion (TruncatedSVD)
        M = scores_mat.copy()
        for i in range(n_models):
            row_obs = observed[i]
            if np.any(row_obs):
                M[i, ~row_obs] = np.mean(M[i, row_obs])
            else:
                M[i, ~row_obs] = 0.5

        if actual_rank < min(n_models, n_items):
            svd = TruncatedSVD(n_components=actual_rank, random_state=seed)
            S_hat = svd.fit_transform(M) @ svd.components_
        else:
            S_hat = M.copy()
        S_hat = np.clip(S_hat, 0.0, 1.0)

        # CrossPPI: debiased point estimate for each model
        model_scores = {}
        for i, m in enumerate(models):
            row_obs = observed[i]
            if np.any(row_obs):
                residual_mean = float(np.mean(scores_mat[i, row_obs] - S_hat[i, row_obs]))
                imputed = S_hat[i, :].copy() + residual_mean
                imputed[row_obs] = scores_mat[i, row_obs]
                model_scores[m] = list(np.clip(imputed, 0.0, 1.0))
            else:
                model_scores[m] = list(S_hat[i, :])

        results_map[b] = model_scores

    return [results_map[b] for b in budgets]
