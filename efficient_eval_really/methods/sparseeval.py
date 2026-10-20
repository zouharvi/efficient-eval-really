import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def sparseeval_budgets(
    data: Data,
    budgets: Budgets,
    refinement_steps: int = 20,
    lambda_reg: float = 1e-4,
) -> ModelScoresAtBudget:
    """SparseEval: Efficient Evaluation of Large Language Models by Sparse Optimization.

    Algorithm from Zhang et al. (ICLR 2026):
    'SparseEval: Efficient Evaluation of Large Language Models by Sparse Optimization'

    Formulates item selection as sparse optimization. For a given budget:
    1. Initializes an anchor set via K-Means clustering on item proxy response patterns.
    2. Iteratively refines the anchors: fits a proxy linear/ridge model to predict overall
       performance from current anchors, computes prediction residuals, replaces the weakest
       anchor (lowest Anchor Importance Score) with the most informative candidate item
       (highest Candidate Importance Score).
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    # S in R^(n_items, n_models)
    S = np.array([[item["scores_metrics"][m][metric] for m in models] for item in data], dtype=float)
    y = np.mean(S, axis=0)  # target mean model score across items

    results: ModelScoresAtBudget = []
    for budget in budgets:
        k = max(1, min(budget // n_models, n_items))
        if k >= n_items:
            results.append({m: [item["scores"][m] for item in data] for m in models})
            continue

        # Step 1: Anchor Initialization via K-Means
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            kmeans.fit(S)
            anchors = pairwise_distances(kmeans.cluster_centers_, S).argmin(axis=1).tolist()
            anchors = list(dict.fromkeys(anchors))
            if len(anchors) < k:
                remaining = [i for i in range(n_items) if i not in anchors]
                anchors.extend(remaining[: k - len(anchors)])

        anchors_set = set(anchors)
        candidates_set = set(range(n_items)) - anchors_set

        # Step 2: Iterative Refinement
        R = min(refinement_steps, len(candidates_set))
        for _ in range(R):
            A = list(anchors_set)
            X_A = S[A, :].T  # (n_models, k)

            reg_mat = lambda_reg * np.eye(len(A))
            try:
                w_A = np.linalg.solve(X_A.T @ X_A + reg_mat, X_A.T @ y)
            except np.linalg.LinAlgError:
                w_A = np.linalg.lstsq(X_A, y, rcond=None)[0]

            # Residual vector
            e = X_A @ w_A - y

            # Candidate Importance Score (CIS): |S[j, :] @ e|
            cand_list = list(candidates_set)
            cand_features = S[cand_list, :]
            cis = np.abs(cand_features @ e)
            best_cand_idx = cand_list[int(np.argmax(cis))]

            # Anchor Importance Score (AIS): |w_i * (S[i, :] @ e)|
            anchor_features = S[A, :]
            ais = np.abs(w_A * (anchor_features @ e))
            worst_anchor_idx = A[int(np.argmin(ais))]

            # Swap
            anchors_set.remove(worst_anchor_idx)
            anchors_set.add(best_cand_idx)
            candidates_set.remove(best_cand_idx)
            candidates_set.add(worst_anchor_idx)

        final_anchors = list(anchors_set)
        results.append({m: [data[i]["scores"][m] for i in final_anchors] for m in models})

    return results
