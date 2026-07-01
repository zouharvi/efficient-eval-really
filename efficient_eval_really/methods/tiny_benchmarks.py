import numpy as np
from efficient_eval_really.methods import Data, Budgets, ModelScoresAtBudget


def _binarize(S: np.ndarray) -> np.ndarray:
    """Binarize at threshold that preserves the overall mean."""
    return (S >= np.mean(S)).astype(float)


def _kmeans_anchors(X: np.ndarray, k: int) -> list[int]:
    """K-Means clustering; return index of item nearest each centroid."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    return pairwise_distances(kmeans.cluster_centers_, X).argmin(axis=1).tolist()


def _fit_irt(Y: np.ndarray, D: int = 10) -> np.ndarray:
    """Fit multidim 2PL IRT on Y (n_models, n_items) via MLE. Returns (n_items, 2*D) embeddings."""
    from scipy.optimize import minimize
    n_models, n_items = Y.shape
    eps = 1e-8

    mean_p = np.clip(Y.mean(axis=0), eps, 1 - eps)
    # warm-start: first dim = scalar 2PL, remaining dims = 0
    A0 = np.zeros((n_items, D)); A0[:, 0] = 1.0
    B0 = np.zeros((n_items, D)); B0[:, 0] = np.log(mean_p / (1 - mean_p))
    x0 = np.concatenate([A0.ravel(), B0.ravel(), np.zeros(n_models * D)])

    def nll(x):
        A = x[:n_items * D].reshape(n_items, D)
        B = x[n_items * D:2 * n_items * D].reshape(n_items, D)
        theta = x[2 * n_items * D:].reshape(n_models, D)
        # logits[m, i] = sum_d A[i,d]*theta[m,d] - B[i,d]
        logits = (A[None, :, :] * theta[:, None, :] - B[None, :, :]).sum(axis=2)
        p = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        return -(Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps)).sum()

    res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 500})
    A = res.x[:n_items * D].reshape(n_items, D)
    B = res.x[n_items * D:2 * n_items * D].reshape(n_items, D)
    return np.hstack([A, B])  # (n_items, 2*D)


def tiny_benchmarks_budgets(data: Data, budgets: Budgets, method: str) -> ModelScoresAtBudget:
    """tinyBenchmarks item selection.

    method='clustering': K-Means on binarized response patterns.
    method='irt':        K-Means on 2PL IRT (disc, diff) embeddings.
    """
    models = list(data[0]["scores"].keys())
    S = np.array([[item["scores"][m] for m in models] for item in data])  # (n_items, n_models)
    n_items, n_models = S.shape

    Y = _binarize(S)
    if method == "clustering":
        X = Y                      # (n_items, n_models)
    elif method == "irt":
        X = _fit_irt(Y.T)          # (n_items, 2)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    results: ModelScoresAtBudget = []
    for budget in budgets:
        k = max(1, min(budget // n_models, n_items))
        if k >= n_items:
            results.append({m: [item["scores"][m] for item in data] for m in models})
            continue
        anchors = _kmeans_anchors(X, k)
        results.append({m: [data[i]["scores"][m] for i in anchors] for m in models})
    return results
