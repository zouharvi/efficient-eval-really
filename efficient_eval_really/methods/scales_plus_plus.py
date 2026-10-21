import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def scales_plus_plus_budgets(
    data: Data,
    budgets: Budgets,
    n_pca_components: int = 3,
    seed: int = 0,
    **kwargs,
) -> ModelScoresAtBudget:
    """Scales++: Compute Efficient Evaluation Subset Selection.

    Algorithm from Bean et al. (2025):
    'Scales++: Compute Efficient Evaluation Subset Selection with Cognitive Scales Embeddings'

    Uses item embeddings representing cognitive demands, applies PCA dimensionality
    reduction to 3 dimensions, and performs K-Means clustering to select the representative
    item closest to each cluster center.
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    # Extract item embeddings or fallback to proxy metric vector
    if "embedding" in data[0]:
        embeddings = np.array([item["embedding"] for item in data], dtype=float)
    else:
        embeddings = np.array(
            [[item["scores_metrics"][m][metric] for m in models] for item in data],
            dtype=float,
        )

    # Reduce dimensionality with PCA as in Scales++ (to 3 dimensions)
    d = min(n_pca_components, embeddings.shape[1], n_items)
    if d >= 1:
        pca = PCA(n_components=d, random_state=seed)
        X = pca.fit_transform(embeddings)
    else:
        X = embeddings

    results: ModelScoresAtBudget = []
    for budget in budgets:
        k = max(1, min(budget // n_models, n_items))

        if k >= n_items:
            results.append({m: [float(item["scores"][m]) for item in data] for m in models})
            continue

        import warnings

        from sklearn.exceptions import ConvergenceWarning

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto")
            kmeans.fit(X)

        centers = kmeans.cluster_centers_  # (k, d)
        # Pairwise distance from all items to each center: (n_items, k)
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        coreset = np.argmin(dists, axis=0).tolist()

        results.append({m: [float(data[idx]["scores"][m]) for idx in coreset] for m in models})

    return results
