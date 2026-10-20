import numpy as np
import scipy.cluster.hierarchy
from sklearn.metrics.pairwise import cosine_distances
from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def diffuse_budgets(data: Data, budgets: Budgets) -> ModelScoresAtBudget:
    """DiffUse: Label-Efficient Model Selection for Text Generation.

    Algorithm from Ashury-Tahan et al. (IBM Research, ACL 2024):
    'Label-Efficient Model Selection for Text Generation (DiffUse)'

    Clusters semantic difference representations of item behaviors across models
    using hierarchical clustering (Ward's linkage), selecting the representative example
    closest to each cluster centroid.
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    if "embedding" in data[0]:
        base_emb = np.array([item["embedding"] for item in data], dtype=float)
    else:
        base_emb = np.array(
            [[item["scores_metrics"][m][metric] for m in models] for item in data],
            dtype=float,
        )

    # Discrepancy representation: item embedding modulated by cross-model deviation
    S_metric = np.array([[item["scores_metrics"][m][metric] for m in models] for item in data], dtype=float)
    discrepancy = np.std(S_metric, axis=1, keepdims=True)
    diff_vectors = base_emb * (1.0 + discrepancy)

    results: ModelScoresAtBudget = []
    for budget in budgets:
        k = max(1, min(budget // n_models, n_items))

        if k >= n_items:
            results.append({m: [float(item["scores"][m]) for item in data] for m in models})
            continue

        if k == 1:
            best_idx = int(np.argmax(discrepancy.ravel()))
            results.append({m: [float(data[best_idx]["scores"][m])] for m in models})
            continue

        # Ward hierarchical clustering on difference vectors
        Z = scipy.cluster.hierarchy.linkage(diff_vectors, method="ward")
        cluster_labels = scipy.cluster.hierarchy.fcluster(Z, t=k, criterion="maxclust")

        coreset: list[int] = []
        for c in range(1, k + 1):
            members = np.where(cluster_labels == c)[0]
            if len(members) == 0:
                continue
            center = np.mean(diff_vectors[members], axis=0, keepdims=True)
            dists = cosine_distances(diff_vectors[members], center).ravel()
            best_member = int(members[np.argmin(dists)])
            coreset.append(best_member)

        results.append({
            m: [float(data[idx]["scores"][m]) for idx in coreset]
            for m in models
        })

    return results
