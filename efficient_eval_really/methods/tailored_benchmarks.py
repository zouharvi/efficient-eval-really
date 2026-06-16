import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from efficient_eval_really.methods import Data, Budgets, ModelScoresSubset, ModelScoresAtBudget


def _kmedoids(D: np.ndarray, k: int, seed: int = 42) -> list[int]:
    """PAM K-Medoids on a precomputed distance matrix."""
    n = D.shape[0]
    if k >= n:
        return list(range(n))
    medoids = list(np.random.default_rng(seed).choice(n, k, replace=False))
    for _ in range(100):
        assignments = np.argmin(D[:, medoids], axis=1)
        new_medoids = []
        for c in range(k):
            members = np.where(assignments == c)[0]
            new_medoids.append(medoids[c] if len(members) == 0
                               else int(members[np.argmin(D[np.ix_(members, members)].sum(1))]))
        if new_medoids == medoids:
            break
        medoids = new_medoids
    return medoids


def tailored_benchmarks_budgets(data: Data, budgets: Budgets, g_frac: float = 0.5) -> ModelScoresAtBudget:
    """TailoredBench (Yuan et al., ACL 2025): per-model tailored item subsets via K-Medoids."""
    models = list(data[0]["scores"].keys())
    S = np.array([[item["scores"][m] for m in models] for item in data])  # (n_items, n_models)
    n_items, n_models = S.shape

    def D(X: np.ndarray) -> np.ndarray:
        return pairwise_distances(StandardScaler().fit_transform(X), metric="manhattan")

    results: ModelScoresAtBudget = []
    for budget in budgets:
        k = max(1, min(budget // n_models, n_items))
        k_g = max(1, round(k * g_frac))

        # Phase 1: global G-set — K-Medoids on standardized item vectors (items × models)
        g_set = _kmedoids(D(S), k_g)

        if k == k_g:
            results.append({m: [data[i]["scores"][m] for i in g_set] for m in models})
            continue

        # Phase 2: per-model tailored N-set
        # Find similar source models using pairwise distances on G-set scores
        D_models = D(S[g_set].T)  # (n_models, n_models)
        threshold = D_models[np.triu_indices(n_models, k=1)].mean()
        n_src = max(1, min(int(np.mean((D_models <= threshold).sum(axis=1) - 1)), n_models - 1))

        model_scores_subset: ModelScoresSubset = {}
        for i, model in enumerate(models):
            src = list(np.argsort(np.where(np.arange(n_models) == i, np.inf, D_models[i]))[:n_src])
            n_set = _kmedoids(D(S[:, src]), k, seed=i)
            model_scores_subset[model] = [data[j]["scores"][model] for j in n_set]

        results.append(model_scores_subset)

    return results
