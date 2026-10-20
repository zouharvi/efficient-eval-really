import numpy as np
from sklearn.cluster import KMeans
from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def select_label_evaluate_budgets(
    data: Data,
    budgets: Budgets,
    seed: int = 42,
) -> ModelScoresAtBudget:
    """Select, Label, Evaluate: Active Testing in NLP.

    Algorithm from Purificato et al. (Amazon / Sapienza, 2026):
    'Select, Label, Evaluate: Active Testing in NLP'

    Partitions benchmark items into semantic clusters using input embeddings to ensure
    domain coverage, and selects the most informative item within each cluster based on
    cross-model predictive uncertainty and disagreement.
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    if "embedding" in data[0]:
        embeddings = np.array([item["embedding"] for item in data], dtype=float)
    else:
        embeddings = np.array(
            [[item["scores_metrics"][m][metric] for m in models] for item in data],
            dtype=float,
        )

    S_metric = np.array([[item["scores_metrics"][m][metric] for m in models] for item in data], dtype=float)
    uncertainty = np.std(S_metric, axis=1)

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
            labels = kmeans.fit_predict(embeddings)

        coreset: list[int] = []
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            # Select the most uncertain/informative member in cluster c
            best_member = int(members[np.argmax(uncertainty[members])])
            coreset.append(best_member)

        results.append({
            m: [float(data[idx]["scores"][m]) for idx in coreset]
            for m in models
        })

    return results
