import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def smart_filtering_budgets(
    data: Data,
    budgets: Budgets,
    easy_percentile: float = 90.0,
    sim_threshold: float = 0.95,
) -> ModelScoresAtBudget:
    """SMART Filtering of Benchmark Datasets.

    Algorithm from Gupta et al. (Meta AI, NAACL 2025):
    'Improving Model Evaluation using SMART Filtering of Benchmark Datasets'

    Applies two-stage filtering:
    1. Filters out easy / ceiling examples using model agreement across proxy metrics.
    2. Uses embedding cosine similarity to deduplicate semantically similar examples,
       prioritizing items with the highest discriminative variance across models.
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    # 1. Filter out ceiling / easy examples
    avg_scores = np.array([float(np.mean([item["scores_metrics"][m][metric] for m in models])) for item in data])
    cutoff = float(np.percentile(avg_scores, easy_percentile))
    candidate_indices = [i for i in range(n_items) if avg_scores[i] < cutoff]
    if len(candidate_indices) < 2:
        candidate_indices = list(range(n_items))

    # 2. Embedding-based deduplication
    if "embedding" in data[0]:
        cand_embeddings = np.array([data[i]["embedding"] for i in candidate_indices], dtype=float)
    else:
        cand_embeddings = np.array(
            [[data[i]["scores_metrics"][m][metric] for m in models] for i in candidate_indices],
            dtype=float,
        )

    sim_matrix = cosine_similarity(cand_embeddings)
    vars_cand = np.array([float(np.std([data[i]["scores_metrics"][m][metric] for m in models])) for i in candidate_indices])
    sorted_cand = np.argsort(vars_cand)[::-1]

    kept_indices: list[int] = []
    for idx in sorted_cand:
        real_idx = candidate_indices[idx]
        if not kept_indices:
            kept_indices.append(real_idx)
            continue
        kept_sub_idx = [candidate_indices.index(ki) for ki in kept_indices]
        max_sim = float(np.max(sim_matrix[idx, kept_sub_idx]))
        if max_sim < sim_threshold:
            kept_indices.append(real_idx)

    # If deduplication reduced the pool below n_items, append remaining candidates
    if len(kept_indices) < n_items:
        for idx in sorted_cand:
            real_idx = candidate_indices[idx]
            if real_idx not in kept_indices:
                kept_indices.append(real_idx)

    results: ModelScoresAtBudget = []
    for budget in budgets:
        k = max(1, min(budget // n_models, n_items))

        if k >= n_items:
            results.append({m: [float(item["scores"][m]) for item in data] for m in models})
            continue

        coreset = kept_indices[:k]
        results.append({m: [float(data[idx]["scores"][m]) for idx in coreset] for m in models})

    return results
