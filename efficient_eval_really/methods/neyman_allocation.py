import numpy as np
from efficient_eval_really.methods import Data, Budgets, ModelScoresAtBudget


def neyman_allocation_budgets(
    data: Data,
    budgets: Budgets,
    method: str = "proxy",
    n_strata: int = 5,
    delta: float = 0.75,
) -> ModelScoresAtBudget:
    """Stratified sampling with Approximate Neyman allocation.

    Allocation Methods:
      'proxy' : allocation based on within stratum standard deviation (SC), more budget to uncertain strata
      'oracle': allocation based on true score standard deviation in strata
      'proportional': allocation based on proportional size of strata.

    Returns Horvitz-Thompson (HT) stratified estimate"""

    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))

    # Score matrix S[i, m] -- shape (n_items, n_models)
    S = np.array([[item["scores_metrics"][m][metric] for m in models] for item in data])
    n_items, n_models = S.shape

    # Surrogate signal approximated for our setup 
    # Semantic entropy (SE): Cross-model score variance per item
    # Self-consistency (SC): Cross-model mean score per item
    se_proxy = S.var(axis=1)
    sc_proxy = S.mean(axis=1)
    # normalize sc_proxy so p*(1-p) is well-defined
    sc_min, sc_max = sc_proxy.min(), sc_proxy.max()
    sc_proxy = (sc_proxy - sc_min) / (sc_max - sc_min) if sc_max > sc_min else np.full_like(sc_proxy, 0.5)

    # Adaptive Stratification
    # identifying zero vs. non-zero variance items
    zero_idx = np.where(se_proxy == 0.0)[0].tolist() 
    nonzero_idx = np.where(se_proxy > 0.0)[0] 
    # creating strata
    strata: list[list[int]] = [zero_idx]
    n_rem = n_strata - 1 
    if len(nonzero_idx) > 0 and n_rem > 0:
        edges = np.quantile(se_proxy[nonzero_idx], np.linspace(0, 1, n_rem + 1))
        bins = np.digitize(se_proxy[nonzero_idx], edges[1:-1])  
        for h in range(n_rem):
            strata.append(nonzero_idx[bins == h].tolist())
    strata = [s for s in strata if len(s) > 0]
    H = len(strata)

    N_h = np.array([len(s) for s in strata], dtype=float)

    # Allocation Methods
    if method == "proxy":
        p_h = np.array([sc_proxy[np.array(s, dtype=int)].mean() for s in strata])
        weights = N_h * np.sqrt(np.clip(p_h * (1.0 - p_h) + delta, 0.0, None))
    elif method == "oracle":
        sigma_h = np.array([S[np.array(s, dtype=int)].std() for s in strata])
        weights = N_h * (sigma_h + 1e-8)
    elif method == "proportional":
        weights = N_h.copy()
    else:
        raise ValueError(f"Unknown method: {method!r}")

    rng = np.random.default_rng(42)
    results: ModelScoresAtBudget = []

    for budget in budgets:
        k = max(H, min(budget // n_models, n_items))

        if k >= n_items:
            results.append({m: [item["scores"][m] for item in data] for m in models})
            continue

        # Neyman allocation
        raw = k * weights / weights.sum()
        m_h = np.clip(np.round(raw).astype(int), 1, N_h.astype(int))

        # Fix rounding 
        diff = k - int(m_h.sum())
        for idx in np.argsort(-weights):
            if diff == 0:
                break
            if diff > 0:
                step = min(diff, int(N_h[idx]) - int(m_h[idx]))
            else:
                step = -min(-diff, int(m_h[idx]) - 1)
            m_h[idx] += step
            diff -= step

        # Uniform sampling without replacement within each stratum
        sampled: list[list[int]] = [
            rng.choice(strata[h], size=int(m_h[h]), replace=False).tolist()
            for h in range(H)
        ]

        # Horvitz-Thompson estimator (correcting for strata sampling rate)
        out: dict[str, list[float]] = {}
        for m in models:
            ht = sum(
                N_h[h] * float(np.mean([data[i]["scores"][m] for i in sampled[h]]))
                for h in range(H)
            )
            out[m] = [ht / n_items]
        results.append(out)

    return results
