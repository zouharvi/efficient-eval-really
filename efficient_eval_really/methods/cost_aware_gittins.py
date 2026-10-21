import numpy as np
import scipy.optimize
import scipy.stats

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def _precompute_gittins_roots(H: int, c: float, v0: float = 0.04, tau2: float = 0.25) -> np.ndarray:
    """Precompute the Gittins index root schedule r_n for n=0..H-1.

    Under Gaussian observation noise tau^2 and cost c, solves the centered dynamic program
    recursively such that q_n(r_n) = 0.
    """
    v = np.zeros(H + 1)
    v[0] = v0
    for n in range(H):
        v[n + 1] = (v[n] * tau2) / (v[n] + tau2)

    sigma = np.zeros(H)
    for n in range(H):
        sigma[n] = np.sqrt(max(1e-12, (v[n] ** 2) / (v[n] + tau2)))

    roots = np.zeros(H)
    r_next = 0.0

    for n in range(H - 1, -1, -1):
        sig = sigma[n]
        if sig < 1e-8 or c <= 0:
            roots[n] = r_next
            continue

        def obj(delta: float) -> float:
            z = delta / sig
            return float(delta * scipy.stats.norm.cdf(z) + sig * scipy.stats.norm.pdf(z) - c)

        try:
            delta = scipy.optimize.brentq(obj, -10.0 * sig, max(10.0 * sig, 5.0 * c + 2.0))
        except ValueError:
            delta = 0.0
        r_n = r_next + delta
        roots[n] = r_n
        r_next = r_n

    return roots


def cost_aware_gittins_budgets(
    data: Data,
    budgets: Budgets,
    c: float = 0.005,
    mu0: float = 0.5,
    v0: float = 0.04,
    tau2: float = 0.25,
    seed: int = 0,
    **kwargs,
) -> ModelScoresAtBudget:
    """Cost-Aware Gittins Index Policy for Bayesian Bandit LLM Evaluation.

    Algorithm from Xie et al.:
    'Efficient Cost-Aware LLM Evaluation via Bayesian Bandit Gittins Indices'

    Maintains a Gaussian posterior belief over each candidate LLM's benchmark accuracy.
    At each step, calculates the Gittins index G_k = mu_k - r_{n_k} balancing current performance
    and the remaining value of information under evaluation costs.
    """
    models = list(data[0]["scores"].keys())
    n_models = len(models)
    n_items = len(data)

    scores_mat = np.array([[item["scores"][m] for m in models] for item in data], dtype=float)
    costs_arr = np.array([item["cost"] for item in data], dtype=float)

    roots = _precompute_gittins_roots(H=n_items, c=c, v0=v0, tau2=tau2)

    mu = np.full(n_models, mu0, dtype=float)
    v = np.full(n_models, v0, dtype=float)
    n_pulls = np.zeros(n_models, dtype=int)
    item_indices = np.zeros(n_models, dtype=int)

    model_scores = [[] for _ in range(n_models)]
    current_cost = 0.0

    # Initial warm-up: evaluate 1 item per model
    warmup = min(1, n_items)
    for m in range(n_models):
        for i in range(warmup):
            sc = scores_mat[i, m]
            model_scores[m].append(sc)
            v_new = (v[m] * tau2) / (v[m] + tau2)
            mu[m] = mu[m] + (v[m] / (v[m] + tau2)) * (sc - mu[m])
            v[m] = v_new
            n_pulls[m] += 1
            current_cost += costs_arr[i]
        item_indices[m] = warmup

    results_map = {}
    sorted_budgets = sorted(set(budgets))

    for budget in sorted_budgets:
        while current_cost < budget:
            avail = np.where(item_indices < n_items)[0]
            if len(avail) == 0:
                break

            # Gittins index: G_k = mu_k - r_{n_k}
            gittins_indices = np.array([mu[k] - roots[min(n_pulls[k], n_items - 1)] for k in avail])

            chosen_idx = avail[int(np.argmax(gittins_indices))]
            item_idx = item_indices[chosen_idx]
            sc = scores_mat[item_idx, chosen_idx]
            model_scores[chosen_idx].append(sc)

            v_new = (v[chosen_idx] * tau2) / (v[chosen_idx] + tau2)
            mu[chosen_idx] = mu[chosen_idx] + (v[chosen_idx] / (v[chosen_idx] + tau2)) * (sc - mu[chosen_idx])
            v[chosen_idx] = v_new
            n_pulls[chosen_idx] += 1
            current_cost += costs_arr[item_idx]
            item_indices[chosen_idx] += 1

        results_map[budget] = {m: list(model_scores[i]) if len(model_scores[i]) > 0 else [mu0] for i, m in enumerate(models)}

    return [results_map[b] for b in budgets]
