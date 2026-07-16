import numpy as np
from efficient_eval_really.methods import Data, Budgets, ModelScoresAtBudget
from efficient_eval_really.methods.tiny_benchmarks import _binarize


def _fit_psn_irt(
    Y: np.ndarray,  # (models,items)
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 0.003,
    hidden: int = 64,
    embed: int = 128,
    seed: int = 42,
):
    """Fit PSN-IRT.

    Architecture:
      model_branch: Embedding(n_models, hidden) + bias -> ReLU -> Linear(hidden, embed) -> Linear(embed, 1) = theta
      item_branch:  Embedding(n_items,  hidden) + bias -> ReLU -> Linear(hidden, embed) -> Linear(embed, 4) = b, a, c_raw, d_raw

    Returns arrays a, b, c, d (each shape (n_items,)) and scalar mean_theta.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    n_models, n_items = Y.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class _PSN(nn.Module):
        def __init__(self):
            super().__init__()

            # Faithful equivalent of the original Linear(n_models, hidden) on one-hot input:
            # Linear(n, h)(e_i) = W[:,i] + b  =  Embedding[i] + shared_bias
            self.model_embed = nn.Embedding(n_models, hidden)
            self.model_bias = nn.Parameter(torch.zeros(hidden))
            self.model_net = nn.Sequential(nn.ReLU(), nn.Linear(hidden, embed))
            self.model_ability_out = nn.Linear(embed, 1)

            # item branch (same structure)
            self.item_embed = nn.Embedding(n_items, hidden)
            self.item_bias = nn.Parameter(torch.zeros(hidden))
            self.item_net = nn.Sequential(nn.ReLU(), nn.Linear(hidden, embed))
            self.item_params_out = nn.Linear(embed, 4)  # b, a, c_raw, d_raw

        def forward(self, model_idx, item_idx):
            theta = self.model_ability_out(self.model_net(self.model_embed(model_idx) + self.model_bias))
            raw = self.item_params_out(self.item_net(self.item_embed(item_idx) + self.item_bias))
            b = raw[:, 0:1]
            a = raw[:, 1:2]
            c = torch.sigmoid(raw[:, 2:3])
            d = torch.sigmoid(raw[:, 3:4])
            prob = c + (d - c) * torch.sigmoid(a * (theta - b))
            return prob

    psn = _PSN().to(device)
    optimizer = torch.optim.Adam(psn.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # Build (model_idx, item_idx, label) triplets 
    rows, cols = np.where(~np.isnan(Y))
    labels = Y[rows, cols].astype(np.float32)
    perm = np.random.default_rng(seed).permutation(len(rows))
    rows, cols, labels = rows[perm], cols[perm], labels[perm]
    n_triplets = len(rows)

    for _ in range(epochs):
        psn.train()
        for start in range(0, n_triplets, batch_size):
            end = min(start + batch_size, n_triplets)
            sid = torch.tensor(rows[start:end], dtype=torch.long, device=device)
            qid = torch.tensor(cols[start:end], dtype=torch.long, device=device)
            lbl = torch.tensor(labels[start:end], device=device)

            prob = psn(sid, qid).squeeze()
            loss = criterion(prob, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Extract parameters for all items
    psn.eval()
    a_parts, b_parts, c_parts, d_parts = [], [], [], []
    with torch.no_grad():
        for start in range(0, n_items, batch_size):
            end = min(start + batch_size, n_items)
            qid = torch.arange(start, end, dtype=torch.long, device=device)
            raw = psn.item_params_out(psn.item_net(psn.item_embed(qid) + psn.item_bias))
            b_parts.append(raw[:, 0].cpu().numpy())
            a_parts.append(raw[:, 1].cpu().numpy())
            c_parts.append(torch.sigmoid(raw[:, 2]).cpu().numpy())
            d_parts.append(torch.sigmoid(raw[:, 3]).cpu().numpy())

        # Mean model ability across all models
        all_model_idx = torch.arange(n_models, dtype=torch.long, device=device)
        thetas = psn.model_ability_out(psn.model_net(psn.model_embed(all_model_idx) + psn.model_bias)).squeeze().cpu().numpy()

    return (
        np.concatenate(a_parts),
        np.concatenate(b_parts),
        np.concatenate(c_parts),
        np.concatenate(d_parts),
        float(np.mean(thetas)),
    )


def _fit_4pl_irt(Y: np.ndarray) -> tuple:
    """Fit 4PL IRT via MLE on Y """
    from scipy.optimize import minimize

    n_models, n_items = Y.shape
    eps = 1e-8

    mean_p = np.clip(Y.mean(axis=0), eps, 1 - eps)
    x0 = np.concatenate([
        np.ones(n_items),                            # a
        -np.log(mean_p / (1 - mean_p)),              # b
        np.full(n_items, -3.0),                      # raw_c  
        np.full(n_items, 3.0),                       # raw_d  
        np.zeros(n_models),                          # theta
    ])

    def nll(x):
        a     = x[:n_items]
        b     = x[n_items:2 * n_items]
        c     = 1 / (1 + np.exp(-x[2 * n_items:3 * n_items]))
        d     = 1 / (1 + np.exp(-x[3 * n_items:4 * n_items]))
        theta = x[4 * n_items:]
        logits = a[None, :] * (theta[:, None] - b[None, :])
        sig    = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        P      = np.clip(c[None, :] + (d - c)[None, :] * sig, eps, 1 - eps)
        return -(Y * np.log(P) + (1 - Y) * np.log(1 - P)).sum()

    res   = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 500})
    x     = res.x
    a     = x[:n_items]
    b     = x[n_items:2 * n_items]
    c     = 1 / (1 + np.exp(-x[2 * n_items:3 * n_items]))
    d     = 1 / (1 + np.exp(-x[3 * n_items:4 * n_items]))
    theta = x[4 * n_items:]
    return a, b, c, d, float(theta.mean())


def _fisher_select(
    data: Data,
    models: list,
    n_items: int,
    n_models: int,
    budgets: Budgets,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    mean_theta: float,
) -> ModelScoresAtBudget:
    """Rank items by 4PL Fisher information and return scores."""
    a, b, c, d = a.astype(np.float64), b.astype(np.float64), c.astype(np.float64), d.astype(np.float64)
    P = c + (d - c) / (1.0 + np.exp(np.clip(-a * (mean_theta - b), -500, 500)))
    P = np.clip(P, 1e-6, 1 - 1e-6)
    d_minus_c = np.clip(d - c, 1e-6, None)
    # high a and P near middle -> high Fisher info
    fisher = (a ** 2 * (P - c) ** 2 * (d - P) ** 2) / (d_minus_c ** 2 * P * (1 - P))
    order = np.argsort(fisher)[::-1]  # descending

    results: ModelScoresAtBudget = []
    for budget in budgets:
        k = max(1, min(budget // n_models, n_items))
        if k >= n_items:
            results.append({m: [item["scores"][m] for item in data] for m in models})
            continue
        selected = order[:k].tolist()
        results.append({m: [data[i]["scores"][m] for i in selected] for m in models})
    return results


def _prepare_training_matrix(data: Data, models: list) -> tuple:
    """Build binarized matrix from scores_metrics."""
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    S = np.array([[item["scores_metrics"][m][metric] for m in models] for item in data])
    n_items, n_models = S.shape
    Y = _binarize(S)  # (n_items, n_models)
    return Y.T, n_items, n_models   


def lost_in_benchmarks_budgets(data: Data, budgets: Budgets, method: str) -> ModelScoresAtBudget:
    models = list(data[0]["scores"].keys())
    Y, n_items, n_models = _prepare_training_matrix(data, models)
    if method == "psn":
        a, b, c, d, mean_theta = _fit_psn_irt(Y)
    elif method == "4pl_baseline":
        a, b, c, d, mean_theta = _fit_4pl_irt(Y)
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return _fisher_select(data, models, n_items, n_models, budgets, a, b, c, d, mean_theta)
