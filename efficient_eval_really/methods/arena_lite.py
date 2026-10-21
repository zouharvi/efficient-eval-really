import numpy as np

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def arena_lite_budgets(data: Data, budgets: Budgets, seed: int = 0, **kwargs) -> ModelScoresAtBudget:
    """Arena-Lite: Efficient LLM Evaluation via Tournament-Based Direct Comparisons.

    Algorithm based on Son et al. (EMNLP 2024):
    'Arena-Lite: Efficient and Reliable Large Language Model Evaluation via Tournament-Based Direct Comparisons'

    Conducts single-elimination tournament brackets across evaluation prompts to rank models.
    Fair budget accounting & 0-cost battle harvesting (Option A):
    1. Evaluating a model on a prompt for the first time incurs 1 budget unit.
    2. Whenever a model is evaluated on a prompt, it automatically battles against all other
       models already evaluated on that prompt for 0 additional generation cost.
    3. Tournament winners advance to subsequent rounds.
    4. Aggregates observed model scores and match win rates across the budget schedule.
    """
    rng = np.random.default_rng(seed)
    models = list(data[0]["scores"].keys())
    n_models = len(models)
    n_items = len(data)

    scores_mat = np.array([[item["scores"][m] for m in models] for item in data], dtype=float)
    costs_arr = np.array([item["cost"] for item in data], dtype=float)

    wins = np.zeros(n_models, dtype=float)
    matches = np.zeros(n_models, dtype=float)
    evaluated_per_item = [set() for _ in range(n_items)]
    model_scores = [[] for _ in range(n_models)]
    current_cost = 0.0

    sorted_budgets = sorted(set(budgets))
    results_map = {}

    bracket_size = 1 << int(np.ceil(np.log2(n_models)))
    n_byes = bracket_size - n_models

    item_idx = 0
    budget_idx = 0

    def evaluate_model_on_item(m: int, item: int):
        nonlocal current_cost
        if m in evaluated_per_item[item]:
            return
        current_cost += costs_arr[item]
        model_scores[m].append(scores_mat[item, m])

        # Automatic 0-cost battles against all co-evaluated models on this item
        s_m = scores_mat[item, m]
        for other in evaluated_per_item[item]:
            s_other = scores_mat[item, other]
            matches[m] += 1
            matches[other] += 1
            if s_m > s_other:
                wins[m] += 1.0
            elif s_other > s_m:
                wins[other] += 1.0
            else:
                wins[m] += 0.5
                wins[other] += 0.5

        evaluated_per_item[item].add(m)

    while budget_idx < len(sorted_budgets):
        target_budget = sorted_budgets[budget_idx]

        if current_cost >= target_budget or item_idx >= n_items:
            rates = np.where(matches > 0, wins / matches, 0.5)
            results_map[target_budget] = {m: list(model_scores[i]) if len(model_scores[i]) > 0 else [float(rates[i])] for i, m in enumerate(models)}
            budget_idx += 1
            continue

        participants = list(range(n_models)) + [-1] * n_byes
        rng.shuffle(participants)

        current_round = participants
        while len(current_round) > 1 and current_cost < target_budget:
            next_round = []
            for i in range(0, len(current_round), 2):
                m1 = current_round[i]
                m2 = current_round[i + 1] if i + 1 < len(current_round) else -1

                if m1 == -1 and m2 == -1:
                    next_round.append(-1)
                elif m1 == -1:
                    next_round.append(m2)
                elif m2 == -1:
                    next_round.append(m1)
                else:
                    evaluate_model_on_item(m1, item_idx)
                    if current_cost < target_budget:
                        evaluate_model_on_item(m2, item_idx)

                    s1 = scores_mat[item_idx, m1]
                    s2 = scores_mat[item_idx, m2]

                    if s1 > s2:
                        next_round.append(m1)
                    elif s2 > s1:
                        next_round.append(m2)
                    else:
                        next_round.append(m1 if rng.random() < 0.5 else m2)
            current_round = next_round

        item_idx += 1

    rates = np.where(matches > 0, wins / matches, 0.5)
    for b in sorted_budgets:
        if b not in results_map:
            results_map[b] = {m: list(model_scores[i]) if len(model_scores[i]) > 0 else [float(rates[i])] for i, m in enumerate(models)}

    return [results_map[b] for b in budgets]
