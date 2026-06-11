import subset2evaluate.select_subset
from efficient_eval_really.methods import Data, Budget, Budgets, ModelScoresSubset, ModelScoresAtBudget
import copy

def subset2evaluate_to_ours_budgets(data: Data, budgets: Budgets, method: str, **kwargs) -> ModelScoresAtBudget:
    data = copy.deepcopy(data)
    for item in data:
        item["scores_true"] = item["scores"] # type: ignore
        item["scores"] = item["scores_metrics"] # type: ignore
    data_sorted: Data = subset2evaluate.select_subset.basic(data=data, method=method, **kwargs) # type: ignore
    model_scores: ModelScoresAtBudget = []
    models = data_sorted[0]["scores"].keys()
    # we're modifying the data, so let's copy it

    for budget in budgets:
        # the budget is inflated by the number of models
        budget = int(budget / len(models))
        model_scores.append({
            model: [item["scores_true"][model] for item in data_sorted[:budget]] # type: ignore
            for model in models
        })
    return model_scores

def subset2evaluate_to_ours(data: Data, budget: Budget, method: str, **kwargs) -> ModelScoresSubset:
    return subset2evaluate_to_ours_budgets(data=data, budgets=[budget], method=method, **kwargs)[0]