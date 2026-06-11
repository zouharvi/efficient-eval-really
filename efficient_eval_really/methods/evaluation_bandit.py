from efficient_eval_really.methods import Data, Budget, Budgets, ModelScoresSubset, ModelScoresAtBudget
import evaluation_bandit.algorithms

def evaluation_bandit_to_ours(data: Data, budget: Budget, method: str, **kwargs) -> ModelScoresSubset:
    if method == "upper_confidence_bound":
        return evaluation_bandit.algorithms.upper_confidence_bound(data=data, budgets=[budget])[0]
    else:
        raise ValueError(f"Unknown method_name: {method}")

def evaluation_bandit_to_ours_budgets(data: Data, budgets: Budgets, method: str, **kwargs) -> ModelScoresAtBudget:
    if method == "upper_confidence_bound":
        return evaluation_bandit.algorithms.upper_confidence_bound(data=data, budgets=budgets)
    elif method == "uniform":
        return evaluation_bandit.algorithms.uniform(data=data, budgets=budgets)
    else:
        raise ValueError(f"Unknown method_name: {method}")