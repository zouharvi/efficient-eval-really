from typing import Callable, TypedDict, NotRequired, Union
import functools


Model = str
BudgetFrac = float
Budget = int
Budgets = list[int]

class DataItem(TypedDict):
    scores: dict[Model, float]
    scores_metrics: NotRequired[dict[str, dict[Model, float]]]
    cost: float
    src: NotRequired[str]
    tgt: NotRequired[dict[Model, str]]
    domain: NotRequired[str]


Data = list[DataItem]
# don't use DataSubset
DataSubset = Data

ModelScores = dict[Model, list[float]]
ModelScoresSubset = ModelScores

ModelScoresAtBudget = list[ModelScoresSubset]

"""
Technically, all methods output ModelScoresSubset

def fn(data: Data, budget: Budget) -> ModelScoresSubset:
    pass
    

One might be tempted to do subset selection and return the selected items, like subset2evaluate does:

def fn(data: Data, budget: Budget) -> DataSubset:
    pass

However, this is incompatible with iterative model+item selection, so instead use ModelScoresSubset as above.

Furthermore, many methods are incremental, so we can save some compute by computing across multiple budgets

def fn(data: Data, budgets: Budgets) -> ModelScoresAtBudget:
    pass
"""


# define all methods
from efficient_eval_really.methods.subset2evaluate import subset2evaluate_to_ours_budgets
from efficient_eval_really.methods.evaluation_bandit import evaluation_bandit_to_ours_budgets

METHODS_BUDGETS: dict[str, Callable[[Data, Budgets], ModelScoresAtBudget]] = {
    "subset2evaluate_metricvar": functools.partial(subset2evaluate_to_ours_budgets, method="metric_var", metric="metric"),
    "subset2evaluate_metricavg": functools.partial(subset2evaluate_to_ours_budgets, method="metric_avg", metric="metric"),
    "subset2evaluate_metriccons": functools.partial(subset2evaluate_to_ours_budgets, method="metric_cons", metric="metric"),
    "evaluation_bandit_ucb": functools.partial(evaluation_bandit_to_ours_budgets, method="upper_confidence_bound"),
    "evaluation_bandit_uniform": functools.partial(evaluation_bandit_to_ours_budgets, method="uniform"),
}