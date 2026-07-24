from typing import Callable, TypedDict, NotRequired, Union
import functools


Model = str
BudgetFrac = float
Budget = int
Budgets = list[int]

class DataItem(TypedDict):
    scores: dict[Model, float]
    scores_metrics: dict[str, dict[Model, float]]
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
from efficient_eval_really.methods.tailored_benchmarks import tailored_benchmarks_budgets
from efficient_eval_really.methods.anchor_points import anchor_points_budgets
from efficient_eval_really.methods.tiny_benchmarks import tiny_benchmarks_budgets
from efficient_eval_really.methods.feature_selection_regression import feature_selection_regression_budgets
from efficient_eval_really.methods.lost_in_benchmarks import lost_in_benchmarks_budgets
from efficient_eval_really.methods.neyman_allocation import neyman_allocation_budgets
from efficient_eval_really.methods.custom import weighted_sampling_with_priors, weighted_sampling_with_priors_corrected
from efficient_eval_really.methods.custom import weighted_sampling_with_priors, weighted_sampling_with_priors_corrected

METHODS_BUDGETS: dict[str, Callable[[Data, Budgets], ModelScoresAtBudget]] = {
    "Metric Variance": functools.partial(subset2evaluate_to_ours_budgets, method="metric_var", metric="metric"),
    "Metric Average (difficulty)": functools.partial(subset2evaluate_to_ours_budgets, method="metric_avg", metric="metric"),
    "Metric Consistency": functools.partial(subset2evaluate_to_ours_budgets, method="metric_cons", metric="metric"),
    "UCB": functools.partial(evaluation_bandit_to_ours_budgets, method="upper_confidence_bound"),
    "Weighted Sampling": functools.partial(evaluation_bandit_to_ours_budgets, method="weighted_sampling"),
    "Random (uniform)": functools.partial(evaluation_bandit_to_ours_budgets, method="uniform"),
    "Random (nonsquare)": functools.partial(evaluation_bandit_to_ours_budgets, method="uniform_nonsquare"),
    "Tailored Benchmarks": tailored_benchmarks_budgets,
    "Anchor Points": anchor_points_budgets,
    "Tiny Benchmarks (clustering)": functools.partial(tiny_benchmarks_budgets, method="clustering"),
    "Tiny Benchmarks (IRT)": functools.partial(tiny_benchmarks_budgets, method="irt"),
    "mRMR": feature_selection_regression_budgets,
    "Lost in Benchmarks (PSN-IRT)": functools.partial(lost_in_benchmarks_budgets, method="psn"),
    "Lost in Benchmarks (4PL Baseline)": functools.partial(lost_in_benchmarks_budgets, method="4pl_baseline"),
    "Neyman Allocation (proxy)": functools.partial(neyman_allocation_budgets, method="proxy"),
    "Neyman Allocation (oracle)": functools.partial(neyman_allocation_budgets, method="oracle"),
    "Neyman Allocation (proportional)": functools.partial(neyman_allocation_budgets, method="proportional"),
    "Weighted Sampling with Priors": weighted_sampling_with_priors,
    "Weighted Sampling w/ Priors": weighted_sampling_with_priors,
    # "Weighted Sampling w/ Priors Corrected": weighted_sampling_with_priors_corrected,
}