import functools
from collections.abc import Callable
from typing import NotRequired, TypedDict, Union

Model = str
BudgetFrac = float
Budget = int
Budgets = list[int]

class DataItem(TypedDict):
    scores: dict[Model, float]
    scores_metrics: dict[str, dict[Model, float]]
    cost: float
    embedding: NotRequired[list[float]]
    src: NotRequired[str]
    tgt: NotRequired[dict[Model, str]]
    domain: NotRequired[str]


Data = list[DataItem]
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
from efficient_eval_really.methods.active_testing import active_testing_budgets
from efficient_eval_really.methods.anchor_points import anchor_points_budgets
from efficient_eval_really.methods.atlas import atlas_budgets
from efficient_eval_really.methods.bias_bounded import bias_bounded_budgets
from efficient_eval_really.methods.confident_rankings import confident_rankings_budgets
from efficient_eval_really.methods.custom import (
    weighted_sampling_with_priors,
    weighted_sampling_with_priors_corrected,
)
from efficient_eval_really.methods.efficient_benchmarking_apis import (
    efficient_benchmarking_apis_budgets,
)
from efficient_eval_really.methods.evaluation_bandit import (
    evaluation_bandit_to_ours_budgets,
)
from efficient_eval_really.methods.feature_selection_regression import (
    feature_selection_regression_budgets,
)
from efficient_eval_really.methods.iq_test_llms import iq_test_llms_budgets
from efficient_eval_really.methods.irt_chatbots import irt_chatbots_budgets
from efficient_eval_really.methods.just_enough_data import just_enough_data_budgets
from efficient_eval_really.methods.lost_in_benchmarks import lost_in_benchmarks_budgets
from efficient_eval_really.methods.misses_the_mark import misses_the_mark_budgets
from efficient_eval_really.methods.neyman_allocation import neyman_allocation_budgets
from efficient_eval_really.methods.optimal_eval_efficiency import (
    optimal_eval_efficiency_budgets,
)
from efficient_eval_really.methods.subset2evaluate import (
    subset2evaluate_to_ours_budgets,
)
from efficient_eval_really.methods.syrs_bandit import syrs_bandit_budgets
from efficient_eval_really.methods.tailored_benchmarks import (
    tailored_benchmarks_budgets,
)
from efficient_eval_really.methods.tiny_benchmarks import tiny_benchmarks_budgets
from efficient_eval_really.methods.valid_best_model import valid_best_model_budgets
from efficient_eval_really.methods.scales_plus_plus import scales_plus_plus_budgets
from efficient_eval_really.methods.smart_filtering import smart_filtering_budgets
from efficient_eval_really.methods.diffuse import diffuse_budgets
from efficient_eval_really.methods.select_label_evaluate import select_label_evaluate_budgets
from efficient_eval_really.methods.sparseeval import sparseeval_budgets
from efficient_eval_really.methods.cost_aware_gittins import cost_aware_gittins_budgets
from efficient_eval_really.methods.cost_efficient_estimation import cost_efficient_estimation_budgets
from efficient_eval_really.methods.faq import faq_budgets
from efficient_eval_really.methods.arena_lite import arena_lite_budgets
from efficient_eval_really.methods.adaptive_arena import adaptive_arena_budgets

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
    "Optimal Eval Efficiency": optimal_eval_efficiency_budgets,
    "IRT Chatbots": irt_chatbots_budgets,
    "SySRs Bandit": syrs_bandit_budgets,
    "Active Testing (LURE)": active_testing_budgets,
    "ATLAS (CAT)": atlas_budgets,
    "Just Enough Data (GST)": just_enough_data_budgets,
    "Misses the Mark (AIPW)": misses_the_mark_budgets,
    "Confident Rankings (Continuous CAT)": confident_rankings_budgets,
    "Valid Best Model (PULSE)": valid_best_model_budgets,
    "Bias Bounded (Submodular)": bias_bounded_budgets,
    "IQ Test for LLMs (Factor Analysis)": iq_test_llms_budgets,
    "Thompson Sampling (APIs)": efficient_benchmarking_apis_budgets,
    "Scales++": scales_plus_plus_budgets,
    "SMART Filtering": smart_filtering_budgets,
    "DiffUse": diffuse_budgets,
    "Select Label Evaluate": select_label_evaluate_budgets,
    "SparseEval": sparseeval_budgets,
    "Cost-Aware Gittins": cost_aware_gittins_budgets,
    "Cost-Efficient Estimation (OED)": cost_efficient_estimation_budgets,
    "FAQ": faq_budgets,
    "Arena-Lite": arena_lite_budgets,
    "Adaptive-Arena": adaptive_arena_budgets,
    "Adaptive-Arena (A-optimality)": adaptive_arena_budgets,
    "Weighted Sampling with Priors": weighted_sampling_with_priors,
    "Weighted Sampling w/ Priors": weighted_sampling_with_priors,
    # "Weighted Sampling w/ Priors Corrected": weighted_sampling_with_priors_corrected,
}