from efficient_eval_really.methods import Data, Budgets, ModelScoresAtBudget
from typing import Callable
import random
import statistics

def weighted_sampling_with_priors(data: Data, budgets: Budgets, gamma: float = 0.5, coldstart=5) -> ModelScoresAtBudget:
    """
    gamma [0, 1]: how much of the ranking is based on human evaluations only as opposed to metrics
    """
    data = list(data)
    # cold start phase
    model_scores = {
        model: [x["scores"][model] for x in data[:coldstart]]
        for model in data[0]["scores"]
    }
    models = list(data[0]["scores"])
    cost = sum(x["cost"] for x in data[:coldstart]) * len(models)

    output = []
    # active learning phase
    while budgets:
        if cost >= budgets[0] or not models:
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})
            continue

        # we want to estimate given the context of all models, not just the ones running
        model_estimate = {
            model: (gamma * statistics.mean(model_scores[model])) + ((1 - gamma) * statistics.mean(x["scores_metrics"][model]["metric"] for x in data))
            for model in models
        }
        models.sort(key=model_estimate.get, reverse=True)  # type: ignore

        model = random.choices(
            models,
            weights=[
                1 / (rank + 1)
                for model, rank in zip(models, range(len(models)))
            ],
            k=1,
        )[0]
        item = data[len(model_scores[model])]
        model_scores[model].append(item["scores"][model])
        cost += item["cost"]

        models = [
            model for model in model_scores if len(model_scores[model]) < len(data)
        ]
    return output


def weighted_sampling_with_priors_correction(data: Data, budgets: Budgets, gamma: float = 0.5) -> ModelScoresAtBudget:
    """
    TODO
    """
    raise NotImplementedError("This method is not yet implemented.")
