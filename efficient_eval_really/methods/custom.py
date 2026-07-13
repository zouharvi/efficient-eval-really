from efficient_eval_really.methods import Data, Budgets, ModelScoresAtBudget
from typing import Callable
import random
import statistics

def weighted_sampling_with_priors(data: Data, budgets: Budgets, gamma: float = 0.95, coldstart=5) -> ModelScoresAtBudget:
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

    def _scores_to_ranking(model_scores) -> dict[str, int]:
        model_estimate = {
            model: statistics.mean(model_scores[model])
            for model in models
        }
        # output rankg 0, 1, 2, ... with 0 being the best model
        return {model: rank for rank, model in enumerate(sorted(model_estimate, key=model_estimate.get, reverse=True))} # type: ignore

        
    ranking_metrics = _scores_to_ranking({model: [x["scores_metrics"][model]["metric"] for x in data] for model in models})

    output = []
    # active learning phase
    while budgets:
        if cost >= budgets[0] or not models:
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})
            continue

        ranking_real = _scores_to_ranking(model_scores)
        # TODO: this needs to be better
        # decay gamma as we get closer to the budget
        gamma_now = gamma ** (cost / budgets[0])
        model_estimate = {
            model: (gamma_now * ranking_real[model]) + ((1 - gamma_now) * ranking_metrics[model])
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


def weighted_sampling_with_priors_corrected(data: Data, budgets: Budgets, coldstart: int = 5, batch_size: int = 10) -> ModelScoresAtBudget:
    """
    A variant of weighted_sampling_with_priors that corrects for the bias introduced by the sampling method.
    This method uses a linear regression model to predict the missing scores based on the available scores and metrics.
    """
    import sklearn.linear_model

    data = list(data)
    # cold start phase
    model_scores = {
        model: [x["scores"][model] for x in data[:coldstart]]
        for model in data[0]["scores"]
    }
    models = list(data[0]["scores"])
    cost = sum(x["cost"] for x in data[:coldstart]) * len(models)

    # one-hot encoding
    one_hot_item = {item_i: [0] * len(data) for item_i in range(len(data))}
    for i in range(len(data)):
        one_hot_item[i][i] = 1
    one_hot_model = {model: [0] * len(models) for model in models}
    for i, model in enumerate(models):
        one_hot_model[model][i] = 1

    output = []
    # active learning phase
    while budgets:
        if cost >= budgets[0] or not models:
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})
            continue

        estimator_model = sklearn.linear_model.LinearRegression()
        # TODO: we need to make this faster
        estimator_model.fit(
            [
                # encode item_i one-hot
                [x["scores_metrics"][model]["metric"]] + one_hot_item[item_i]
                for model in list(data[0]["scores"])
                for item_i, x in enumerate(data[:len(model_scores[model])])
            ],
            [
                x["scores"][model]
                for model in list(data[0]["scores"])
                for x in data[:len(model_scores[model])]
            ],
        )
        # predict missing values
        estimated_scores = {
            model: list(estimator_model.predict([
                [x["scores_metrics"][model]["metric"]] + one_hot_item[item_i]
                for item_i, x in enumerate(data)
            ])) # type: ignore
            for model in models
        }

        # skip recomputation sometimes
        for _ in range(batch_size):
            model_estimate = {
                # use predictions for the missing values
                model: statistics.mean(model_scores[model] + estimated_scores[model][len(model_scores[model]):])
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
            if not models:
                break
    return output
