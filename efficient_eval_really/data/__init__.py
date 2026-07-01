import functools
from efficient_eval_really.methods import Data
from typing import Callable

def load_data_synth(
    seed=0, models=20, items=500, heteroscedastic=False, bins=None,
) -> Data:
    import numpy as np

    random = np.random.RandomState(seed)

    models_latent = np.clip(random.normal(loc=0.70, scale=0.25, size=models), 0, 1)
    items_latent = random.normal(loc=0, scale=1, size=items)

    model_latent_mean = np.mean(models_latent)

    data_out: Data = []
    for item_latent in items_latent:
        scores_dict = {}
        scores_metric_dict = {}
        for model_i, model_latent in enumerate(models_latent):
            if heteroscedastic:
                error = random.normal(loc=0, scale=model_latent)
            else:
                error = random.normal(loc=0, scale=model_latent_mean)
            score = np.clip(model_latent + item_latent + error, 0, 1)
            if bins:
                # get closest bin, not digitize
                score = bins[np.argmin(np.abs(bins - score))]
            scores_dict[f"model_{model_i + 1}"] = float(score)
            scores_metric_dict[f"model_{model_i + 1}"] = {"metric": float(np.clip(score + random.normal(loc=0, scale=0.05), 0, 1))}
        data_out.append({"scores": scores_dict, "scores_metrics": scores_metric_dict, "cost": 1, "domain": "synth"})
    return data_out

def load_data_subset2evaluate_translation():
    import subset2evaluate.utils

    data = subset2evaluate.utils.load_data_wmt("wmt25", "en-cs_CZ")
    data_new = []
    for line in data:
        data_new.append({
            "src": line["src"], "tgt": line["tgt"],
            "scores": {model: line["scores"][model]["human"]/100 for model in line["scores"]},
            "scores_metrics": {model: {"metric": line["scores"][model]["MetricX-25"]} for model in line["scores"]},
            "cost": 1,
            "domain": "wmt25/en-cs",
        })
    return data_new

def load_data_subset2evaluate_summeval():
    import subset2evaluate.utils

    data = subset2evaluate.utils.load_data_summeval(normalize=False, load_extra=True)
    data_new = []
    for line in data:
        data_new.append({
            "src": line["src"], "tgt": line["tgt"],
            "scores": {model: line["scores"][model]["human_overall"] for model in line["scores"]},
            "scores_metrics": {model: {"metric": line["scores"][model]["unieval_overall"]} for model in line["scores"]},
            "cost": 1,
            "domain": "summeval",
        })
    return data_new


load_data_synth_binary : Callable[..., Data] = functools.partial(load_data_synth, bins=[0, 1])
load_data_synth_likert : Callable[..., Data] = functools.partial(load_data_synth, bins=[0, 0.25, 0.5, 0.75, 1])
load_data_synth_hetero : Callable[..., Data] = functools.partial(load_data_synth, heteroscedastic=True)
load_data_synth_homo : Callable[..., Data] = functools.partial(load_data_synth, heteroscedastic=False)
load_data_synth_5models : Callable[..., Data] = functools.partial(load_data_synth, models=5, heteroscedastic=False)
load_data_synth_200models : Callable[..., Data] = functools.partial(load_data_synth, models=200, heteroscedastic=False)


# TODO: some selector for src-based?
DATA_FN = lambda: {
    "Binary (s)": load_data_synth_binary(),
    "Likert (s)": load_data_synth_likert(),
    "Heterosc. (s)": load_data_synth_hetero(),
    "Homosc. (s)": load_data_synth_homo(),
    "5 models": load_data_synth_5models(),
    "200 models": load_data_synth_200models(),
    "Translation": load_data_subset2evaluate_translation(),
    "Summarization": load_data_subset2evaluate_summeval()
}