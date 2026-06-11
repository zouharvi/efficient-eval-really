import functools
from efficient_eval_really.methods import Data
from typing import Callable

def load_data_synth(
    seed=0, models=100, items=500, heteroscedastic=False, bins=None,
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


load_data_synth_binary : Callable[..., Data] = functools.partial(load_data_synth, bins=[0, 1])
load_data_synth_likert : Callable[..., Data] = functools.partial(load_data_synth, bins=[0, 0.25, 0.5, 0.75, 1])
load_data_synth_hetero : Callable[..., Data] = functools.partial(load_data_synth, heteroscedastic=True)
load_data_synth_homo : Callable[..., Data] = functools.partial(load_data_synth, heteroscedastic=False)

synth_kwargs = dict(seed=0, models=20, items=500)
DATA_FN = lambda: {
    "synth_binary": load_data_synth_binary(**synth_kwargs),
    "synth_likert": load_data_synth_likert(**synth_kwargs),
    "synth_hetero": load_data_synth_hetero(**synth_kwargs),
    "synth_homo": load_data_synth_homo(**synth_kwargs),
}