import functools

def load_data_synth(
    seed=0, models=100, items=500, heteroscedastic=False, bins=None,
) -> list[dict[str, list[dict]]]:
    import numpy as np

    random = np.random.RandomState(seed)

    models_latent = np.clip(random.normal(loc=0.70, scale=0.25, size=models), 0, 1)
    items_latent = random.normal(loc=0, scale=1, size=items)

    model_latent_mean = np.mean(models_latent)

    data_out = []
    for item_latent in items_latent:
        scores_dict = {}
        for model_i, model_latent in enumerate(models_latent):
            if heteroscedastic:
                error = random.normal(loc=0, scale=model_latent)
            else:
                error = random.normal(loc=0, scale=model_latent_mean)
            score = np.clip(model_latent + item_latent + error, 0, 1)
            if bins:
                # get closest bin, not digitize
                score = bins[np.argmin(np.abs(bins - score))]
            scores_dict[f"model_{model_i + 1}"] = {"human": float(score)}
        data_out.append({"scores": scores_dict, "cost": 1, "domain": "synth"})
    return data_out


load_data_synth_binary = functools.partial(load_data_synth, bins=[0, 1])
load_data_synth_likert = functools.partial(load_data_synth, bins=[0, 0.25, 0.5, 0.75, 1])
load_data_synth_hetero = functools.partial(load_data_synth, heteroscedastic=True)
load_data_synth_homo = functools.partial(load_data_synth, heteroscedastic=False)

