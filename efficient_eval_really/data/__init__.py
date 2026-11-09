import functools
from collections.abc import Callable
from dataclasses import dataclass

from efficient_eval_really.methods import Data


@dataclass(frozen=True)
class AnnotatorDrift:
    """Simulates human annotation that changes over (annotation) time.

    Items are annotated in dataset order, so item index i corresponds to time t = i / (items - 1) in [0, 1].
    Each item is annotated (for all models) by one annotator drawn from the current pool.
    The observed score is clip(true + annotator_bias - strictness_drift * t + N(0, noise_sd(t)), 0, 1),
    where noise_sd(t) = annotator_noise + inconsistency_drift * t.

    The clean (pre-annotation) score is kept as ground truth in `scores_true`.
    """

    n_annotators: int = 5
    # sd of the per-annotator constant bias (drawn when the annotator joins the pool)
    bias_scale: float = 0.0
    # per-annotation noise sd at t=0 (per-annotator, drawn uniformly from this range)
    noise_range: tuple[float, float] = (0.05, 0.05)
    # total downward shift of observed scores from first to last item (annotators get stricter)
    strictness_drift: float = 0.0
    # total increase in noise sd from first to last item (annotators get less consistent)
    inconsistency_drift: float = 0.0
    # probability, per item, that a random annotator in the pool is replaced by a fresh one
    replacement_rate: float = 0.0


def load_data_synth(
    seed=0,
    models=20,
    items=500,
    heteroscedastic=False,
    bins=None,
    embedding_dim=64,
    item_heteroscedastic: float = 0.0,
    metric_noise: float = 0.05,
    metric_group_noise: tuple[float, ...] | None = None,
    annotator_drift: AnnotatorDrift | None = None,
) -> Data:
    """
    Synthetic scenario generator. With default arguments the output is identical to the original generator.

    - heteroscedastic: model-dependent noise (noise sd = model latent ability).
    - item_heteroscedastic: item-dependent noise. Noise sd is multiplied by exp(gamma * difficulty) where
      difficulty = -item_latent, normalized so that the average multiplier is 1. 0 turns it off.
    - metric_noise: sd of the automated metric's noise around the true score (metric reliability).
    - metric_group_noise: distribution-dependent metric reliability. Items are split uniformly at random into
      len(metric_group_noise) groups and the metric noise sd for group k is metric_group_noise[k].
      The group is exposed via the `domain` field ("synth/g{k}"). Overrides metric_noise.
    - annotator_drift: corrupts the observed human scores with time-varying annotator behaviour,
      see AnnotatorDrift. Ground truth is stored in `scores_true`.
    """
    import numpy as np

    random = np.random.RandomState(seed)
    # separate streams so that the new options do not change the base data
    random_groups = np.random.RandomState(seed + 1_000)
    random_annot = np.random.RandomState(seed + 2_000)
    if bins is not None:
        bins = np.asarray(bins, dtype=float)

    def _bin(x):
        # get closest bin, not digitize
        return bins[np.argmin(np.abs(bins - x))] if bins is not None else x

    models_latent = np.clip(random.normal(loc=0.70, scale=0.25, size=models), 0, 1)
    items_latent = random.normal(loc=0, scale=1, size=items)

    model_latent_mean = np.mean(models_latent)

    # Fixed random projection matrix for generating synthetic item embeddings
    W = random.normal(size=(models, embedding_dim))

    # item-dependent noise multiplier, harder items (lower latent) are noisier
    # E[exp(g*Z)] = exp(g^2/2) for Z~N(0,1), so normalize to keep the average noise level comparable
    items_noise_mult = np.exp(-item_heteroscedastic * items_latent - item_heteroscedastic**2 / 2)

    if metric_group_noise is not None:
        items_group = random_groups.randint(0, len(metric_group_noise), size=items)
    else:
        items_group = None

    if annotator_drift is not None:
        ad = annotator_drift

        def _new_annotator():
            return (random_annot.normal(0, ad.bias_scale), random_annot.uniform(*ad.noise_range))

        annotators = [_new_annotator() for _ in range(ad.n_annotators)]

    data_out: Data = []
    for item_i, item_latent in enumerate(items_latent):
        scores_dict = {}
        scores_true_dict = {}
        scores_metric_dict = {}

        if annotator_drift is not None:
            t = item_i / max(1, items - 1)
            if random_annot.rand() < ad.replacement_rate:
                annotators[random_annot.randint(len(annotators))] = _new_annotator()
            annot_bias, annot_noise = annotators[random_annot.randint(len(annotators))]
            annot_shift = annot_bias - ad.strictness_drift * t
            annot_noise = annot_noise + ad.inconsistency_drift * t

        cur_metric_noise = metric_group_noise[items_group[item_i]] if items_group is not None else metric_noise

        for model_i, model_latent in enumerate(models_latent):
            if heteroscedastic:
                error_scale = model_latent
            else:
                error_scale = model_latent_mean
            error = random.normal(loc=0, scale=error_scale * items_noise_mult[item_i])
            score_raw = np.clip(model_latent + item_latent + error, 0, 1)
            score = _bin(score_raw)
            model_name = f"model_{model_i + 1}"
            scores_true_dict[model_name] = float(score)
            scores_metric_dict[model_name] = {"metric": float(np.clip(score + random.normal(loc=0, scale=cur_metric_noise), 0, 1))}

            if annotator_drift is not None:
                # annotators perceive the continuous quality, then report on the same scale (bins)
                score_obs = _bin(np.clip(score_raw + annot_shift + random_annot.normal(0, annot_noise), 0, 1))
                scores_dict[model_name] = float(score_obs)
            else:
                scores_dict[model_name] = float(score)

        score_vec = np.array([scores_true_dict[f"model_{model_i + 1}"] for model_i in range(models)], dtype=float)
        emb = score_vec @ W + random.normal(scale=0.05, size=embedding_dim)
        emb_norm = float(np.linalg.norm(emb))
        emb = (emb / (emb_norm if emb_norm > 0 else 1.0)).tolist()

        item_out = {
            "scores": scores_dict,
            "scores_metrics": scores_metric_dict,
            "cost": 1,
            "embedding": emb,
            "domain": f"synth/g{items_group[item_i]}" if items_group is not None else "synth",
        }
        if annotator_drift is not None:
            item_out["scores_true"] = scores_true_dict
        data_out.append(item_out)  # type: ignore
    return data_out


def load_data_subset2evaluate_translation():
    import subset2evaluate.utils

    data = subset2evaluate.utils.load_data_wmt("wmt25", "en-cs_CZ")
    data_new = []
    for line in data:
        data_new.append(
            {
                "src": line["src"],
                "tgt": line["tgt"],
                "scores": {model: line["scores"][model]["human"] / 100 for model in line["scores"]},
                "scores_metrics": {model: {"metric": line["scores"][model]["MetricX-25"]} for model in line["scores"]},
                "cost": 1,
                "domain": "wmt25/en-cs",
            }
        )
    return data_new


def load_data_subset2evaluate_summeval():
    import subset2evaluate.utils

    data = subset2evaluate.utils.load_data_summeval(normalize=False, load_extra=False)
    data_new = []
    for line in data:
        data_new.append(
            {
                "src": line["src"],
                "tgt": line["tgt"],
                "scores": {model: line["scores"][model]["human_overall"] for model in line["scores"]},
                "scores_metrics": {model: {"metric": line["scores"][model]["unieval_overall"]} for model in line["scores"]},
                "cost": 1,
                "domain": "summeval",
            }
        )
    return data_new


load_data_synth_binary: Callable[..., Data] = functools.partial(load_data_synth, bins=[0, 1])
load_data_synth_likert: Callable[..., Data] = functools.partial(load_data_synth, bins=[0, 0.25, 0.5, 0.75, 1])
load_data_synth_hetero: Callable[..., Data] = functools.partial(load_data_synth, heteroscedastic=True)
load_data_synth_homo: Callable[..., Data] = functools.partial(load_data_synth, heteroscedastic=False)
load_data_synth_5models: Callable[..., Data] = functools.partial(load_data_synth, models=5, heteroscedastic=False)
load_data_synth_200models: Callable[..., Data] = functools.partial(load_data_synth, models=200, heteroscedastic=False)

# item-dependent heteroscedasticity: harder items are noisier
load_data_synth_item_hetero: Callable[..., Data] = functools.partial(load_data_synth, item_heteroscedastic=0.5)

# automated metric reliability: increasingly noisy metric (default is 0.05)
load_data_synth_metric_noise_010: Callable[..., Data] = functools.partial(load_data_synth, metric_noise=0.10)
load_data_synth_metric_noise_020: Callable[..., Data] = functools.partial(load_data_synth, metric_noise=0.20)
load_data_synth_metric_noise_040: Callable[..., Data] = functools.partial(load_data_synth, metric_noise=0.40)

# distribution-dependent metric reliability: metric is good on half of the items and poor on the other half
load_data_synth_metric_groups: Callable[..., Data] = functools.partial(load_data_synth, metric_group_noise=(0.02, 0.40))

# annotator drift
load_data_synth_annot_strictness: Callable[..., Data] = functools.partial(
    load_data_synth, annotator_drift=AnnotatorDrift(strictness_drift=0.3)
)
load_data_synth_annot_inconsistency: Callable[..., Data] = functools.partial(
    load_data_synth, annotator_drift=AnnotatorDrift(inconsistency_drift=0.3)
)
load_data_synth_annot_replacement: Callable[..., Data] = functools.partial(
    load_data_synth, annotator_drift=AnnotatorDrift(bias_scale=0.15, noise_range=(0.02, 0.20), replacement_rate=0.02)
)


# TODO: some selector for src-based? or maybe the method should return None if it isn't supported?
# TODO: mark how many times the dataset should be run and averaged
_DATA: dict[str, Callable[[], Data]] = {
    "Binary (s)": load_data_synth_binary,
    "Likert (s)": load_data_synth_likert,
    "Heterosc. (s)": load_data_synth_hetero,
    "Homosc. (s)": load_data_synth_homo,
    "5 models": load_data_synth_5models,
    "200 models": load_data_synth_200models,
    "Item heterosc. (s)": load_data_synth_item_hetero,
    "Metric noise 0.1 (s)": load_data_synth_metric_noise_010,
    "Metric noise 0.2 (s)": load_data_synth_metric_noise_020,
    "Metric noise 0.4 (s)": load_data_synth_metric_noise_040,
    "Metric by group (s)": load_data_synth_metric_groups,
    "Annot. strictness (s)": load_data_synth_annot_strictness,
    "Annot. inconsistency (s)": load_data_synth_annot_inconsistency,
    "Annot. replacement (s)": load_data_synth_annot_replacement,
    "Translation": load_data_subset2evaluate_translation,
    # "Summarization": load_data_subset2evaluate_summeval,
}


def get_data_dict_all() -> dict[str, Data]:
    return {k: v() for k, v in _DATA.items()}


def shuffle_items(data: Data, seed: int) -> Data:
    import numpy as np

    rng = np.random.default_rng(seed)
    return [data[i] for i in rng.permutation(len(data))]
