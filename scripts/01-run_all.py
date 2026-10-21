import argparse
import functools
import json
import os
import statistics
import time
import multiprocessing

import numpy as np
import tqdm

import efficient_eval_really.methods
import efficient_eval_really.utils
from efficient_eval_really.data import get_data_dict_all, shuffle_items

args = argparse.ArgumentParser()
args.add_argument("--no-reuse", action="store_true", help="Don't reuse results from computed/01-results_all.json even if it exists")
args.add_argument("--workers", type=int, default=10, help="Number of workers for parallelization. 0 turns off parallelization.")
args = args.parse_args()

os.chdir(os.path.dirname(__file__) + "/../")
os.makedirs("computed", exist_ok=True)

# load previous results if they happen to exist
if os.path.exists("computed/01-results_all.json") and not args.no_reuse:
    with open("computed/01-results_all.json", "r") as f:
        results_out = json.load(f)
else:
    results_out = {}

# instantiate and load data
DATA = get_data_dict_all()
BUDGETS_FRAC = np.linspace(0.1, 1.0, 10)

# prepare the parallelization queue
parallelization_queue = []
for method_name, method_fn in efficient_eval_really.methods.METHODS_BUDGETS.items():
    results_out[method_name] = results_out.get(method_name, {})
    for data_name, data in DATA.items():
        # skip if already computed
        if not args.no_reuse and data_name in results_out[method_name]:
            continue

        budgets = [int(b * len(data) * len(data[0]["scores"])) for b in BUDGETS_FRAC]
        model_scores_all = {model: [item["scores"][model] for item in data] for model in data[0]["scores"]}
        parallelization_queue.append((method_name, method_fn, data_name, data, budgets, model_scores_all))



# do the computation
def _my_compute_method_with_data(x):
    method_name, method_fn, data_name, data, budgets, model_scores_all = x

    if isinstance(method_fn, functools.partial):
        n_seeds = method_fn.keywords.get("seeds") or 1
        assert isinstance(n_seeds, int), f"{n_seeds} is type {type(n_seeds)} but should be int"
    else:
        n_seeds = 1

    t0 = time.perf_counter()
    all_seed_results = []
    for s in range(n_seeds):
        data_cur = shuffle_items(data, seed=s) if n_seeds > 1 else data
        model_scores_subset_at_budgets = method_fn(data_cur, budgets, seed=s)
        res_seed = [
            efficient_eval_really.utils.meta_evaluate_scores_subset(model_scores_subset_at_budget, model_scores_all)
            for model_scores_subset_at_budget in model_scores_subset_at_budgets
        ]
        all_seed_results.append(res_seed)
    runtime = time.perf_counter() - t0
    runtime_avg = (runtime / n_seeds) / len(budgets) if budgets else (runtime / n_seeds)

    # Average meta-eval metrics across seeds for each budget
    results_local = []
    for b_idx in range(len(budgets)):
        b_dict = {}
        for key in all_seed_results[0][b_idx]:
            vals = [all_seed_results[s][b_idx][key] for s in range(n_seeds)]
            valid_vals = [v for v in vals if not np.isnan(v)]
            b_dict[key] = float(np.mean(valid_vals)) if valid_vals else float("nan")
        results_local.append(b_dict)

    return (method_name, results_local, data_name, runtime_avg)

with multiprocessing.Pool(args.workers) as pool:
    results = list(
        tqdm.tqdm(
            pool.imap_unordered(_my_compute_method_with_data, parallelization_queue),
            total=len(parallelization_queue),
        )
    )

# collate the results
for method_name, results_local, data_name, runtime_avg in tqdm.tqdm(results):
    existing_entry = results_out.get(method_name, {}).get(data_name, {})
    existing_runtime = existing_entry.get("runtime") if isinstance(existing_entry, dict) else None

    results_out[method_name][data_name] = {}
    for key in results_local[0]:
        results_out[method_name][data_name][key] = statistics.mean([result[key] for result in results_local])

    if not args.no_reuse and existing_runtime is not None:
        results_out[method_name][data_name]["runtime"] = existing_runtime
    else:
        results_out[method_name][data_name]["runtime"] = runtime_avg

with open("computed/01-results_all.json", "w") as f:
    json.dump(results_out, f, indent=2)
