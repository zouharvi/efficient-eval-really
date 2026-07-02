import os
import statistics
import efficient_eval_really.utils
import efficient_eval_really.methods
from efficient_eval_really.data import get_data_dict_all
import numpy as np
import json
import tqdm
import argparse

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
for method_name, method_fn in tqdm.tqdm(list(efficient_eval_really.methods.METHODS_BUDGETS.items())):
    results_out[method_name] = results_out.get(method_name, {})
    for data_name, data in tqdm.tqdm(DATA.items()):
        # skip if already computed
        if not args.no_reuse and data_name in results_out[method_name]:
            continue

        budgets = [int(b * len(data) * len(data[0]['scores'])) for b in BUDGETS_FRAC]
        model_scores_all = {
            model: [item['scores'][model] for item in data] for model in data[0]['scores'] 
        }
        parallelization_queue.append((method_name, method_fn, data_name, data, budgets, model_scores_all))

# do the computation
def _my_compute_function_coz_lambdas_dont_pickle(x):
    method_name, method_fn, data_name, data, budgets, model_scores_all = x
    return (method_name, method_fn(data, budgets), data_name, model_scores_all, budgets)
if args.workers > 0:
    import multiprocessing
    with multiprocessing.Pool(args.workers) as pool:
        results = list(tqdm.tqdm(pool.imap_unordered(
            _my_compute_function_coz_lambdas_dont_pickle, parallelization_queue),
            total=len(parallelization_queue)
        ))
else:
    results = [
        _my_compute_function_coz_lambdas_dont_pickle(x)
        for x in tqdm.tqdm(parallelization_queue)
    ]

# collate the results
for method_name, model_scores_subset_at_budgets, data_name, model_scores_all, budgets in tqdm.tqdm(results):
    results_local = []
    for budget, model_scores_subset_at_budget in zip(budgets, model_scores_subset_at_budgets):
        results_local.append(efficient_eval_really.utils.meta_evaluate_scores_subset(model_scores_subset_at_budget, model_scores_all))

    results_out[method_name][data_name] = {}
    for key in results_local[0].keys():
        results_out[method_name][data_name][key] = statistics.mean([result[key] for result in results_local])

with open("computed/01-results_all.json", "w") as f:
    json.dump(results_out, f, indent=2)