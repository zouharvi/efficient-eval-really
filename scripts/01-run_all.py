import os
import statistics
import efficient_eval_really.utils
import efficient_eval_really.methods
from efficient_eval_really.data import DATA_FN
import numpy as np
import json
import tqdm
import argparse

args = argparse.ArgumentParser()
args.add_argument("--reuse", action="store_true", help="Reuse results from computed/01-results_all.json if it exists")
args.add_argument("--workers", type=int, default=5, help="Number of workers for parallelization. 0 turns off parallelization.")
args = args.parse_args()

os.chdir(os.path.dirname(__file__) + "/../")
os.makedirs("computed", exist_ok=True)

# load previous results if they happen to exist
if os.path.exists("computed/01-results_all.json") and args.reuse:
    with open("computed/01-results_all.json", "r") as f:
        results_out = json.load(f)
else:
    results_out = {}

# instantiate and load data
DATA_FN = DATA_FN()
BUDGETS_FRAC = np.linspace(0.1, 1.0, 10)

# TODO: parallelize

for method_name, method_fn in tqdm.tqdm(list(efficient_eval_really.methods.METHODS_BUDGETS.items())):
    results_out[method_name] = results_out.get(method_name, {})
    for data_name, data in tqdm.tqdm(DATA_FN.items()):
        # skip if already computed
        if args.reuse and data_name in results_out[method_name]:
            continue

        budgets = [int(b * len(data) * len(data[0]['scores'])) for b in BUDGETS_FRAC]
        model_scores_all = {
            model: [item['scores'][model] for item in data] for model in data[0]['scores'] 
        }
        model_scores_subset_at_budgets = method_fn(data, budgets)
        results_local = []
        for budget, model_scores_subset_at_budget in zip(budgets, model_scores_subset_at_budgets):
            results_local.append(efficient_eval_really.utils.meta_evaluate_scores_subset(model_scores_subset_at_budget, model_scores_all))

        results_out[method_name][data_name] = {}
        for key in results_local[0].keys():
            results_out[method_name][data_name][key] = statistics.mean([result[key] for result in results_local])

with open("computed/01-results_all.json", "w") as f:
    json.dump(results_out, f, indent=2)