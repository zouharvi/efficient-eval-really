from typing import TypedDict

Model = str
BudgetFrac = float
Budget = int
Budgets = list[int]

class DataItem(TypedDict):
    src: str
    tgt: dict[Model, str]
    scores: dict[Model, float]

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