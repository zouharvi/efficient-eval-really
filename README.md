# Efficient Evaluation, Really?

> There is an epidemic of efficient evaluation methods that promise more robust and economical evaluation results with fractions of the cost.
> These works oftentimes ignore each other and break when used outside of their original setup.
> In this project, we try to unify these works with a general problem statement, see how much they generalize, and explain why and where they might be applicable.
> This should result in more confidence of practitioners to adopt efficient evaluation.

- `papers.toml` contains an annotated list of efficient evaluation papers

## Repository

First install this repository locally with `pip install -e .`.
The `efficient_eval_really/methods` includes all the methods with unified callable signature that is run on all the setups in `efficient_eval_really/data` when running
```bash
python3 scripts/01-run_all.py`
```

By default this script will re-use existing results and will not recompute them (can be turned off with `--no-reuse`).
It also CPU-parallelizes by default, which can be turned off with `--workers 0`.

## Contributing

This project is open to new contributors! Get in touch.