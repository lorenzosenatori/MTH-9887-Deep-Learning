# Attention Factors for Statistical Arbitrage: presentation and replication

Group 2 (Transformers), Assignment 1 of Deep Learning in Finance (MTH 9889 Data Science III),
Baruch MFE, Fall 2026. Chaithanya Pakala, Lorenzo Senatori, Daniel Tuzes.

**The paper.** Elliot L. Epstein, Rose Wang, Jaewon Choi and Markus Pelger (2025). "Attention
Factors for Statistical Arbitrage." *6th ACM International Conference on AI in Finance (ICAIF
'25)*, Singapore. [doi:10.1145/3768292.3770398](https://doi.org/10.1145/3768292.3770398),
[arXiv:2510.11616](https://arxiv.org/abs/2510.11616). A copy of the PDF (version 1) is in
[`paper/2510.11616v1.pdf`](paper/2510.11616v1.pdf).

## What is here

| Folder | What it holds |
|---|---|
| [`replication/`](replication/) | Our implementation of the paper (Deliverable 2): code, tests, results and write-up |
| [`slides/`](slides/) | The slides of the presentation (Deliverable 1), as a PDF |
| [`paper/`](paper/) | The paper, version 1 from arXiv, as a PDF |

Inside `replication/`:

```
src/attnfactors/      the code: data panel, characteristics, model, benchmarks, training
scripts/              entry points: download.py, build.py, run.py, report.py (round 1);
                      replicate_*.py (rounds 2 to 4)
tests/                34 tests, no data needed
results/              committed results: tables, figure, metrics.json and daily.csv of every run
notebooks/            results.ipynb: the main results, step by step
README.md             scope, choices the paper leaves open, data, how to run everything
RESULTS.md            round 1 beside the paper, and "What did not work"
README_ISSUES.md      rounds 2 and 3: the differences from the paper, checked one by one
DIFFERENCES.md        round 4: a universe rebuilt to match the firms the paper names
RUNS.md               every run, with its settings, results, commit and sources
AI_USAGE.md           how coding agents and LLMs were used
```

**Rounds.** The numbers below and on the slides are from round 1, the first implementation.
Rounds 2 to 4 reran the model with other readings of the paper: the loss exactly as printed,
inputs and PCA as the cited sources build them, and a universe ranked by company value. Finding
5 below comes from round 2. The rebuilt universe of round 4 moves the Sharpe ratios by at most
0.15, so it does not explain the gap to the paper.

## What we implemented, and what we left out

- **Implemented:** the full model, with attention factors built from 39 firm characteristics,
  a long convolution trading policy, and one loss with trading costs that trains both. Also both
  benchmarks: PCA with the same trading policy, and PCA with an OU threshold rule. All on the
  paper's sample: CRSP daily returns, the 500 largest stocks, retrained every year on an
  8-year window, out of sample 1998 to 2021.
- **Run:** the model with 8, 30 and 100 factors; the PCA baseline with 8 and 30; and the OU rule
  with 1 to 100. One row of Table 3 (without past returns). Variants of the choices the paper leaves open.
- **Left out:** other numbers of factors, the other rows of Table 3, and Figures 3 and 4 by name.
  One or two seeds per run. The paper's exact universe and characteristic formulas are not
  public, so ours are rebuilt.

## What we found

1. **Before costs the level reproduces in part:** a Sharpe ratio of 3.00 against the paper's
   3.97 with 30 factors. After costs we get 0.99 against 2.28.
2. **The ranking depends on how the baseline is built.** With the covariance PCA that the
   paper describes, the PCA baseline beats the model (3.47 against 3.00). With the
   correlation PCA of Avellaneda and Lee, the model wins, as in the paper.
3. **Two claims do not reproduce:** 100 factors do not help (3.04 before costs, 0.33 after),
   and removing past returns barely matters.
4. **After costs, almost all of the profit is earned before 2003.**
5. **The scale of the Sharpe ratio in the loss decides the level:** as printed (per day) the
   model earns about 0.1 after costs, and annualized about 1.0.

## Run it

```bash
cd replication
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python scripts/report.py --from-saved   # rebuilds the tables and the figure from committed results
python -m pytest tests -q               # the tests, no data needed
jupyter lab notebooks/results.ipynb     # the main results, step by step
```

Retraining needs CRSP and Compustat access through WRDS. The data are licensed and are not in
the repository; `replication/README.md` gives the full commands, from download to every run.

## Read next

- [`replication/notebooks/results.ipynb`](replication/notebooks/results.ipynb): the main
  results, with figures; it reads on GitHub without running.
- [`replication/RESULTS.md`](replication/RESULTS.md): every number beside the paper's, and
  **"What did not work"**.
- [`replication/README.md`](replication/README.md): scope, choices the paper leaves open, data,
  and how to run everything.
- [`replication/DIFFERENCES.md`](replication/DIFFERENCES.md) and
  [`replication/README_ISSUES.md`](replication/README_ISSUES.md): later rounds of runs, with a
  rebuilt universe and other readings of the paper.
- [`replication/AI_USAGE.md`](replication/AI_USAGE.md): how coding agents and LLMs were used.
