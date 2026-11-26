# AI usage

The assignment asks for a disclosure of how coding agents and LLMs were used.

**What was done with an AI assistant.** The code in this folder (the WRDS client and downloader,
the data panel, the 39 characteristics, the model, the benchmarks, the rolling-window training,
the report script and the tests) was written by Claude Code (Anthropic; model Claude Fable 5.2)
in one working session on 2026-11-21 and 2026-11-22, at the direction of a group member who supplied the paper,
the assignment and WRDS access. The same session ran the experiments.

**How it was checked.**
- Unit tests in `tests/` for the algebra of the model, the timing and the data helpers.
- An audit by four independent AI reviewers (information from the future; fidelity to the
  paper's equations; data construction; numerics), each followed by a second reviewer who had to
  reproduce a finding before it counted. The audit confirmed 33 findings, 5 of them bugs, none a
  look-ahead leak. All were fixed or documented before the reported runs. `RESULTS.md` describes
  two of the bugs under "What did not work" (the ignored API filters and the masking bug of the
  PCA benchmarks). Other findings fixed in the code include the firm's market equity (summed
  over its share classes), the timing of the delisting return, a split-safe SUV, average ranks
  for ties, batches that cover every date, and a check against stale caches. The record of the
  audit itself is not in the repository.
- A second AI reviewer recomputed every number of the write-up from the result files. It found
  two overstated sentences, now corrected, and runs that predated a later speed-up of the code;
  those were rerun with the committed code (`RESULTS.md`, "Which code produced the numbers").
- Two later reviews by AI reviewers compared the README, `RESULTS.md` and the site with the run
  folders. They found that the README listed Table 3 and both benchmarks with 100 factors as
  run (they were not), and that `RESULTS.md` called the benchmark "ahead" where only its Sharpe
  ratio is higher, and not by a statistically clear margin. Both were corrected, and the
  comparison with bootstrap standard errors (`results/comparison.md`) was added.
- Tuning choices were made on 1996 to 1997 only; the 1998 to 2021 results were not used to
  choose anything.

**Rounds 2 and 3 (2026-11-23).** Two further rounds of runs were also made by AI assistants, at
the direction of a group member; `README_ISSUES.md` describes them.
- Round 2 (`scripts/replicate_daily_sharpe.py`, `scripts/replicate_adjusted_loss.py` and their
  comparison scripts, `tests/test_adjusted_loss.py`) was written and run with OpenAI Codex. The
  same assistant drew up the list of differences from the paper that round 3 answers.
- Round 3 (`scripts/replicate_as_cited.py`, `scripts/as_cited_comparison.py`,
  `scripts/check_issues.py`, `src/attnfactors/pca_regression.py`, the option `"cited"` of
  `chars.build`) was written and run with Claude Code (model Claude Fable 5.2), which also wrote
  `README_ISSUES.md`. It first checked each listed difference against the code and found that
  one run of round 2 could not answer its question (30 optimizer steps a year).
- Two AI research agents read the sources that the paper cites: the public code of the
  predecessor paper (github.com/gregzanotti/dlsa-public) and the list of characteristics in the
  arXiv versions of Chen, Pelger and Zhu. The final Internet Appendix of that paper sits behind a
  paywall and was not read. Statements about these sources in `README_ISSUES.md` rest on the
  agents' quotations; nobody in the group has compared them with the originals yet.
- Checks: 21 unit tests; the default inputs were rebuilt after the code change and are identical
  to those of round 1 bit for bit; one AI reviewer recomputed the numbers of `README_ISSUES.md`
  from the result files and corrected its text, and another reviewed the new code for use of
  future information and found none (`README_ISSUES.md`, "How this file and round 3 were
  checked").

**Round 4 (2026-11-23).** The round on a rebuilt universe (`scripts/replicate_company_universe.py`,
`scripts/company_comparison.py`, `DIFFERENCES.md`) was also written and run with Claude Code
(model Claude Fable 5.2).
- The rule of the universe is the assistant's inference from the 108 firms that the paper's
  figures name; `DIFFERENCES.md` shows the evidence and what it leaves unexplained.
- The first versions of the builder had defects, and the runs were restarted three times. An AI
  reviewer found that a fallback for missing share counts valued depositary receipts without
  their receipt ratio in 1998 to 2002; the fallback was removed. The first results showed
  single-day portfolio moves of up to 19 percent: share classes that hardly trade had become
  members, and quotes on days without a trade were taken as prices. Both rules were changed
  before the reported runs.
- The reviewer then rebuilt the universe month by month from files dated up to each ranking
  month and got the committed universe exactly, and found no remaining quote artifact. It
  considers two delisting returns of January 2021 (-94 and -96 percent, forced delistings of
  receipts) not credible as realized returns; they are CRSP's records and were left in.
  `DIFFERENCES.md` gives their effect.
- Checks: 24 unit tests; the builder with the rule of rounds 1 to 3 gives back their universe.
  After the runs, the extreme daily returns of the new universe were compared with those of the
  old one. From 1998 to 2021 a member has a daily return beyond 80 percent in size on 13 days
  in the new universe and on 11 in the old one; 10 are the same (events of 2001, 2008 and
  2021), and the new ones include the two delisting returns above.
- A second AI reviewer recomputed the numbers of `DIFFERENCES.md` from the result files. Every
  table reproduced. Six sentences of the assistant's text were wrong and were corrected. The
  largest: the text said the paper does not describe its market row, and the note of the
  paper's Table 2 calls it the equally weighted market portfolio. Others: a count of firms (200
  for 100), a claim about the period before 2003 that fails for one run, and a claim about
  volatility that holds for 30 factors only.

**Open choices and the repository (2026-11-22 to 2026-11-26).** The open-choice experiments
(100 factors, no past returns, the correlation-matrix baseline), the notebook
`notebooks/results.ipynb`, `requirements.txt` and the root README were added with Claude Code
(model Claude Opus 5.5) at the direction of a group member, and checked with the unit tests and
the committed result files.

**What a human still has to do.** The group owns what it submits. Before presenting a number
from `RESULTS.md`, rerun the run that produced it or read its
`results/runs/<name>/metrics.json` (the full log, `runs/<name>/train.log`, exists only on the
machine that ran it), and read the code path that computes it. The characteristic definitions in `chars.py` are a
reconstruction; compare them with the Internet Appendix of Chen, Pelger and Zhu (2024) if the
group cites them as the paper's.

**The slides.** The slide frames (`slides/attention-factors-slides.pdf`) and the website they were
built from were first drafted by the AI assistant and checked by AI reviewers. The group then chose,
cut and reworded the frames; what was said in the talk was the group's own.
