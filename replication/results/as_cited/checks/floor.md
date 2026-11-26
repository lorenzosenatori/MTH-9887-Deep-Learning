## The lower bound on the explained-variance term

The saved model of round 1 (30 factors, seed 0) for a test year, replayed over its eight training years in
batches of 64 days (fixed cuts; the training used random cuts). A term is one security in one batch.

| Model for the test year | Batches in which the bound binds | Terms below -1 | Mean term, bounded | Mean term, as printed | Difference, points | Lowest term |
|---|---|---|---|---|---|---|
| 1998 | 10 of 32 | 12 of 15,961 | 0.2734 | 0.2705 | 0.30 | -13.1 |
| 2010 | 30 of 32 | 54 of 15,976 | 0.4395 | 0.4138 | 2.57 | -78.9 |
| 2021 | 25 of 32 | 43 of 16,008 | 0.4556 | 0.4070 | 4.85 | -163.2 |

The same models with one variance per security over the whole training window:

| Model for the test year | Securities ever in the universe | ... with a return on half of the days | Terms below -1 among them | Mean term |
|---|---|---|---|---|
| 1998 | 940 | 468 | 0 | 0.3005 |
| 2010 | 949 | 469 | 0 | 0.5273 |
| 2021 | 889 | 476 | 0 | 0.5361 |
