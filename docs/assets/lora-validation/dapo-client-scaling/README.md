# DAPO client-scaling figures

Copied from [PR #42](https://github.com/modal-projects/lilo/pull/42), commit
`b27c78f`, without changing the measurements or figures. The tested runtime was
based on `c64087c11fc5d080f360397da53cc061dbdfb83c`, with the benchmark provider
and runner described in that PR. These results use the inference changes in
PR #40; they do not benchmark the documentation PR's runtime checkout.

- `sweep.png` shows total and per-client output TPS, mean update time, and GPU
  cost versus a Tinker token-price estimate.
- `summary.json` retains the summary, individual-client metrics, prices and
  original run IDs.
- `trainer-batch-tokens.png` shows per-client batch tokens at submission time
  against the configured microbatch budget.
- `trainer-activity.png` and `.pdf` show trainer call intervals. The line is the
  fraction of each centered 60-second window spent inside those calls.
- `trainer-activity.json` retains every plotted interval and client batch,
  source log hashes, operation totals and timing boundaries. Actual GPU
  utilization and per-microbatch token fill were not recorded.
- `elapsed-time.json` includes the four sweep supervisor records, the six
  successful measurement windows, source hashes and hashes of the copied files.
  Total elapsed time spans the first attempted launch through final cleanup;
  measured time is the sum of the six successful windows after warmup.

The source branch contains `scripts/analyze_dapo_client_sweep.py` and
`scripts/plot_dapo_trainer_activity.py`, which produced the figures. The latter
can render directly from the committed activity JSON. All 378 measured updates
were checked against the extracted forward/backward sequence counts, optimizer
counts and snapshot counts. No new GPU runs were performed for this guide.
