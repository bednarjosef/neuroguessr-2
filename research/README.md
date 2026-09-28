# Research record

How NeuroGuessr 2 got from 535 km to 34.1 km, kept as it was written during the work.

| | |
|---|---|
| [`RESEARCH_LOG.md`](RESEARCH_LOG.md) | Cross-session log: champion history, banked wins, dead ends, open ideas |
| [`findings.md`](findings.md) | Per-session notebook of every experiment and its verdict |
| [`results.tsv`](results.tsv) | Machine-readable ledger of all logged runs (commit, score, memory, keep/discard/crash) |
| [`progress.png`](progress.png), [`analysis.ipynb`](analysis.ipynb) | Classifier-phase ratchet progress (535 → 210 km on the 60 k subset) |
| [`FULLRUN.md`](FULLRUN.md) | Runbook for the full 1.2 M-image classifier run |
| [`plans/`](plans/) | Design docs for the retrieval roadmap and the C4/C5 encoder runs |
| [`logs/`](logs/) | Raw logs from the retrieval grids and the C5 run |
| [`ENGINE.md`](ENGINE.md), [`program.md`](program.md) | The keep/reset loop the agent followed and its task config (run via `/goal`) |
