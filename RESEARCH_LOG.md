# Research log — neuroguessr-2 (image geolocalization)

**Durable, cross-session memory.** This is the **distilled** knowledge base — the champion,
what's banked, what's a dead end, and what to try next — kept readable at a glance. The raw
`results.tsv` (every run's score) and `findings.md` (the live notebook) are **also committed
and pushed** now, as the full history behind this summary. The `/goal` command reads this file
FIRST at session start and updates + pushes it (with the raw ledger + notebook) during and at
the end of a session, so every future session — even a fresh clone — starts with the
accumulated knowledge and doesn't repeat dead ends.

Objective: **`median_km`** (median great-circle error on the val split), **lower is better**.

---

## Current champion

_(none yet — first session will seed this from the stock baseline)_

- **median_km:** —
- **train.py commit:** —
- **one-line:** —
- **full metric panel (last full-val eval):** mean_km — · acc@25km — · acc@2500km — · geoguessr —

## Banked wins (confirmed to help — keep these, don't re-litigate)

_(each entry: the change, the median_km delta, and why it likely helped)_

- —

## Dead ends & mistakes (tried, did NOT help or broke — do NOT repeat)

_(each entry: what was tried, what happened, and the takeaway so it isn't retried blind)_

- —

## Open ideas / next to try (ranked)

_(carry unfinished/promising directions forward across sessions)_

- —

## Environment / gotchas learned

_(anything about the box, dataset, VRAM ceilings, throughput, DINOv3 quirks, etc.)_

- —

---

## Session history

_(one dated block per session: dates, champion at start → end, headline results)_

<!-- template:
### 2026-07-21
- Champion at start: <median_km or "baseline"> → at end: <median_km>
- Experiments run: <N>, kept: <k>
- Headline: <what moved the metric>
- Notes / follow-ups for next session: <...>
-->
