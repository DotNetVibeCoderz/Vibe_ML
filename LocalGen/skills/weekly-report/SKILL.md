---
name: weekly-report
description: Produces a weekly engineering report from a metrics file, using the house template.
version: 1.0.0
author: Gravicode Studios
tags: [reporting, metrics, example]
---

# Weekly report

An example skill. It shows the three things a skill can carry: instructions, a template asset, and
a script that produces the data the template needs.

## When to use this

The user asks for a weekly report, a sprint summary, or "how did we do this week".

## Steps

1. Run `scripts/summarise.py` with the path to a metrics CSV. It prints totals and week-over-week
   change as JSON.
2. Read `assets/report-template.md`.
3. Fill in each section from the script's output. Where the data does not cover a section, write
   "no data this week" — do not estimate a figure.
4. Save the result to the workspace as `weekly-report-<week>.md` and tell the user the path.

## Notes

- Figures come from the script, never from your own arithmetic.
- Keep the template's headings and their order; teams read these side by side across weeks.
