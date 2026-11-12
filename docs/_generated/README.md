# Generated documents

Every file in this directory is script output. Do not edit one by hand: the
script that writes it also checks it, and a hand edit fails the check rather than
surviving as a second, quietly diverging copy of a number.

| File | Producer | Refresh | Check |
| --- | --- | --- | --- |
| [`gate_arms.md`](gate_arms.md) | [`scripts/generate_gate_arm_report.py`](../../scripts/generate_gate_arm_report.py) from the release receipts in `output/reports/` | `--from-receipts output/reports` | `--check` |
| [`gate_arms.json`](gate_arms.json) | the same script; the record `gate_arms.md` renders | `--from-receipts output/reports` | `--check` |

## What `gate_arms.md` is allowed to say

It reports what a run measured on the host named in the record: the wall time of
each step in each gate arm, and how many tests that arm's pytest step reported as
passed. It is not a claim about a different machine, a different load, or a
different commit, and it is not a result: the recorded numbers in
[`../../data/claims.json`](../../data/claims.json) are the results.

The `--check` mode enforces two things that a hand-written page cannot: the
Markdown is exactly what the committed record renders, and every arm and step it
names still exists in `.venv/bin/python -I -S scripts/verify_release.py --list`.
An arm that is renamed or split therefore fails the documentation gate instead of
leaving a page that describes a command the repository no longer has.

## Refreshing after a gate change

```bash
# One receipt per arm, named for the profile it ran.
.venv/bin/python -I -S scripts/verify_release.py lint --receipt output/reports/lint.json
.venv/bin/python -I -S scripts/verify_release.py types --receipt output/reports/types.json
.venv/bin/python -I -S scripts/verify_release.py docs --receipt output/reports/docs.json
.venv/bin/python -I -S scripts/verify_release.py quick --receipt output/reports/quick.json
.venv/bin/python -I -S scripts/verify_release.py measurement --receipt output/reports/measurement.json
.venv/bin/python -I -S scripts/verify_release.py contract --receipt output/reports/contract.json
.venv/bin/python -I -S scripts/verify_release.py gnn --receipt output/reports/gnn.json

.venv/bin/python scripts/generate_gate_arm_report.py --from-receipts output/reports
.venv/bin/python scripts/generate_gate_arm_report.py --check
```

The `full` arm is recorded only when a prepared release workstation ran it; its
row says so when it did not, because a publication-acceptance number from a host
that could not have accepted the publication would be worse than a blank.
