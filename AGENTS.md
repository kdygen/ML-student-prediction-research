# AGENTS.md

This repository is an active machine learning research project.

The primary objective is to develop reliable early prediction models for student outcomes using the Open University Learning Analytics Dataset (OULAD).

---

## General Behavior

Prefer making progress over asking unnecessary clarification questions.

If a reasonable assumption can be made without risking data loss or changing research conclusions, proceed and document the assumption.

Ask questions only when:
- work could be overwritten,
- an irreversible change is required,
- credentials or external access are needed,
- multiple incompatible interpretations exist.

---

## Research Philosophy

This repository is a research notebook, not a production application.

Preserve research history.

Do not aggressively clean or simplify code.

Previous experiments are valuable even if they are no longer active.

---

## Protected Work

Do not:

- overwrite reviewed checkpoints
- rewrite completed experiments
- remove commented code
- delete exploratory feature ideas
- remove previous analyses
- remove comparison sections

If replacing something, create a new version instead.

---

## Notebook Rules

The notebook is the research history.

It is allowed to:

- import new modules
- call extracted functions
- add new experiment sections

It is NOT allowed to:

- delete historical experiments
- reorganize the notebook heavily
- remove commented feature ideas

---

## Refactoring

Refactor incrementally.

Preferred workflow:

Analyze

↓

Plan

↓

Small change

↓

Verification

↓

Summary

↓

Repeat

Never perform a large repository-wide refactor in one step.

---

## Experiments

When asked to run experiments:

Do not stop to ask unnecessary questions.

Reasonable defaults should be used automatically.

Every experiment should produce:

- objective
- hypothesis
- implementation
- metrics
- observations
- conclusion

Store experiment outputs inside:

results/
reports/

---

## Data Integrity

Never introduce information unavailable at the prediction time.

Always consider possible leakage.

If leakage is suspected, explain why.

---

## Coding Principles

Prefer:

simple

readable

reproducible

modular

Avoid unnecessary abstractions.

---

## Commits

Never commit automatically.

Summarize changes first.

Wait for confirmation.

---

## When Unsure

Preserve work.

Prefer adding over replacing.

Small reversible changes are preferred over large rewrites.