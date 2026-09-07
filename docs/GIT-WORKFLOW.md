# Git Workflow

The repository uses `main` as the default integration branch. Feature work is
implemented on a dedicated branch and merged only after local verification.

```text
main
  │
  └── feature/<scope>
            │
            ├── implementation commits
            ├── tests
            ├── documentation
            │
            ↓
          push
            │
            ↓
       Pull Request
            │
            ↓
     review / local tests
            │
            ↓
        merge → main
```

## Current S2.5 branch

```text
s2.5-training-infrastructure
```

## Before starting work

```bash
git checkout main
git pull origin main
git checkout s2.5-training-infrastructure
```

## Before opening the PR

Keep the feature branch current with `main`:

```bash
git fetch origin
git checkout main
git pull origin main
git checkout s2.5-training-infrastructure
git merge main
pytest -v
git push origin s2.5-training-infrastructure
```

Resolve conflicts on the feature branch, rerun the full test suite, and only
then open/update the Pull Request.

## Merge policy

- `main` remains the stable integration branch.
- Do not develop S2.5 directly on `main`.
- Do not force-push shared `main`.
- A feature branch must pass the relevant tests before merge.
- For S2.5, the final gate is the full `pytest -v` suite plus the training
  smoke/infrastructure tests.
