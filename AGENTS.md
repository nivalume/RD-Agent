# AGENTS.md

Working notes for coding agents in this repository.

## Branching

`main` mirrors the upstream repository and exists only as the synchronisation point.
Never commit, merge or push work to `main`, and never open a pull request against it.

`dev` is the integration branch, cut from `main`. All work lands there.

- Branch off `dev` for every change: `feature/<name>`, `fix/<name>`, `chore/<name>`.
- Merge finished branches back into `dev` (pull request against `dev`, or a local merge commit).
- To take upstream changes, update `main` from upstream and merge `main` into `dev`;
  never merge `dev` or a feature branch into `main`.

```bash
git checkout dev && git pull
git checkout -b feature/<name>
# ... work ...
git checkout dev && git merge --no-ff feature/<name>
git push origin dev
```

## Tests

```bash
pytest test/scenarios/crypto -q
```
