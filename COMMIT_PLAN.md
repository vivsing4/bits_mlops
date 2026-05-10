# Commit Plan

Goal: create clean, reviewable commits grouped by concern.

## Recommended Commit 1

Message:
chore(docs): add final submission readiness checklist

Files:
- ASSIGNMENT.md

Summary:
- Adds a final-pass completion section with requirement coverage, validation notes, and evidence paths.

## Recommended Commit 2

Message:
docs(readme): add deliverables map and local deployment verification steps

Files:
- README.md
- deploy/k8s-manifests/README.md

Summary:
- Expands repository documentation for assignment deliverables and practical local verification of API and metrics endpoints.

## Recommended Commit 3

Message:
chore(repo): add reporting guidance folders and adjust ignore rules

Files:
- .gitignore
- reports/README.md
- screenshots/README.md

Summary:
- Preserves generated report artifacts as ignored while tracking report documentation.
- Adds explicit screenshot/report guidance for submission evidence.

## Hygiene Notes

- .DS_Store is modified. Keep it uncommitted unless you intentionally want to track this binary change.
- Untracked directories reports/ and screenshots/ are expected because new README files were added.

## Suggested Staging Order

1. Stage Commit 1 files and commit.
2. Stage Commit 2 files and commit.
3. Stage Commit 3 files and commit.
4. Verify .DS_Store is not staged before push.
