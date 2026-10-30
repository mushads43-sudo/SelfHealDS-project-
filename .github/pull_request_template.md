## What changed
<!-- One or two sentences. Link the issue if any. -->

## Type
- [ ] feat (new capability)   - [ ] fix   - [ ] research (rules / evaluation)   - [ ] mlops / infra   - [ ] docs

## Research-integrity checklist
- [ ] No change to detection/repair behaviour, OR the change is intentional and described above
- [ ] `pytest` passes locally
- [ ] `python mlops/validate_model.py` passes (precision, F1, unsafe-repair rate within thresholds)
- [ ] If rules changed: `RULESET_VERSION` in `backend/research.py` bumped and CHANGELOG.md updated
