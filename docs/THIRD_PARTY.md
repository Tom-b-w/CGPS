# Third-party provenance

## DOTA

- Upstream: <https://github.com/skylineeeeen/DOTA>
- Recorded upstream commit: `ad2ea737325a8c8ef3a588ec86ab64a99d1f9107`
- Location: `third_party/dota/`
- License: MIT; retained as `third_party/dota/LICENSE`

The working source used by the paper differs from that commit in three files:

- `datasets/utils.py`: local dataset-path compatibility changes
- `dota_gda_em_aug.py`: streaming covariance compatibility changes
- `utils.py`: augmented-view and loader compatibility changes

The repository vendors the actual working files rather than replacing them
with a clean upstream checkout. Generated caches, logs, nested Git metadata,
and unused experiment outputs are excluded.

## ReTA prompt resources

- Upstream: <https://github.com/Evelyn1ywliang/ReTA>
- Recorded upstream commit: `b69a33c45e5b9a1fbfbac561eca4b5e307aca51d`
- Location: `assets/prompts/`
- License: MIT; retained as `third_party/RETA_LICENSE`

Only the ten CuPL JSON files required by the main table are included. Their
contents are unchanged; filenames use the descriptive
`cupl_prompts_<dataset>.json` pattern.

## Integrity note

The commit identifiers record provenance, not a claim that the vendored trees
are byte-identical to those commits. The disclosed compatibility patches are
part of the evaluated pipeline.
