# Project identity and evidence preservation

The research is presented as two linked projects from 20 September 2026:

| Project | Repository | First independent project version | Primary result |
| --- | --- | --- | --- |
| ARFTR | [arftr](https://github.com/abdullahuseyinli-dot/arftr) | 1.0.0 | 85.383648% macro-F1; adaptive Okutama development |
| POLAR Posture Recognition | [polar-posture-recognition](https://github.com/abdullahuseyinli-dot/polar-posture-recognition) | 1.0.0 | 93.988333% macro-F1; four-class locked POLAR test |

These are different tasks, not a score ranking. ARFTR owns the temporal architecture
and continued research lineage. The companion owns the still-image benchmark,
its representation/ensemble comparison, and person-centric transfer follow-ups.

## Versioning

Independent project 1.0.0 starts a new naming boundary; it does not renumber the
experiments or assert that their models have changed. The original study tags
`polar-study-v1.0.0` and `polar-study-v2.0.0`, v3 reports and the legacy
`3.1.0.dev0` working revision remain historical identifiers. No old tag is moved.
The `v1.0.0` tag in each separately named repository identifies its new project
package. Citation and deposit metadata use the corresponding project version.
No Zenodo deposition or DOI is claimed by this separation.

## Why historical files remain here

ARFTR's upstream models depend on earlier `hac` modules and hash-locked paths.
Moving those files or deleting the old results would break provenance and replay.
They remain as a research archive, while the public landing pages and current
guides have separate scopes. The companion is a curated extraction of the
earlier released code plus the relevant later representation screen, not a copy
of the entire continuation repository.

The original `human-activity-classification` repository is renamed `arftr`, retaining
its commits and historical tags. Existing historical links should redirect on GitHub.
The workstation directory remains at its existing path to preserve local execution
locks; new users clone `arftr`. Both projects retain `hac` imports and must be
installed in separate virtual environments.

No dataset, checkpoint, feature cache, model implementation or numerical result is
rewritten by the separation. Aggregate evidence checks, unit tests and a public-only
checkout validate the presentation without claiming checkpoint reproduction.
