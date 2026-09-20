# Versioned technical reports

These are unchanged **historical v1/v2/v3 reports**, not PDFs of the September
ARFTR continuation. The latest outcome is in the
[current research overview](../../docs/RESEARCH_OVERVIEW.md) and
[ARFTR evidence package](../../results/arftr_development/README.md).

| Report | Scope |
| --- | --- |
| [POLAR v1](polar_public_report_v1.0.0.pdf) | Source-audited image benchmark |
| [V-COCO v2](vcoco_v2_external_transfer_v2.0.0.pdf) | Person-level transfer and representation comparisons |
| [Temporal v3](vcoco_v3_motion_identifiability_v3.0.0.pdf) | Sealed Okutama confirmation and budgeted routing |
| [CPTR development](okutama_cptr_development_v3.0.0.pdf) | Architecture extension and negative grouped-OOF result |

`vcoco_v3_motion_identifiability_v3.0.0.pdf` presents
the V-COCO mechanism study and the locked Okutama-Action static, temporal,
distillation, and fixed-budget routing experiments.

`okutama_cptr_development_v3.0.0.pdf` is the companion architecture development
report. It records the component sequence, five-seed validation result,
recording-grouped cross-fit, faithfulness interventions, and retained failure modes.

Both PDFs were rendered directly from their Markdown sources. The commands below
document historical builds: use a separate checkout for each tagged version, not
the active research worktree, and do not overwrite archived PDFs while exploring:

```bash
python tools/build_study_papers.py docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md \
  -o output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf
python tools/build_study_papers.py docs/OKUTAMA_CPTR_DEVELOPMENT.md \
  -o output/pdf/okutama_cptr_development_v3.0.0.pdf
```

`vcoco_v2_external_transfer_v2.0.0.pdf` remains the person-level V-COCO report from
the v2 release. Rebuild it from its tagged source with:

```bash
git checkout polar-study-v2.0.0
python tools/build_study_papers.py docs/VCOCO_V2_EXTERNAL_TRANSFER.md \
  -o output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf
```

`polar_public_report_v1.0.0.pdf` remains the source-overlap-controlled POLAR report
from the v1 release. Rebuild it from its tagged source with:

```bash
git checkout polar-study-v1.0.0
python tools/build_study_papers.py docs/POLAR_PUBLIC_REPORT.md \
  -o output/pdf/polar_public_report_v1.0.0.pdf
```

The release-candidate inventory is `results/human_activity_study_v3.0.0_manifest.json`;
checksums for both reports and the manifest are recorded in
`release/HUMAN_ACTIVITY_STUDY_V3.0.0_SHA256SUMS.txt`. The v1 and v2 inventories remain
available with their tagged releases. The v3 checksum file uses release-asset
basenames, so it can be verified after downloading the two PDFs, the manifest, and the
checksum file into one directory:

```bash
sha256sum -c HUMAN_ACTIVITY_STUDY_V3.0.0_SHA256SUMS.txt
```
