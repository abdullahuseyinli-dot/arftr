from __future__ import annotations

import json
import zipfile

import build_okutama_cptr_replay_bundle as builder
import numpy as np
import pandas as pd
import pytest


def paired_row(
    sample_id: str,
    scenario: str,
    track: str,
    *,
    scope: str,
    fold: str,
) -> dict[str, str]:
    return {
        "scope": scope,
        "development_role": "train" if scope == "grouped_crossfit_oof" else "validation",
        "sample_id": sample_id,
        "recording_id": scenario,
        "track_id": track,
        "fold": fold,
        "label_index": "1",
        "label": "standing",
        "transition_window": "False",
        "window_any_occluded": "False",
    }


def test_role_safe_index_uses_audit_offsets_and_auditor_string_sort(monkeypatch):
    monkeypatch.setattr(builder, "EXPECTED_ROWS", 3)
    monkeypatch.setattr(builder, "EXPECTED_OOF_ROWS", 2)
    monkeypatch.setattr(builder, "EXPECTED_VALIDATION_ROWS", 1)
    monkeypatch.setattr(builder, "EXPECTED_SCENARIOS", 1)
    monkeypatch.setattr(builder, "EXPECTED_FEATURE_ROWS", 5)
    paired = pd.DataFrame(
        [
            paired_row(
                "train__1.2.2__track-2__frame-000060",
                "2.2",
                "1.2.2::2",
                scope="grouped_crossfit_oof",
                fold="fold-0",
            ),
            paired_row(
                "train__1.2.2__track-10__frame-000030",
                "2.2",
                "1.2.2::10",
                scope="grouped_crossfit_oof",
                fold="fold-0",
            ),
            paired_row(
                "train__2.2.2__track-3__frame-000090",
                "2.2",
                "2.2.2::3",
                scope="fixed_development_validation",
                fold="fixed_validation",
            ),
        ]
    )
    audit = {
        "status": "OKUTAMA_DEVELOPMENT_ARCHIVE_AND_CENTRES_AUDITED",
        "selected_centres": 5,
        "recording_evidence": [
            {"recording_id": "1.1.1", "scenario_id": "1.1", "selected_centres": 2},
            {"recording_id": "1.2.2", "scenario_id": "2.2", "selected_centres": 2},
            {"recording_id": "2.2.2", "scenario_id": "2.2", "selected_centres": 1},
        ],
    }

    result = builder.derive_role_safe_index(paired, audit)

    assert result["feature_index"].tolist() == [2, 3, 4]
    # The historical pandas sort treats the string track "10" before "2".
    assert result["sample_id"].tolist()[:2] == [
        "train__1.2.2__track-10__frame-000030",
        "train__1.2.2__track-2__frame-000060",
    ]


def test_role_safe_index_rejects_an_incomplete_allowed_recording(monkeypatch):
    monkeypatch.setattr(builder, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(builder, "EXPECTED_OOF_ROWS", 1)
    monkeypatch.setattr(builder, "EXPECTED_VALIDATION_ROWS", 1)
    monkeypatch.setattr(builder, "EXPECTED_SCENARIOS", 1)
    monkeypatch.setattr(builder, "EXPECTED_FEATURE_ROWS", 3)
    paired = pd.DataFrame(
        [
            paired_row(
                "train__1.2.2__track-1__frame-000030",
                "2.2",
                "1.2.2::1",
                scope="grouped_crossfit_oof",
                fold="fold-0",
            ),
            paired_row(
                "train__2.2.2__track-1__frame-000030",
                "2.2",
                "2.2.2::1",
                scope="fixed_development_validation",
                fold="fixed_validation",
            ),
        ]
    )
    audit = {
        "status": "OKUTAMA_DEVELOPMENT_ARCHIVE_AND_CENTRES_AUDITED",
        "selected_centres": 3,
        "recording_evidence": [
            {"recording_id": "1.2.2", "scenario_id": "2.2", "selected_centres": 2},
            {"recording_id": "2.2.2", "scenario_id": "2.2", "selected_centres": 1},
        ],
    }

    with pytest.raises(RuntimeError, match="incomplete"):
        builder.derive_role_safe_index(paired, audit)


def test_forbidden_broad_metadata_is_rejected_before_read(tmp_path):
    forbidden = tmp_path / "development_metadata.csv"
    forbidden.write_text("secret", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Forbidden"):
        builder.capture(forbidden)


def test_materialization_rechecks_declared_array_against_source_lock(tmp_path):
    repository = tmp_path / "repository"
    directory = repository / ".runs" / "feature_store"
    directory.mkdir(parents=True)
    array_path = directory / "tight.npy"
    np.save(array_path, np.arange(3, dtype=np.float32))
    digest = builder.sha256_file(array_path)
    declaration_path = directory / "store.json"
    declaration = {"arrays": {"tight": {"path": "tight.npy", "sha256": digest}}}
    relative = array_path.relative_to(repository).as_posix()
    lock = {
        "materialization_inputs": {
            "feature_arrays": {
                "base_tight": {
                    "path": relative,
                    "sha256": digest,
                    "size_bytes": array_path.stat().st_size,
                    "verification": "opaque_byte_stream_no_array_decoding_or_indexing",
                }
            }
        }
    }

    observed = builder.verify_declared_array_receipts(
        declaration_path,
        declaration,
        store_name="base",
        expected_names={"tight"},
        materialization_lock=lock,
        repository_root=repository,
    )

    assert observed == lock["materialization_inputs"]["feature_arrays"]
    array_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="digest changed"):
        builder.verify_declared_array_receipts(
            declaration_path,
            declaration,
            store_name="base",
            expected_names={"tight"},
            materialization_lock=lock,
            repository_root=repository,
        )


def test_window_mask_loader_joins_by_id_and_rejects_extra_members(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "EXPECTED_ROWS", 2)
    path = tmp_path / "eligible_window_masks.npz"
    np.savez_compressed(
        path,
        sample_ids=np.asarray(["b", "a"]),
        window_occluded=np.asarray([[True] * 17, [False] * 17], dtype=bool),
    )
    values, digest = builder._load_window_masks(path, np.asarray(["a", "b"]))
    assert not values[0].any()
    assert values[1].all()
    assert digest == builder.sha256_file(path)

    bad = tmp_path / "bad.npz"
    np.savez_compressed(
        bad,
        sample_ids=np.asarray(["a", "b"]),
        window_occluded=np.zeros((2, 17), dtype=bool),
        unexpected=np.ones(1),
    )
    with pytest.raises(RuntimeError, match="must contain"):
        builder._load_window_masks(bad, np.asarray(["a", "b"]))


def test_mask_summary_is_json_serializable():
    value = {
        "status": "OKUTAMA_CPTR_ROLE_SAFE_EXACT_WINDOW_MASKS_COMPLETE",
        "rows": builder.EXPECTED_ROWS,
        "access_accounting": {"disallowed_annotation_members_read": 0},
    }
    assert json.loads(json.dumps(value))["access_accounting"] == {
        "disallowed_annotation_members_read": 0
    }


def test_declared_store_dtype_contract_accepts_fp16_parts_only(tmp_path):
    part_path = tmp_path / "part_tokens.npy"
    np.save(part_path, np.zeros((2, 3, 7, 4), dtype=np.float16))
    declaration_path = tmp_path / "store.json"
    declaration = {"arrays": {"part_tokens": {"path": part_path.name}}}

    result = builder._load_declared_array(
        declaration_path,
        declaration,
        "part_tokens",
        (2, 3, 7, 4),
        "float16",
    )

    assert isinstance(result, np.memmap)
    assert result.dtype == np.float16
    with pytest.raises(RuntimeError, match="shape/dtype changed"):
        builder._load_declared_array(
            declaration_path,
            declaration,
            "part_tokens",
            (2, 3, 7, 4),
            "float32",
        )


def test_retained_teacher_predictions_are_hash_bound_and_aligned(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "EXPECTED_OOF_ROWS", 4)
    monkeypatch.setattr(builder, "EXPECTED_FOLDS", ("fold-0", "fold-1"))
    monkeypatch.setattr(builder, "EXPECTED_SEEDS", (42, 43))
    index = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "scope": ["grouped_crossfit_oof"] * 4,
            "fold": ["fold-0", "fold-0", "fold-1", "fold-1"],
            "recording_id": ["g0", "g0", "g1", "g1"],
            "track_id": ["t0", "t1", "t2", "t3"],
            "label_index": [0, 1, 2, 0],
        }
    )
    entries = []
    for fold in range(2):
        ids = np.asarray(["a", "b"] if fold == 0 else ["c", "d"])
        selected = index.set_index("sample_id").loc[ids]
        for seed in (42, 43):
            values = np.full((2, 3), 0.1, dtype=np.float32)
            values[:, (fold + seed) % 3] = 0.8
            path = tmp_path / f"teacher-f{fold}-s{seed}.npz"
            np.savez_compressed(
                path,
                sample_ids=ids[::-1],
                recording_ids=selected.loc[ids[::-1], "recording_id"].to_numpy(dtype=str),
                track_ids=selected.loc[ids[::-1], "track_id"].to_numpy(dtype=str),
                labels=selected.loc[ids[::-1], "label_index"].to_numpy(),
                probabilities=values[::-1],
            )
            entries.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "files": {
                        "teacher_predictions": {
                            "path": path.name,
                            "bytes": path.stat().st_size,
                            "sha256": builder.sha256_file(path),
                        }
                    },
                }
            )

    ids, seeds, probabilities = builder._load_primary_retained_teacher_probabilities(
        {"checkpoints": entries}, index, repository_root=tmp_path
    )

    assert ids.tolist() == ["a", "b", "c", "d"]
    assert seeds.tolist() == [42, 43]
    assert probabilities.shape == (4, 2, 3)
    assert np.allclose(probabilities.sum(axis=2), 1.0)
    entries[0]["files"]["teacher_predictions"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="differs from inventory"):
        builder._load_primary_retained_teacher_probabilities(
            {"checkpoints": entries}, index, repository_root=tmp_path
        )


def test_distinct_window_contains_legacy_slots_and_declared_center():
    legacy, legacy_center = builder.sample_indices_with_centre(
        17, centre_index=8, samples=8, span_frames=8
    )
    positions = [
        int(np.flatnonzero(builder.DISTINCT_SHORT_INDICES == value)[0]) for value in legacy
    ]

    assert builder.DISTINCT_SHORT_INDICES.tolist() == [4, 5, 6, 7, 8, 10, 11, 12]
    assert builder.DISTINCT_SHORT_INDICES[builder.DISTINCT_SHORT_CENTRE] == 8
    assert builder.DISTINCT_SHORT_INDICES[positions].tolist() == legacy.tolist()
    assert legacy_center == builder.DISTINCT_SHORT_CENTRE == 4


def annotation_line(track: int, frame: int, *, occluded: bool = False) -> str:
    return f"{track} 0 0 30 60 {frame} 0 {int(occluded)} 0 Person Standing"


def test_exact_mask_extractor_reads_only_allowed_recording_members(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "EXPECTED_ROWS", 1)
    monkeypatch.setattr(builder, "EXPECTED_SCENARIOS", 1)
    sample_id = "train__1.1.1__track-5__frame-000030"
    index = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "provider_recording_id": "1.1.1",
                "label": "standing",
                "transition_window": "False",
                "window_any_occluded": "True",
            }
        ]
    )
    index_path = tmp_path / "eligible_index.csv"
    index.to_csv(index_path, index=False)
    inventory_path = tmp_path / "input_inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "status": builder.STATUS,
                "artifacts": {"eligible_index.csv": {"sha256": builder.sha256_file(index_path)}},
            }
        ),
        encoding="utf-8",
    )
    archive_path = tmp_path / "TrainSetFrames.zip"
    frames = [30 + value for value in builder.WINDOW_OFFSETS]
    multi = "\n".join(annotation_line(99, frame, occluded=frame == frames[0]) for frame in frames)
    tracking = "\n".join(annotation_line(5, frame, occluded=frame == frames[0]) for frame in frames)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Labels/MultiActionLabels/3840x2160/1.1.1.txt", multi)
        archive.writestr("Labels/SingleActionTrackingLabels/3840x2160/1.1.1.txt", tracking)
        archive.writestr(
            "Labels/MultiActionLabels/3840x2160/2.2.2.txt",
            b"this disallowed member must never be parsed",
        )
    monkeypatch.setattr(builder, "EXPECTED_ARCHIVE_BYTES", archive_path.stat().st_size)
    monkeypatch.setattr(builder, "EXPECTED_ARCHIVE_SHA256", builder.sha256_file(archive_path))
    output = tmp_path / "eligible_window_masks.npz"
    summary_path = tmp_path / "window_mask_summary.json"

    summary = builder.extract_role_safe_window_masks(
        archive_path=archive_path,
        inventory_path=inventory_path,
        eligible_index_path=index_path,
        output_path=output,
        summary_path=summary_path,
    )

    with np.load(output, allow_pickle=False) as arrays:
        assert arrays["sample_ids"].tolist() == [sample_id]
        assert arrays["window_occluded"].shape == (1, 17)
        assert arrays["window_occluded"].sum() == 1
    assert summary["annotation_members_read_count"] == 2
    assert summary["access_accounting"]["disallowed_annotation_members_read"] == 0
    assert summary["archive_wide_crc_scan_performed"] is False
