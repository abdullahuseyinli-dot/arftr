import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOTS = (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "experiments")
for root in IMPORT_ROOTS:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


# These unchanged historical integration tests verify real private artifacts.
# Keep them executable locally without requiring dataset redistribution for CI.
LOCAL_ARTIFACT_TESTS = {
    "test_crossing_event_runner.py::test_protocol_preserves_approved_promotion_and_no_grid": ".runs/research_20260919/post_pdi_aerial_review_v1/NEXT_TRIAL_PROTOCOL.json",
    "test_evidence_utility_split_contract.py::test_protocol_and_locked_task_population": ".runs/research_20260908/source_swap_v1/data/memory_data.npz",
    "test_generalized_arftr_queue_v2.py::test_queue_inventory_is_fixed_smallest_first": ".runs/research_20260913/generalized_arftr_ancestors_v2/execution_lock.json",
    "test_generalized_arftr_queue_v2.py::test_first_population_is_independently_audited": ".runs/research_20260913/generalized_arftr_ancestors_v2/populations/9d28b86187f07f35e17dadaca9a217103302b20b87212c373da7f5e0d4af4adc/independent_audit.json",
    "test_matr_artifacts.py::test_selected_fold_loader_preserves_nested_identity_and_seed_contract": ".runs/research_20260908/source_swap_v1/data/memory_data.npz",
    "test_okutama_frame_supervision_runner.py::test_v2_protocol_and_metadata_receipt_validate_before_lock": ".runs/research_20260912/cached_frame_supervision_v2/data",
}


def pytest_addoption(parser):
    parser.addoption(
        "--require-local-artifacts",
        action="store_true",
        help="Fail rather than skip if selected historical integration assets are absent",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        key = item.nodeid.replace("\\", "/").removeprefix("tests/")
        relative = LOCAL_ARTIFACT_TESTS.get(key)
        if relative is None:
            continue
        item.add_marker(pytest.mark.local_artifacts)
        if not (REPOSITORY_ROOT / relative).exists():
            if config.getoption("--require-local-artifacts"):
                raise pytest.UsageError(f"Required historical artifact missing: {relative}")
            item.add_marker(
                pytest.mark.skip(
                    reason=f"Historical integration requires local artifact: {relative}"
                )
            )
