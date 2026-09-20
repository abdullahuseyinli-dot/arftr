from __future__ import annotations

import run_okutama_generalized_arftr_queue_v2 as queue


def test_queue_inventory_is_fixed_smallest_first() -> None:
    inventory = queue.populations()
    assert len(inventory) == 14
    assert inventory == sorted(inventory, key=lambda item: (item[1], item[0]))
    assert inventory[0] == (
        "9d28b86187f07f35e17dadaca9a217103302b20b87212c373da7f5e0d4af4adc",
        2228,
    )


def test_first_population_is_independently_audited() -> None:
    assert queue.audit_passed(
        "9d28b86187f07f35e17dadaca9a217103302b20b87212c373da7f5e0d4af4adc"
    )
