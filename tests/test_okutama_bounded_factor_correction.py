import json
from pathlib import Path

import run_okutama_bounded_factor_correction as runner


def test_protocol_contract_and_model_matrix() -> None:
    protocol = json.loads(Path(runner.PROTOCOL).read_text(encoding="utf-8"))
    runner.validate_protocol(protocol)
    assert runner.model_spec(runner.ARMS[1]) == (True, False, True)
    assert runner.model_spec(runner.ARMS[2]) == (False, False, True)
    assert runner.model_spec(runner.ARMS[3]) == (False, True, True)
    assert runner.model_spec(runner.ARMS[4]) == (False, False, False)
    assert runner.make_model(protocol, runner.ARMS[2]).trainable_parameters == 100_613
    assert runner.make_model(protocol, runner.ARMS[3]).trainable_parameters == 100_355
