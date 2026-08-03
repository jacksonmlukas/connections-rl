"""Regression tests for the entropy/KL probe's adapter switching.

The sweep holds one base model in memory and swaps named PEFT adapters to
measure each checkpoint. Getting that toggle wrong is silently catastrophic:
if `_activate` were a no-op, every checkpoint would be measured under the same
policy and all KL values would collapse to zero while still looking plausible.

`PeftModel` exposes `disable_adapter()` only as a *context manager*; the
programmatic layer toggles live on the tuner (`model.base_model`). An earlier
version called `disable_adapters()`/`enable_adapters()`, which exist on
transformers' PEFT integration but not on `PeftModel`, and the sweep died with
an AttributeError. These tests pin both API shapes.
"""

from __future__ import annotations

import pytest

from connections_rl.eval.entropy_kl import _activate


class _Tuner:
    """Stands in for peft's LoraModel (`model.base_model`)."""

    def __init__(self, log: list[str]) -> None:
        self._log = log

    def disable_adapter_layers(self) -> None:
        self._log.append("disable_layers")

    def enable_adapter_layers(self) -> None:
        self._log.append("enable_layers")


class _PeftLike:
    """Stands in for peft's PeftModel: no disable_adapters/enable_adapters."""

    def __init__(self) -> None:
        self.log: list[str] = []
        self.base_model = _Tuner(self.log)

    def set_adapter(self, name: str) -> None:
        self.log.append(f"set:{name}")


class _TransformersLike:
    """Stands in for a PreTrainedModel using transformers' PEFT integration."""

    def __init__(self) -> None:
        self.log: list[str] = []

    def disable_adapters(self) -> None:
        self.log.append("disable")

    def enable_adapters(self) -> None:
        self.log.append("enable")

    def set_adapter(self, name: str) -> None:
        self.log.append(f"set:{name}")


def test_base_disables_adapter_layers_on_peft_model() -> None:
    m = _PeftLike()
    _activate(m, "base")
    assert m.log == ["disable_layers"]


def test_named_adapter_reenables_layers_before_selecting() -> None:
    """The ordering matters: a prior 'base' pass leaves adapter layers off.

    If `set_adapter` were called without re-enabling first, every checkpoint
    measured after a base pass would silently run as the base model.
    """
    m = _PeftLike()
    _activate(m, "base")
    m.log.clear()
    _activate(m, "ckpt-150")
    assert m.log == ["enable_layers", "set:ckpt-150"]


def test_falls_back_to_transformers_peft_integration_api() -> None:
    m = _TransformersLike()
    _activate(m, "base")
    assert m.log == ["disable"]
    m.log.clear()
    _activate(m, "sft")
    assert m.log == ["enable", "set:sft"]


@pytest.mark.parametrize("which", ["sft", "__sft__", "ckpt-50"])
def test_every_non_base_name_selects_that_adapter(which: str) -> None:
    m = _PeftLike()
    _activate(m, which)
    assert m.log[-1] == f"set:{which}"
