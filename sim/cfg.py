"""Small helpers for reading scenario YAML: strict keys, numbers, pairs and exact times.

Scenarios are written by hand, so the parser must fail loudly and clearly on typos rather than
fall back to a default. Two YAML traps are handled here once for every module:

* PyYAML reads ``1e-3`` (no decimal point) as the *string* "1e-3", while ``1.0e-3`` is a float.
  Every numeric field therefore goes through :func:`num`, which accepts both spellings and
  rejects booleans (``yes``/``no`` are booleans in YAML 1.1).
* Times and frame rates are exact rationals (``fractions.Fraction``), so that "is this
  exposure straddling a video frame boundary?" never depends on floating-point rounding.
  :func:`rational` converts a YAML number through its decimal text (``0.0123`` becomes exactly
  123/10000) and also accepts ratios such as ``"1/30"``; :func:`seconds` is it for times.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any


def check_keys(cfg: Mapping[str, Any], allowed: set[str] | frozenset[str], where: str) -> None:
    """Refuse keys outside `allowed`: they are almost always typos."""
    if not isinstance(cfg, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(cfg).__name__}")
    unknown = set(cfg) - set(allowed)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")


def require(cfg: Mapping[str, Any], keys: set[str], where: str) -> None:
    missing = set(keys) - set(cfg)
    if missing:
        raise ValueError(f"{where}: missing keys {sorted(missing)}")


def num(value: Any, where: str) -> float:
    """A float from a YAML number or numeric string ("1e-3"); booleans are refused."""
    if isinstance(value, bool):
        raise ValueError(f"{where}: expected a number, got a boolean")
    if isinstance(value, (int, float, Fraction)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(Fraction(value.strip()))
        except (ValueError, ZeroDivisionError):
            pass
    raise ValueError(f"{where}: expected a number, got {value!r}")


def integer(value: Any, where: str) -> int:
    """An exact integer: ints as they are (a seed of 2**53 + 1 stays itself), "1e3" or 4.0 too."""
    if isinstance(value, bool):
        raise ValueError(f"{where}: expected an integer, got a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            exact = Fraction(value.strip())
        except (ValueError, ZeroDivisionError):
            exact = None
        if exact is not None and exact.denominator == 1:
            return int(exact)
    raise ValueError(f"{where}: expected an integer, got {value!r}")


def rational(value: Any, where: str, what: str = "an exact number") -> Fraction:
    """An exact rational: 0.5 -> 1/2, 0.0123 -> 123/10000, "1/30" -> 1/30. `what` names it in errors."""
    if isinstance(value, bool):
        raise ValueError(f"{where}: expected {what}, got a boolean")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(repr(value))  # the decimal the author wrote, not its binary approximation
    if isinstance(value, str):
        try:
            return Fraction(value.strip())
        except (ValueError, ZeroDivisionError):
            pass
    raise ValueError(f"{where}: expected {what}, got {value!r}")


def seconds(value: Any, where: str) -> Fraction:
    """An exact time in seconds (see :func:`rational`)."""
    return rational(value, where, "a time in seconds")


def pair(value: Any, where: str, kind: type = float) -> tuple[Any, Any]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 2:
        raise ValueError(f"{where}: expected a pair [x, y], got {value!r}")
    first, second = value
    if kind is int:
        return integer(first, where), integer(second, where)
    return num(first, where), num(second, where)


def choice(value: Any, options: tuple[str, ...] | list[str], where: str) -> str:
    if value not in options:
        raise ValueError(f"{where}: must be one of {sorted(options)}, got {value!r}")
    return str(value)
