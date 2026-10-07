"""The misalignment detector: the real algorithm.

It sees only what it would see on real hardware -- camera frames and the calibration
software's blending setup. It must never import anything from ``sim/`` (enforced by
``tests/test_imports.py``).
"""
