"""A static demo site: sample data from the simulator, the planned algorithm, the tests so far.

``python -m scripts.make_site`` builds it into ``out/site`` and can serve it on localhost. The
build renders sample frames from the repo's own scenarios, computes a few illustrations from
them, runs the test suite, and reads the project's documents (CLAUDE.md, docs/findings.md,
detector.yaml). The pages are plain HTML with the data in ``data/*.js`` files.

What the demo may and may not claim. The detector is not built yet (CLAUDE.md section 8), so
nothing here is detector output. Where a figure computes something the detector will compute
-- a cepstrum, a brightness ratio, an edge profile -- it does so with the simulator's ground
truth at hand (the aligned twin of a sweep, the true camera position) and says so on the page:
these are teaching illustrations of why each method should work.

Like the harness, ``demo/`` imports the simulator (``sim/``) and the harness helpers in
``scripts/``; it never imports ``detector/`` and reads ``detector.yaml`` as text instead.
``tests/test_imports.py`` holds it to that.
"""
