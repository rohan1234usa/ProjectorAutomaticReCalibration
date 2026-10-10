/* Overview page: headline numbers, the before/after slider, the build-order strip. */
(function () {
  "use strict";
  const { D, $, el, restoreHash, num, fillValues, compare } = window.Demo;

  document.addEventListener("DOMContentLoaded", () => {
    fillValues();
    const t = D.tests;
    if (t) {
      const bad = t.totals.failed + t.totals.error;
      $("#kpi-tests").textContent = `${t.totals.passed} / ${t.totals.total}`;
      $("#kpi-tests-sub").textContent = bad ? `${bad} failing in this build`
        : t.totals.slow_not_run ? `plus ${t.totals.slow_not_run} slow ones, not run in this build` : "slow tests included";
    }
    const phases = (D.common && D.common.phases) || [];
    $("#kpi-phases").textContent = `${phases.filter((p) => p.state === "done").length} of ${phases.length}`;

    const s = D.samples && D.samples.shift;
    const box = $("#hero-compare");
    if (s) {
      const eight = s.steps.find((x) => x.size_px === 8);
      compare(box, s.steps[0].crops[0], eight.crops[0], ["aligned", "B moved 8 px"], true);
      box.append(el("figcaption", {}, `A ${s.steps[0].crops[0].w}-pixel-wide patch of the overlap at the camera's own resolution, enlarged. `
        + `Scenario shift_sweep, frame ${s.frame}; true offset ${num(eight.offset_mm, 2)} mm.`));
    } else {
      box.append(el("p", { class: "note", text: "Sample images were not built. Run python -m scripts.make_site to render them." }));
    }

    const strip = $("#roadmap-strip");
    for (const p of phases) {
      strip.append(el("div", { class: `phase ${p.state}` },
        el("div", { class: "num", text: `Phase ${p.number}` }),
        el("div", { class: "name", text: p.title }),
        el("div", { style: { marginTop: "6px" } }, el("span", { class: `badge ${p.state}`, text: p.state === "done" ? "done" : p.state === "next" ? "next" : "later" }))));
    }
    restoreHash();
  });
})();
