/* Overview page: headline numbers, the before/after slider, the build-order strip. */
(function () {
  "use strict";
  const { D, $, el, restoreHash, num, fillValues, compare } = window.Demo;

  document.addEventListener("DOMContentLoaded", () => {
    fillValues();
    const o = D.overview || {}, st = D.common.status;
    const t = o.tests;
    if (t) {
      const bad = t.failed + t.error;
      $("#kpi-tests").textContent = `${t.passed} / ${t.total}`;
      $("#kpi-tests-sub").textContent = bad ? `${bad} failing in this build`
        : t.slow_not_run ? `plus ${t.slow_not_run} slow ones, not run in this build` : "slow tests included";
    }
    $("#kpi-phases").textContent = `${st.done.length} of ${st.total}`;
    $("#kpi-phases-sub").textContent = st.next ? `next: phase ${st.next}, ${st.next_title.toLowerCase()}` : "every phase is done";
    $("#roadmap-now").textContent = st.next ? `Phases ${st.done.join(", ")} are done; phase ${st.next} is next.` : "Every phase is done.";

    const h = o.hero;
    const box = $("#hero-compare");
    if (h) {
      const sizes = h.sizes_px.filter((v) => v > 0);
      $("#hero-sizes").textContent = `from ${sizes[0]} to ${sizes[sizes.length - 1]} pixels`;
      compare(box, h.aligned, h.moved, ["aligned", `B moved ${h.size_px} px`], true);
      box.append(el("figcaption", {}, `A ${h.aligned.w}-pixel-wide patch of the overlap at the camera's own resolution, enlarged. `
        + `Scenario shift_sweep, frame ${h.frame}; true offset ${num(h.offset_mm, 2)} mm.`));
    } else {
      box.append(el("p", { class: "note", text: "Sample images were not built. Run python -m scripts.make_site to render them." }));
    }

    const strip = $("#roadmap-strip");
    for (const p of D.common.phases) {
      strip.append(el("div", { class: `phase ${p.state}` },
        el("div", { class: "num", text: `Phase ${p.number}` }),
        el("div", { class: "name", text: p.title }),
        el("div", { style: { marginTop: "6px" } }, el("span", { class: `badge ${p.state}`, text: p.state }))));
    }
    restoreHash();
  });
})();
