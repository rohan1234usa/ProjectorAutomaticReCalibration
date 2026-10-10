/* Algorithm page: evidence from simulator frames, the cepstrum panels, and the interactive models. */
(function () {
  "use strict";
  const { D, $, el, restoreHash, svg, num, provenance, fillValues, framedImg, overlayFigure, segmented } = window.Demo;
  const { line } = window.Charts;
  const pix = (img, alt) => framedImg(img, alt, { pixelated: true });
  const moved = (e) => (e.crossing.moved != null && e.crossing.aligned != null ? e.crossing.moved - e.crossing.aligned : null);

  function edgeCharts(box, edges, unit, fmtY, sizePx) {
    for (const e of edges) {
      const card = el("div");
      const xs = e.xs;
      const all = [...e.aligned, ...e.moved];
      const lo = Math.min(...all), hi = Math.max(...all), pad = 0.08 * (hi - lo || 1);
      const shift = moved(e);
      line(card, {
        title: e.name, height: 190, table: false,
        x: { min: xs[0], max: xs[xs.length - 1], name: "x", unit: " mm", fmt: (v) => num(v, 0), count: 3 },
        y: { min: lo - pad, max: hi + pad, fmt: fmtY, unit, count: 3 },
        vrefs: [{ x: e.x_mm, label: "calibrated" }],
        series: [{ name: "aligned", color: "var(--axis)", points: xs.map((x, i) => [x, e.aligned[i]]) },
                 { name: `B moved ${sizePx} px`, color: e.name.startsWith("B") ? "var(--b)" : "var(--a)", points: xs.map((x, i) => [x, e.moved[i]]) }],
      });
      card.append(el("p", { class: "small muted", text: shift === null ? "edge not found" : `edge moved ${num(shift, 2)} mm` }));
      box.append(card);
    }
  }

  function boundary(A, C) {
    const b = A.boundary;
    edgeCharts($("#raster-edges"), b.raster, " e⁻", (v) => num(v, 0), b.size_px);
    edgeCharts($("#picture-edges"), b.picture, "", (v) => num(v, 2), b.size_px);
    const moves = (who) => {
      const found = b.raster.filter((e) => e.name.startsWith(who)).map(moved);
      return found.map((v) => (v === null ? "edge not found" : `${num(v, 2)} mm`)).join(" and ");
    };
    $("#raster-shift").textContent = `${moves("B")} (true ${num(Math.hypot(...b.true_shift_mm), 2)} mm)`;
    $("#raster-still").textContent = moves("A");
    window.Toys.boundaryToy($("#boundary-toy"), A.arrangements, C.config.values);
    const up = A.uplift;
    const fig = $("#uplift-fig");
    const show = (i) => fig.replaceChildren(framedImg(i ? up.on : up.off, "Black frame", { eager: true, tag: i ? "black uplift on" : "black uplift off" }));
    $("#uplift-controls").append(segmented(["Uplift off", "Uplift on"], 0, show, "Black-level uplift"));
    show(0);
  }

  function rectified(A) {
    const r = A.rectified;
    overlayFigure($("#rectified-fig"), r.img, r.layers, { alt: "The camera frame redrawn on a millimetre grid of the screen", layersOn: ["a", "b", "overlap", "markers"] });
  }

  function echo(A) {
    const e = A.echo;
    window.Toys.cepstrumToy($("#cepstrum-toy"), e.row, e.floor_px);
    const box = $("#cepstrum-panels");
    const W = e.window, size = 2 * W + 1;
    for (const r of e.results) {
      const frame = framedImg(r.img, `Cepstrum, B moved ${r.size_px} px`, { pixelated: true, eager: true });
      const o = svg("svg", { class: "overlay", viewBox: `-0.5 -0.5 ${size} ${size}`, "aria-hidden": "true" });
      o.append(svg("circle", { cx: W, cy: W, r: e.floor_px, fill: "none", stroke: "#fab219", "stroke-width": 0.25, "stroke-dasharray": "0.6 0.4" }));
      if (r.size_px) for (const s of [1, -1]) o.append(svg("circle", { cx: W + s * r.echo_px[0], cy: W + s * r.echo_px[1], r: 1.6, fill: "none", stroke: "#fff", "stroke-width": 0.22 }));
      if (r.verdict !== "none") o.append(svg("path", { d: `M${W + r.at[0] - 1.2},${W + r.at[1]}h2.4M${W + r.at[0]},${W + r.at[1] - 1.2}v2.4`, stroke: "#e66767", "stroke-width": 0.35 }));
      frame.append(o);
      const words = { resolved: "resolved", unresolved: "unresolved", none: "can't tell" };
      const why = { resolved: `echo at ${num(r.estimate_mm, 2)} mm`, unresolved: "an echo inside the floor", none: "nothing above the thresholds" };
      box.append(el("div", { class: "card", style: { padding: "12px" } },
        el("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "8px" } },
          el("strong", { text: r.size_px ? `${r.size_px} px` : "Aligned" }), el("span", { class: "small muted", text: `${num(r.offset_mm, 2)} mm` })),
        frame,
        el("p", { class: "small", style: { margin: "8px 0 2px" } }, el("span", { class: `badge ${r.verdict}`, text: words[r.verdict] }), " ", why[r.verdict]),
        el("p", { class: "small muted", style: { margin: 0 } }, `Peak ${num(r.peak, 1)}, needs ${num(r.threshold, 1)} (null ${num(r.null, 1)})`)));
    }
    box.after(el("p", { class: "small muted", text: `Each panel is the core-minus-control cepstrum within ±${W} camera px of the origin, on one shared scale. `
      + "The dashed yellow circle is the floor, the white rings are where the true echo lies, and the red cross is the strongest peak beyond the floor." }));
  }

  function hotspot(A, C) {
    const h = A.hotspot;
    $("#hotspot-map").append(framedImg(h.map, `Brightness ratio, B moved ${h.size_px} px over aligned`, { eager: true, tag: `blue darker, red brighter, ±${num(100 * h.vmax, 0)}%` }));
    const p = h.profile, xs = p.x_mm;
    const [x0, x1] = h.overlap_x_mm;
    $("#hotspot-rms").textContent = `${num(100 * h.model_rms, 5)}% rms noiseless and ${num(100 * h.noisy_rms, 4)}% for one noisy `
      + `exposure, against a ${num(h.change_pct, 2)}% ${h.change}`;
    const all = [...p.measured, ...p.noisy].map((v) => 100 * v);
    line($("#hotspot-profile"), {
      title: "Across the overlap, averaged along its height", height: 240,
      x: { min: xs[0], max: xs[xs.length - 1], name: "x", unit: " mm", fmt: (v) => num(v, 0) },
      y: { min: Math.min(...all) - 0.15, max: Math.max(...all) + 0.15, fmt: (v) => `${num(v, 1)}%`, tip: (v) => `${num(v, 3)}%` },
      vrefs: [{ x: x0, label: "overlap" }, { x: x1 }], refs: [{ y: 0 }],
      series: [{ name: "one noisy exposure", color: "var(--axis)", points: xs.map((x, i) => [x, 100 * p.noisy[i]]) },
               { name: "simulator, noiseless", color: "var(--b)", points: xs.map((x, i) => [x, 100 * p.measured[i]]) },
               { name: "model b(x − u) − b(x)", color: "var(--a)", points: xs.map((x, i) => [x, 100 * p.model[i]]), dash: true }],
    });
    window.Toys.hotspotToy($("#hotspot-toy"), { widthPx: (x1 - x0) / A.pitch_mm, factor: h.factor, lampWarn: C.config.values.LAMP_WARN });
  }

  function border(A) {
    const b = A.border;
    const ys = b.y_mm;
    const colors = ["var(--axis)", "var(--ord-3)", "var(--ord-6)", "var(--ord-7)"];
    line($("#border-profile"), {
      title: "Brightness across the picture's bottom edge, inside the overlap", height: 250,
      x: { min: ys[0], max: ys[ys.length - 1], name: "y", unit: " mm", label: "position down the screen (mm)", fmt: (v) => num(v, 0) },
      y: { min: -0.05, max: 1, fmt: (v) => num(v, 1), tip: (v) => num(v, 3), unit: " of white" },
      vrefs: [{ x: b.y_edge_mm, label: "calibrated edge" }],
      series: b.profiles.map((p, i) => ({ name: p.label, color: colors[i % colors.length], points: ys.map((y, j) => [y, p.values[j]]) })),
    });
    $("#border-crops").append(...b.profiles.map((p) => el("figure", { class: "fig" }, pix(p.crop, p.label), el("figcaption", { text: p.label }))));
  }

  function reference(A) {
    const r = A.reference;
    $("#ref-source").append(framedImg(r.source, "The picture sent to the projectors"));
    $("#ref-camera").append(framedImg(r.camera, "The camera frame"));
    $("#ref-ring").textContent = r.ring.map((x) => `${x.tag.replace(/_/g, " ")} sent at ${num(x.sent_s, 1)} s`).join("; ")
      + `. The exposure started at ${num(r.t_s, 2)} s.`;
  }

  function config(C) {
    const rows = [];
    for (const sec of C.config.sections) {
      rows.push(el("tr", {}, el("td", { colspan: 3, style: { background: "var(--surface-2)", fontWeight: 650 }, text: sec.name })));
      for (const it of sec.items) {
        rows.push(el("tr", {}, el("td", { class: "mono", text: it.key }), el("td", { class: "num", text: Array.isArray(it.value) ? it.value.join(" of ") : String(it.value) }),
          el("td", { class: "small", text: it.comment })));
      }
    }
    $("#config-table").append(el("div", { class: "table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", { text: "Name" }), el("th", { class: "num", text: "Value" }), el("th", { text: "What it does" }))), el("tbody", {}, rows))));
  }

  document.addEventListener("DOMContentLoaded", () => {
    fillValues();
    const A = D.algorithm, C = D.common;
    window.Toys.decisionToy($("#decision-toy"), C.config.values);
    config(C);
    if (!A) {
      $("main").prepend(el("div", { class: "wrap" }, el("p", { class: "note", text: "Sample data was not built, so the figures are missing. Run python -m scripts.make_site." })));
      return;
    }
    $("#algorithm-made").textContent = `Figures rendered at ${A.meta.quality} quality ${provenance(A.meta)}.`;
    window.Toys.toleranceToy($("#tolerance-toy"), C.config.values, A.pitch_mm);
    rectified(A);
    boundary(A, C);
    echo(A);
    hotspot(A, C);
    border(A);
    reference(A);
    restoreHash();
  });
})();
