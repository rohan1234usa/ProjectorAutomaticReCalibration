/* Algorithm page: evidence from simulator frames, the cepstrum panels, and the interactive models. */
(function () {
  "use strict";
  const { D, $, el, restoreHash, svg, num, fillValues, overlayFigure, segmented } = window.Demo;
  const { line } = window.Charts;
  const pix = (img, alt) => el("div", { class: "frame" }, el("img", { src: img.src, width: img.w, height: img.h, alt, class: "pixelated zoomable", loading: "lazy" }));

  function edgeCharts(box, edges, unit, fmtY, keys) {
    for (const e of edges) {
      const card = el("div");
      const xs = e.xs;
      const all = keys.flatMap((k) => e[k]);
      const lo = Math.min(...all), hi = Math.max(...all), pad = 0.08 * (hi - lo || 1);
      const shift = e.crossing.moved != null && e.crossing.aligned != null ? e.crossing.moved - e.crossing.aligned : null;
      line(card, {
        title: e.name, height: 190, table: false,
        x: { min: xs[0], max: xs[xs.length - 1], name: "x", unit: " mm", fmt: (v) => num(v, 0), count: 3 },
        y: { min: lo - pad, max: hi + pad, fmt: fmtY, unit, count: 3 },
        vrefs: [{ x: e.x_mm, label: "calibrated" }],
        series: [{ name: "aligned", color: "var(--axis)", points: xs.map((x, i) => [x, e.aligned[i]]) },
                 { name: "B moved 8 px", color: e.name.startsWith("B") ? "var(--b)" : "var(--a)", points: xs.map((x, i) => [x, e.moved[i]]) }],
      });
      card.append(el("p", { class: "small muted", text: shift === null ? "edge not found" : `edge moved ${num(shift, 2)} mm` }));
      box.append(card);
    }
  }

  function boundary(A, C, S) {
    const b = A.boundary;
    edgeCharts($("#raster-edges"), b.raster, " e⁻", (v) => num(v, 0), ["aligned", "moved"]);
    edgeCharts($("#picture-edges"), b.picture, "", (v) => num(v, 2), ["aligned", "moved"]);
    const moves = (who) => b.raster.filter((e) => e.name.startsWith(who)).map((e) => e.crossing.moved - e.crossing.aligned);
    $("#raster-shift").textContent = moves("B").map((v) => `${num(v, 2)} mm`).join(" and ") + ` (true ${num(Math.hypot(...b.true_shift_mm), 2)} mm)`;
    $("#raster-still").textContent = moves("A").map((v) => `${num(v, 2)} mm`).join(" and ");
    window.Toys.boundaryToy($("#boundary-toy"), A.arrangements, C.config.values, S.timelines.shift.pitch_mm);
    const up = A.uplift;
    const fig = $("#uplift-fig");
    const show = (i) => {
      const img = i ? up.on.img : up.off.img;
      fig.replaceChildren(el("div", { class: "frame" }, el("img", { src: img.src, width: img.w, height: img.h, alt: "Black frame", class: "zoomable" }),
        el("span", { class: "tag", text: i ? "black uplift on" : "black uplift off" })));
    };
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
    $("#echo-frames").textContent = String(e.frames.length);
    const box = $("#cepstrum-panels");
    const W = e.window, size = 2 * W + 1;
    for (const r of e.results) {
      const frame = el("div", { class: "frame" }, el("img", { src: r.img.src, width: size, height: size, alt: `Cepstrum, B moved ${r.size_px} px`, class: "pixelated zoomable" }));
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

  function hotspot(A, C, S) {
    const h = A.hotspot;
    $("#hotspot-map").append(el("div", { class: "frame" }, el("img", { src: h.map.src, width: h.map.w, height: h.map.h, alt: "Brightness ratio, B moved 2 px over aligned", class: "zoomable" }),
      el("span", { class: "tag", text: `blue darker, red brighter, ±${num(100 * h.vmax, 0)}%` })));
    const p = h.profile, xs = p.x_mm;
    const [x0, x1] = h.overlap_x_mm;
    $("#hotspot-rms").textContent = `${num(100 * h.model_rms, 5)}% rms noiseless and ${num(100 * h.noisy_rms, 4)}% for one noisy `
      + `exposure, against a ${num(Math.abs(h.dip_pct), 2)}% dip`;
    const all = [...p.measured, ...p.noisy].map((v) => 100 * v);
    line($("#hotspot-profile"), {
      title: "Across the overlap, averaged along its height", height: 240,
      x: { min: xs[0], max: xs[xs.length - 1], name: "x", unit: " mm", fmt: (v) => num(v, 0) },
      y: { min: Math.min(...all) - 0.15, max: Math.max(...all) + 0.15, fmt: (v) => `${num(v, 1)}%` },
      vrefs: [{ x: x0, label: "overlap" }, { x: x1 }], refs: [{ y: 0 }],
      series: [{ name: "one noisy exposure", color: "var(--axis)", points: xs.map((x, i) => [x, 100 * p.noisy[i]]) },
               { name: "simulator, noiseless", color: "var(--b)", points: xs.map((x, i) => [x, 100 * p.measured[i]]) },
               { name: "model b(x − u) − b(x)", color: "var(--a)", points: xs.map((x, i) => [x, 100 * p.model[i]]), dash: true }],
    });
    window.Toys.hotspotToy($("#hotspot-toy"), { widthPx: (x1 - x0) / S.timelines.shift.pitch_mm, factor: h.factor, lampWarn: C.config.values.LAMP_WARN });
  }

  function border(A) {
    const b = A.border;
    const ys = b.y_mm;
    const keys = ["aligned", "4 px along", "8 px along"].filter((k) => k in b.profiles);
    const colors = ["var(--axis)", "var(--ord-3)", "var(--ord-6)"];
    line($("#border-profile"), {
      title: "Brightness across the picture's bottom edge, inside the overlap", height: 250,
      x: { min: ys[0], max: ys[ys.length - 1], name: "y", unit: " mm", label: "position down the screen (mm)", fmt: (v) => num(v, 0) },
      y: { min: -0.05, max: 1, fmt: (v) => num(v, 1), unit: " of white" },
      vrefs: [{ x: b.y_edge_mm, label: "calibrated edge" }],
      series: keys.map((k, i) => ({ name: k, color: colors[i], points: ys.map((y, j) => [y, b.profiles[k][j]]) })),
    });
    $("#border-crops").append(...keys.map((k) => el("figure", { class: "fig" }, pix(b.crops[k], k), el("figcaption", { text: k }))));
  }

  function reference(A) {
    const r = A.reference;
    $("#ref-source").append(el("div", { class: "frame" }, el("img", { src: r.source.src, width: r.source.w, height: r.source.h, alt: "The picture sent to the projectors", class: "zoomable", loading: "lazy" })));
    $("#ref-camera").append(el("div", { class: "frame" }, el("img", { src: r.camera.src, width: r.camera.w, height: r.camera.h, alt: "The camera frame", class: "zoomable", loading: "lazy" })));
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
    const A = D.algorithm, C = D.common, S = D.samples;
    window.Toys.decisionToy($("#decision-toy"), C.config.values);
    window.Toys.toleranceToy($("#tolerance-toy"), C.config.values, S ? S.timelines.shift.pitch_mm : 1.0417);
    config(C);
    if (!A || !S) {
      $("main").prepend(el("div", { class: "wrap" }, el("p", { class: "note", text: "Sample data was not built, so the figures are missing. Run python -m scripts.make_site." })));
      return;
    }
    rectified(A);
    boundary(A, C, S);
    echo(A);
    hotspot(A, C, S);
    border(A);
    reference(A);
    restoreHash();
  });
})();
