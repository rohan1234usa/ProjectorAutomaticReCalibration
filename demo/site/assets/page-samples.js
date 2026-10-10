/* Sample data page: frames with their true geometry, the shift series, galleries, timelines. */
(function () {
  "use strict";
  const { D, $, el, restoreHash, num, int, fillValues, overlayFigure, overlaySvg, segmented, jsonView } = window.Demo;
  const { line } = window.Charts;
  const human = (s) => s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
  const facts = (f) => `Frame ${f.frame}, t = ${num(f.t_s, 1)} s, showing ${f.tag.replace(/_/g, " ")}. True offset ${num(f.offset_mm, 3)} mm.`;
  const pixImg = (img, alt) => el("div", { class: "frame" }, el("img", { src: img.src, width: img.w, height: img.h, alt, class: "pixelated zoomable", loading: "lazy" }));

  function installation(S) {
    const fig = S.installation;
    overlayFigure($("#installation-fig"), fig.img, fig.layers, {
      alt: "The simulated camera's view of the screen, with projector boxes drawn on top", eager: true,
      extraToggle: { labels: ["Natural", "Shadows boosted"], srcs: [fig.img, fig.img_log] },
    });
    $("#installation-cap").textContent = `${facts(fig.facts)} Orange: box A. Blue dashed: box B. Green: the overlap. `
      + `Pink: markers (all ${fig.facts.markers_visible} visible).`;
  }

  function shift(S) {
    const sh = S.shift;
    const steps = [...sh.steps, { ...sh.along, along: true }];
    const labels = steps.map((s) => (s.along ? "8 px along" : `${s.size_px} px`));
    const tol = D.common.config.values.TOLERANCE_MM;
    const L = { ...S.installation.layers, markers: [], labels: [], content: null };
    function show(i) {
      const s = steps[i];
      $("#shift-crop0").replaceChildren(pixImg(s.crops[0], `Overlap patch, B moved ${labels[i]}`));
      $("#shift-crop1").replaceChildren(pixImg(s.crops[1], `Another overlap patch, B moved ${labels[i]}`));
      const frame = el("div", { class: "frame" }, el("img", { src: s.diff.src, width: s.diff.w, height: s.diff.h, alt: "What changed", class: "zoomable" }));
      frame.append(overlaySvg(L, { a: true, b: true, overlap: false }));
      $("#shift-diff").replaceChildren(frame);
      $("#shift-diff-cap").textContent = i === 0
        ? "What changed against the aligned twin: nothing at all, this is the twin."
        : `What changed against the aligned twin. ${s.along ? "B moved down, along the overlap's edges." : "B moved right, across the overlap."} `
          + "Where A shines alone: nothing. In the overlap: the doubled content. Where B shines alone: B's whole picture, moved.";
      $("#shift-readouts").replaceChildren(
        el("div", { class: "readout" }, "True offset ", el("b", { text: `${num(s.offset_mm, 3)} mm = ${num(s.offset_px, 2)} px` })),
        el("div", { class: "readout" }, `Tolerance ${num(tol, 2)} mm: `,
          el("span", { class: `verdict ${s.offset_mm > tol ? "yes" : "no"}`, text: s.offset_mm > tol ? "a person would notice (YES)" : "below it (NO)" })),
        el("div", { class: "readout" }, "Largest change where A shines alone: ", el("b", { text: `${num(s.a_only_max_e, 1)} e⁻` })));
    }
    $("#shift-controls").append(segmented(labels, 4, show, "Shift size"));
    show(4);
  }

  function arrangements(S) {
    const grid = $("#arrangements-grid");
    for (const a of S.arrangements) {
      const card = el("div", { class: "card" });
      card.append(el("h3", { text: human(a.preset) }));
      overlayFigure(card, a.img, a.layers, { alt: `${human(a.preset)} arrangement`, chips: false, layersOn: ["a", "b", "overlap"] });
      card.append(el("div", { class: "frame", style: { marginTop: "8px" } },
        el("img", { src: a.blend.src, width: a.blend.w, height: a.blend.h, alt: `Blend map, ${human(a.preset)}`, class: "zoomable", loading: "lazy" })));
      card.append(el("p", { class: "small muted", style: { marginTop: "8px" } },
        `Overlap ${num(a.blend.overlap_m2, 2)} m², ${a.blend.overlap_vertices} corners. ${a.geometry.pieces.length} edge pieces, `
        + `${a.geometry.pieces.filter((p) => p.kind === "inner").length} of them inner edges.`));
      grid.append(card);
    }
  }

  function content(S) {
    const grid = $("#content-grid");
    const words = { textured: "textured", flat: "flat", dark: "dark", skip: "skip", lit: "not dark" };
    for (const c of S.content) {
      const card = el("div", { class: "card", style: { padding: "12px" } });
      card.append(el("div", { class: "frame" }, el("img", { src: c.img.src, width: c.img.w, height: c.img.h, alt: c.label, class: "zoomable", loading: "lazy" }),
        c.mode === "log" ? el("span", { class: "tag", text: "shadows boosted" }) : null));
      card.append(el("div", { style: { display: "flex", justifyContent: "space-between", gap: "8px", margin: "10px 0 4px", alignItems: "baseline" } },
        el("strong", { text: c.label }), el("span", { class: `badge ${c.routing}`, text: words[c.routing] })));
      const stats = [`overlap ${num(c.overlap_mean, 4)} of white (${num(c.overlap_mean_net, 4)} without room light)`];
      if (c.motion !== undefined) stats.push(`changes ${num(c.motion, 4)} of white from the previous frame`);
      if (c.routing !== c.intended) stats.push(`chosen as ${c.intended} content`);
      card.append(el("p", { class: "small muted", text: stats.join("; ") }));
      grid.append(card);
    }
  }

  function dark(A, S) {
    const d = A.dark;
    const fig = el("div");
    const L = { ...S.installation.layers, markers: [], labels: [], content: null };
    const imgs = [d.single, d.mean];
    const show = (i) => {
      fig.replaceChildren();
      const frame = el("div", { class: "frame" }, el("img", { src: imgs[i].src, width: imgs[i].w, height: imgs[i].h, alt: "Black frame, stretched", class: "zoomable" }));
      frame.append(overlaySvg(L, { a: showBoxes, b: showBoxes, overlap: false }));
      fig.append(frame);
    };
    let showBoxes = false, current = 1;
    const boxes = el("button", { class: "chip", type: "button", "aria-pressed": "false" }, el("span", { class: "swatch", style: { background: "var(--a)" } }), "Draw the true boxes");
    boxes.addEventListener("click", () => { showBoxes = !showBoxes; boxes.setAttribute("aria-pressed", String(showBoxes)); show(current); });
    $("#dark-controls").append(segmented(["One frame", "Mean of 12 frames"], 1, (i) => { current = i; show(i); }, "Frames"), boxes);
    $("#dark-fig").append(fig);
    show(1);
    const xs = d.profile.x_mm;
    line($("#dark-profile"), {
      title: "Across the screen, averaged over its height", height: 250,
      x: { min: xs[0], max: xs[xs.length - 1], name: "x", unit: " mm", label: "position across the screen (mm)", fmt: (v) => num(v, 0) },
      y: { min: -4, max: Math.max(24, 2.4 * d.black_e), fmt: (v) => num(v, 0), unit: " e⁻" },
      refs: [{ y: d.black_e, label: "one black level" }, { y: 2 * d.black_e, label: "two black levels (overlap)" }],
      series: [{ name: "one frame", color: "var(--axis)", points: xs.map((x, i) => [x, d.profile.single[i]]) },
               { name: "mean of 12", color: "var(--b)", points: xs.map((x, i) => [x, d.profile.mean[i]]) }],
    });
  }

  function nuisances(S) {
    const list = $("#nuisance-list");
    for (const n of S.nuisances) {
      const card = el("div", { class: "card" });
      const views = [["Before", n.before.img], ["During", n.during.img], ["What changed", n.diff]];
      if (n.next) views.push(["Next frame: what changed", n.next.diff]);
      const box = el("div");
      const show = (i) => {
        const [, img] = views[i];
        box.replaceChildren(el("div", { class: "frame" }, el("img", { src: img.src, width: img.w, height: img.h, alt: `${n.label}: ${views[i][0]}`, class: "zoomable", loading: "lazy" }),
          el("span", { class: "tag", text: views[i][0] })));
      };
      const f = n.during.facts;
      const state = [`lamp B ${num(100 * f.lamp.b, 0)}%`, `room light ${num(f.ambient, 2)}`, f.camera_bump.some((v) => v) ? "camera knocked" : "camera still",
        f.people ? "someone in front" : "nobody in front", `${f.markers_visible} of 8 markers visible`];
      const text = el("div", {},
        el("h3", { text: n.label }), el("p", { text: n.why }),
        el("p", { class: "small" }, el("span", { class: "badge done", text: `truth: aligned, offset ${num(f.offset_mm, 3)} mm` })),
        el("p", { class: "small muted", text: `During (frame ${f.frame}, t = ${num(f.t_s, 1)} s): ${state.join(", ")}. What changed is shown at full colour from ${num(100 * n.diff_scale_rel, 1)}% of white.` }));
      if (n.crops) {
        text.append(el("div", { class: "cols", style: { gap: "8px" } }, pixImg(n.crops[0], "Before, close up"), pixImg(n.crops[1], "During, close up")),
          el("p", { class: "small muted", text: "Close up, before and during." }));
      }
      card.append(el("div", { class: "cols wide-left" }, el("div", {}, el("div", { class: "controls" }, segmented(views.map((v) => v[0]), 1, show, "View")), box), text));
      show(1);
      list.append(card);
    }
  }

  function gain(S) {
    for (const [key, id] of [["matte", "#gain-matte"], ["peak", "#gain-peak"]]) {
      const g = S.gain[key];
      overlayFigure($(id), g.img, g.layers, { alt: `Flat grey on a ${key === "matte" ? "matte" : "gain"} screen`, chips: false, layersOn: ["a", "b", "overlap"] });
    }
    const m = S.gain.matte.profile, p = S.gain.peak.profile;
    line($("#gain-profile"), {
      title: `Brightness along the row through the hotspots (y = ${num(m.y_mm, 0)} mm), room light removed`, height: 240,
      x: { min: m.x_mm[0], max: m.x_mm[m.x_mm.length - 1], name: "x", unit: " mm", label: "position across the screen (mm)", fmt: (v) => num(v, 0) },
      y: { min: 0, max: Math.max(...p.rel) * 1.1, fmt: (v) => num(v, 2), unit: " of white" },
      series: [{ name: "matte screen", color: "var(--axis)", points: m.x_mm.map((x, i) => [x, m.rel[i]]) },
               { name: `gain screen, peak ${S.gain.peak.peak}`, color: "var(--b)", points: p.x_mm.map((x, i) => [x, p.rel[i]]) }],
    });
    const z = S.zoomed;
    for (const f of z.frames) {
      const card = el("figure", { class: "fig" });
      overlayFigure(card, f.img, f.layers, { alt: `Zoomed camera, B moved ${f.size_px} px`, chips: false, layersOn: ["a", "b", "overlap", "markers"] });
      card.append(el("figcaption", { text: f.size_px ? `B moved ${f.size_px} px (${num(f.facts.offset_mm, 2)} mm)` : "Aligned" }));
      $("#zoomed-frames").append(card);
    }
    $("#zoomed-crops").append(...z.crops.map((c, i) => el("figure", { class: "fig" }, pixImg(c, i ? "B moved 8 px" : "Aligned"),
      el("figcaption", { text: i ? "B moved 8 px" : "Aligned" }))));
  }

  function truth(S, C) {
    const tl = S.timelines, tol = C.config.values.TOLERANCE_MM;
    const sh = tl.shift;
    const minutes = (v) => num(v / 60, 0);
    line($("#truth-shift"), {
      title: "shift_sweep: B steps across by 0.25 to 8 px at 615 s", height: 280, endLabels: true,
      x: { min: 0, max: sh.duration_s, name: "t", unit: " s", label: "time (minutes)", fmt: minutes, ticks: [0, 300, 600, 900, 1200] },
      y: { min: 0, max: 9, fmt: (v) => num(v, 0), unit: " mm" },
      bands: [{ x0: 0, x1: sh.trusted_window_s, label: "trusted window" }],
      refs: [{ y: tol, label: `TOLERANCE_MM ${num(tol, 2)}` }],
      series: sh.series.map((s, i) => ({ name: s.name, color: `var(--ord-${i + 1})`, points: s.points, step: true, endLabel: s.size_px >= 1 ? s.name : null })),
    });
    const dr = tl.drift;
    line($("#truth-drift"), {
      title: "slow_drift: 1 px per hour, after the trusted window", height: 250,
      x: { min: 0, max: dr.duration_s, name: "t", unit: " s", label: "time (hours)", fmt: (v) => num(v / 3600, 1), ticks: [0, 1800, 3600, 5400, 7200] },
      y: { min: 0, max: 2.4, fmt: (v) => num(v, 1), unit: " mm" },
      refs: [{ y: tol, label: `tolerance ${num(tol, 2)} mm` }],
      series: [{ name: "true offset", color: "var(--b)", points: dr.points }],
      annotations: dr.crossing_s ? [{ x: dr.crossing_s, y: tol, label: `crosses at ${num(dr.crossing_s / 3600, 2)} h`, dx: -10, dy: -10, anchor: "end", color: "var(--b)" }] : [],
    });
    $("#truth-rotation").append(el("div", { class: "table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, ["pivot", "offset", "angle"].map((h, i) => el("th", { class: i ? "num" : null, text: h })))),
      el("tbody", {}, tl.rotation.map((r) => el("tr", {}, el("td", { text: r.pivot.replace("_", " ") }),
        el("td", { class: "num", text: `${num(r.offset_px, 2)} px · ${num(r.offset_mm, 2)} mm` }), el("td", { class: "num", text: `${num(r.deg, 3)}°` })))))));
    const ln = tl.lanes;
    window.Charts.lanes($("#truth-lanes"), {
      x: { min: ln.t0, max: ln.t1, name: "t", unit: " s", label: "time (s)", fmt: (v) => num(v, 0) },
      lanes: [
        { name: "true offset (mm)", color: "var(--b)", points: ln.lanes.offset_mm, min: 0, max: 1, ticks: [0, 1], fmt: (v) => num(v, 0) },
        { name: "lamp B (gain)", color: "var(--a)", points: ln.lanes.lamp_b, min: 0.8, max: 1.02, ticks: [0.85, 1], fmt: (v) => num(v, 2) },
        { name: "room light (× white)", color: "var(--ov)", points: ln.lanes.room_light, min: 0, max: 0.06, ticks: [0.02, 0.05], fmt: (v) => num(v, 2) },
        { name: "camera knocked", color: "var(--series-4)", points: ln.lanes.camera_bump, min: 0, max: 1.1, ticks: [0, 1], fmt: (v) => (v ? "yes" : "no") },
        { name: "someone in front", color: "var(--marker)", points: ln.lanes.person, min: 0, max: 1.1, ticks: [0, 1], fmt: (v) => (v ? "yes" : "no") },
      ],
    });
  }

  function inputs(S) {
    const inp = S.inputs;
    $("#setup-json").append(jsonView(inp.setup));
    $("#line-json").append(jsonView(inp.line, ["truth", "perturbation", "nuisances", "content", "frame_sha256"]));
    $("#line-cap").textContent = `Scenario shift_sweep, variant ${inp.variant}, frame ${inp.frame}. In a dataset this is one line of metadata.jsonl.`;
  }

  function catalogue(C) {
    const table = el("table", {}, el("thead", {}, el("tr", {}, ["Scenario", "Question it answers", "Variants", "Frames each", "Length"].map((h, i) => el("th", { class: i > 1 ? "num" : null, text: h })))));
    const body = el("tbody");
    for (const c of C.catalogue) {
      const q = el("td");
      q.innerHTML = c.question; // HTML made by demo/markdown.py from CLAUDE.md, with its text escaped
      body.append(el("tr", {}, el("td", { class: "mono", text: c.name }), q, el("td", { class: "num", text: int(c.variants) }),
        el("td", { class: "num", text: int(c.frames_per_variant) }), el("td", { class: "num", text: `${num(c.duration_s / 60, 0)} min` })));
    }
    table.append(body);
    $("#catalogue-table").append(el("div", { class: "table-wrap" }, table));
  }

  document.addEventListener("DOMContentLoaded", () => {
    fillValues();
    const S = D.samples, A = D.algorithm, C = D.common;
    if (!S) {
      $("main").prepend(el("div", { class: "wrap" }, el("p", { class: "note", text: "Sample data was not built. Run python -m scripts.make_site to render it." })));
      return;
    }
    installation(S);
    shift(S);
    arrangements(S);
    content(S);
    if (A) dark(A, S);
    nuisances(S);
    gain(S);
    truth(S, C);
    inputs(S);
    catalogue(C);
    restoreHash();
  });
})();
