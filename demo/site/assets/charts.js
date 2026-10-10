/* Small SVG charts: lines (and steps), horizontal bars, stacked lanes. Thin marks, hairline grid,
   a crosshair tooltip that lists every series, a legend for two or more series, and a data table. */
(function () {
  "use strict";
  const { el, svg, num } = window.Demo;

  function niceTicks(lo, hi, count = 5) {
    if (!(hi > lo)) return [lo];
    const raw = (hi - lo) / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => (hi - lo) / s <= count + 0.5) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const defaultFmt = (v) => {
    if (Math.abs(v) < 1e-9) return "0";
    if (Number.isInteger(v)) return num(v, 0);
    return num(v, Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2);
  };

  /* Value of a series at x: steps hold the last point at or before x; lines interpolate. */
  function valueAt(s, x) {
    const p = s.points;
    if (!p.length || x < p[0][0] - 1e-9 || x > p[p.length - 1][0] + 1e-9) return null;
    let lo = 0, hi = p.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (p[mid][0] <= x) lo = mid; else hi = mid; }
    if (s.step) return p[x >= p[hi][0] ? hi : lo][1];
    const [x0, y0] = p[lo], [x1, y1] = p[hi];
    if (y0 === null || y1 === null) return null;
    return x1 === x0 ? y0 : y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
  }

  function legend(series) {
    return el("div", { class: "legend" }, series.filter((s) => !s.noLegend).map((s) =>
      el("span", {}, el("span", { class: `key${s.dash ? " dash" : ""}`, style: { background: s.color, borderColor: s.color } }), s.name)));
  }

  function tableToggle(container, columns, rows) {
    const btn = el("button", { class: "datatoggle", type: "button", text: "Show the data" });
    const box = el("div", { class: "chart-table table-wrap", hidden: true });
    btn.addEventListener("click", () => {
      if (!box.childNodes.length) {
        const shown = rows.slice(0, 500);
        box.append(el("table", {}, el("thead", {}, el("tr", {}, columns.map((c) => el("th", { class: "num", text: c })))),
          el("tbody", {}, shown.map((r) => el("tr", {}, r.map((v) => el("td", { class: "num", text: v })))))));
        if (rows.length > shown.length) box.append(el("p", { class: "small muted", text: `First ${shown.length} of ${rows.length} rows.` }));
      }
      box.hidden = !box.hidden;
      btn.textContent = box.hidden ? "Show the data" : "Hide the data";
    });
    container.append(btn, box);
  }

  function line(container, spec) {
    const root = el("div", { class: "chart" });
    if (spec.title) root.append(el("div", { class: "toy-title", text: spec.title }));
    if (spec.series.filter((s) => !s.noLegend).length >= 2 && spec.legend !== false) root.append(legend(spec.series));
    const holder = el("div", { style: { position: "relative" } });
    root.append(holder);
    container.append(root);
    const fx = spec.x.fmt || defaultFmt, fy = spec.y.fmt || defaultFmt;
    let lastW = 0;

    function draw() {
      const W = Math.max(300, Math.round(holder.clientWidth || container.clientWidth || 640));
      if (Math.abs(W - lastW) < 8) return;
      lastW = W;
      const H = spec.height || 240;
      const m = { l: spec.marginLeft || 54, r: spec.endLabels ? 76 : 18, t: 12, b: spec.x.label ? 42 : 26 };
      const xs = (v) => m.l + ((v - spec.x.min) / (spec.x.max - spec.x.min)) * (W - m.l - m.r);
      const ys = (v) => H - m.b - ((v - spec.y.min) / (spec.y.max - spec.y.min)) * (H - m.t - m.b);
      const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.aria || spec.title || "chart" });
      for (const b of spec.bands || []) {
        s.append(svg("rect", { class: "band", x: xs(b.x0), y: m.t, width: Math.max(0, xs(b.x1) - xs(b.x0)), height: H - m.t - m.b }));
        if (b.label) s.append(svg("text", { x: xs(b.x0) + 6, y: m.t + 14, class: "label" }, b.label));
      }
      for (const t of spec.y.ticks || niceTicks(spec.y.min, spec.y.max, spec.y.count || 5)) {
        s.append(svg("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: ys(t), y2: ys(t) }),
          svg("text", { x: m.l - 8, y: ys(t) + 4, "text-anchor": "end" }, fy(t)));
      }
      for (const t of spec.x.ticks || niceTicks(spec.x.min, spec.x.max, spec.x.count || Math.max(3, Math.floor(W / 110)))) {
        s.append(svg("line", { class: "axis", x1: xs(t), x2: xs(t), y1: H - m.b, y2: H - m.b + 4 }),
          svg("text", { x: xs(t), y: H - m.b + 17, "text-anchor": "middle" }, fx(t)));
      }
      s.append(svg("line", { class: "axis", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
      if (spec.x.label) s.append(svg("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "label" }, spec.x.label));
      if (spec.y.label) s.append(svg("text", { x: 12, y: (m.t + H - m.b) / 2, class: "label", transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})`, "text-anchor": "middle" }, spec.y.label));
      for (const r of spec.refs || []) {
        s.append(svg("line", { class: "ref", x1: m.l, x2: W - m.r, y1: ys(r.y), y2: ys(r.y) }));
        if (r.label) s.append(svg("text", { x: W - m.r - 4, y: ys(r.y) - 6, "text-anchor": "end", class: "label" }, r.label));
      }
      for (const r of spec.vrefs || []) {
        s.append(svg("line", { class: "ref", x1: xs(r.x), x2: xs(r.x), y1: m.t, y2: H - m.b }));
        if (r.label) s.append(svg("text", { x: xs(r.x) + 5, y: m.t + 12 + (r.dy || 0), class: "label" }, r.label));
      }
      const clipId = `clip${Math.random().toString(36).slice(2)}`;
      s.append(svg("clipPath", { id: clipId }, svg("rect", { x: m.l, y: m.t - 2, width: W - m.l - m.r, height: H - m.t - m.b + 4 })));
      const plot = svg("g", { "clip-path": `url(#${clipId})` });
      for (const se of spec.series) {
        let d = "", pen = false;
        se.points.forEach(([x, y], i) => {
          if (y === null || y === undefined) { pen = false; return; }
          const X = xs(x), Y = ys(Math.max(spec.y.min - 1e9, y));
          if (!pen) { d += `M${X},${Y}`; pen = true; }
          else if (se.step) d += `H${X}V${Y}`;
          else d += `L${X},${Y}`;
        });
        if (se.step && se.points.length) d += `H${xs(spec.x.max)}`;
        plot.append(svg("path", { class: "series", d, stroke: se.color, "stroke-width": se.width || 2, "stroke-dasharray": se.dash ? "6 4" : null, opacity: se.opacity || null }));
        for (const [x, y] of se.dots || []) plot.append(svg("circle", { cx: xs(x), cy: ys(y), r: 4, fill: se.color, stroke: "var(--surface)", "stroke-width": 2 }));
      }
      s.append(plot);
      if (spec.endLabels) {
        const placed = [];
        for (const se of spec.series) {
          if (!se.endLabel) continue;
          const last = se.points[se.points.length - 1];
          let y = ys(last[1]);
          if (placed.some((p) => Math.abs(p - y) < 13)) continue;
          placed.push(y);
          s.append(svg("text", { x: W - m.r + 6, y: y + 4, class: "label" }, se.endLabel));
        }
      }
      for (const a of spec.annotations || []) {
        s.append(svg("circle", { cx: xs(a.x), cy: ys(a.y), r: 4.5, fill: a.color || "var(--ink)", stroke: "var(--surface)", "stroke-width": 2 }),
          svg("text", { x: xs(a.x) + (a.dx ?? 8), y: ys(a.y) + (a.dy ?? -8), class: "label", "text-anchor": a.anchor || "start" }, a.label));
      }
      // Hover: a crosshair that snaps to the pointer's x and lists every series there.
      const cross = svg("line", { class: "crosshair", y1: m.t, y2: H - m.b, visibility: "hidden" });
      const hit = svg("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent", tabindex: 0 });
      s.append(cross, hit);
      holder.replaceChildren(s, tip);
      const show = (px) => {
        const x = spec.x.min + ((px - m.l) / (W - m.l - m.r)) * (spec.x.max - spec.x.min);
        cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
        tip.replaceChildren(el("div", { class: "x", text: `${spec.x.name || "x"} ${fx(x)}${spec.x.unit || ""}` }),
          ...spec.series.filter((se) => !se.noTip).map((se) => {
            const v = valueAt(se, x);
            return el("div", { class: "row" }, el("span", { class: "key", style: { background: se.color } }),
              el("b", { text: v === null ? "–" : `${fy(v)}${spec.y.unit || ""}` }), el("span", { text: se.name }));
          }));
        tip.style.display = "block";
        const box = holder.getBoundingClientRect();
        const left = (px / W) * box.width;
        tip.style.left = `${Math.min(box.width - tip.offsetWidth - 4, left + 12)}px`;
        tip.style.top = "8px";
      };
      const hide = () => { cross.setAttribute("visibility", "hidden"); tip.style.display = "none"; };
      hit.addEventListener("pointermove", (e) => {
        const r = s.getBoundingClientRect();
        show(((e.clientX - r.left) / r.width) * W);
      });
      hit.addEventListener("pointerleave", hide);
      hit.addEventListener("focus", () => show((m.l + W - m.r) / 2));
      hit.addEventListener("blur", hide);
    }
    const tip = el("div", { class: "tip" });
    draw();
    if (window.ResizeObserver) new ResizeObserver(() => draw()).observe(holder);
    if (spec.table !== false) {
      const xsAll = [...new Set(spec.series.flatMap((se) => se.points.map((p) => p[0])))].sort((a, b) => a - b);
      tableToggle(root, [spec.x.name || "x", ...spec.series.map((se) => se.name)],
        xsAll.map((x) => [fx(x), ...spec.series.map((se) => { const v = valueAt(se, x); return v === null ? "–" : fy(v); })]));
    }
    return root;
  }

  /* Horizontal bars, one series: value at the tip, a mark-sized hit target with a tooltip. */
  function bars(container, spec) {
    const root = el("div", { class: "chart" });
    container.append(root);
    const max = spec.max || Math.max(...spec.items.map((i) => i.value)) || 1;
    const fv = spec.fmt || ((v) => num(v, 0));
    const rows = el("div", { style: { display: "grid", gridTemplateColumns: "minmax(96px, 22%) minmax(0, 1fr)", gap: "8px 12px", alignItems: "center" } });
    for (const it of spec.items) {
      const w = `calc(${Math.max(0.005, it.value / max)} * (100% - 128px))`; // 128 px stay free for the value at the tip
      const bar = el("div", { style: { position: "relative", height: "22px" }, title: `${it.label}: ${fv(it.value)}${it.note ? ` (${it.note})` : ""}` },
        el("div", { style: { position: "absolute", left: 0, top: "3px", height: "16px", width: w, background: it.color || "var(--b)", borderRadius: "0 4px 4px 0" } }),
        el("span", { style: { position: "absolute", left: `calc(${w} + 8px)`, top: "1px", fontSize: "13px", color: "var(--ink-2)", whiteSpace: "nowrap", fontVariantNumeric: "tabular-nums" } },
          `${fv(it.value)}${it.note ? ` ${it.note}` : ""}`));
      rows.append(el("div", { style: { fontSize: "14px", textAlign: "right", color: "var(--ink-2)" }, text: it.label }), bar);
    }
    root.append(rows);
    if (spec.table !== false) tableToggle(root, [spec.labelName || "item", spec.valueName || "value"], spec.items.map((i) => [i.label, fv(i.value)]));
    return root;
  }

  /* Several step lanes over one time axis (e.g. what each nuisance is doing). */
  function lanes(container, spec) {
    const root = el("div", { class: "chart" });
    container.append(root);
    for (const [i, lane] of spec.lanes.entries()) {
      const last = i === spec.lanes.length - 1;
      root.append(el("div", { class: "small", style: { margin: i ? "6px 0 0 54px" : "0 0 0 54px", color: "var(--ink-2)" } },
        el("span", { class: "key", style: { display: "inline-block", width: "14px", height: "2px", background: lane.color, verticalAlign: "middle", marginRight: "6px" } }),
        lane.name));
      line(root, {
        height: last ? 92 : 62, legend: false, table: false,
        x: { ...spec.x, label: last ? spec.x.label : null, ticks: last ? spec.x.ticks : [], fmt: spec.x.fmt },
        y: { min: lane.min, max: lane.max, ticks: lane.ticks, fmt: lane.fmt, unit: lane.unit },
        series: [{ name: lane.name, color: lane.color, points: lane.points, step: true }],
      });
    }
    return root;
  }

  window.Charts = { line, bars, lanes, niceTicks, valueAt };
})();
