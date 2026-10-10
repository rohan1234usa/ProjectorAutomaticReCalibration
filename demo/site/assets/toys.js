/* Interactive explainers for the algorithm page. Each is a small model of one idea from
   CLAUDE.md section 4, computed in the browser; none of them is the detector. */
(function () {
  "use strict";
  const { el, svg, num } = window.Demo;
  const { line } = window.Charts;

  function slider(label, min, max, step, value, fmtv, onInput) {
    const out = el("output", { text: fmtv(value) });
    const input = el("input", { type: "range", min, max, step, value, "aria-label": label });
    input.addEventListener("input", () => { out.textContent = fmtv(+input.value); onInput(+input.value); });
    return { node: el("div", { class: "slider" }, el("label", {}, el("span", { text: label }), out), input), input };
  }
  const readout = () => el("div", { class: "readouts" });

  /* ---- FFT (radix 2, in place) ------------------------------------------------------------- */
  function fft(re, im, inverse = false) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = ((inverse ? 2 : -2) * Math.PI) / len;
      for (let i = 0; i < n; i += len) {
        for (let k = 0; k < len / 2; k++) {
          const wr = Math.cos(ang * k), wi = Math.sin(ang * k);
          const ur = re[i + k], ui = im[i + k];
          const vr = re[i + k + len / 2] * wr - im[i + k + len / 2] * wi;
          const vi = re[i + k + len / 2] * wi + im[i + k + len / 2] * wr;
          re[i + k] = ur + vr; im[i + k] = ui + vi;
          re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        }
      }
    }
    if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }

  /* S(x - d) for a fractional d, by a phase ramp in the Fourier domain (circular). */
  function shifted(s, d) {
    const n = s.length, re = Float64Array.from(s), im = new Float64Array(n);
    fft(re, im);
    for (let k = 0; k < n; k++) {
      const f = k <= n / 2 ? k : k - n;
      const a = (-2 * Math.PI * f * d) / n, c = Math.cos(a), si = Math.sin(a);
      [re[k], im[k]] = [re[k] * c - im[k] * si, re[k] * si + im[k] * c];
    }
    fft(re, im, true);
    return Array.from(re);
  }

  /* The cepstrum pipeline of CLAUDE.md 4.3 in one dimension: linear high-pass, Hann window, log power, back. */
  function cepstrum(o) {
    const n = o.length;
    const blur = (x) => { const r = 3, out = new Array(n).fill(0); let ws = 0; const w = [];
      for (let k = -3 * r; k <= 3 * r; k++) { const v = Math.exp(-0.5 * (k / r) ** 2); w.push([k, v]); ws += v; }
      for (let i = 0; i < n; i++) for (const [k, v] of w) out[i] += (v / ws) * x[(i + k + n) % n];
      return out; };
    const b = blur(o);
    const re = new Float64Array(n), im = new Float64Array(n);
    for (let i = 0; i < n; i++) re[i] = (o[i] - b[i]) * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)));
    fft(re, im);
    let mean = 0;
    const p = Array.from(re, (r, k) => r * r + im[k] * im[k]);
    p.forEach((v) => (mean += v / n));
    const lr = Float64Array.from(p, (v) => Math.log(v + 1e-6 * mean)), li = new Float64Array(n);
    fft(lr, li, true);
    return Array.from(lr, (r, k) => Math.hypot(r, li[k]));
  }

  function cepstrumToy(container, row, floorPx) {
    const s0 = row.values;
    const mean = s0.reduce((a, b) => a + b, 0) / s0.length;
    const S = s0.map((v) => v - mean);
    const state = { d: 6, a: 0.5 };
    const sig = el("div"), cep = el("div"), ro = readout();
    const sd = slider("Echo distance d (camera px)", 0, 16, 0.25, state.d, (v) => num(v, 2), (v) => { state.d = v; draw(); });
    const sa = slider("A's share of the overlap a", 0.2, 0.8, 0.05, state.a, (v) => num(v, 2), (v) => { state.a = v; draw(); });
    container.append(el("div", { class: "toy" },
      el("div", { class: "toy-title", text: "One camera row of text through the overlap, seen twice" }),
      el("p", { class: "small muted", text: "O(x) = a·S(x) + (1 − a)·S(x − d). The row is real: 256 camera pixels of an aligned simulator frame. Move d and watch the cepstrum peak follow it; below the floor it merges with the origin." }),
      el("div", { class: "sliders" }, sd.node, sa.node), sig, cep, ro));
    const base = cepstrum(S);
    function draw() {
      const O = shifted(S, state.d).map((v, i) => state.a * S[i] + (1 - state.a) * v);
      const c = cepstrum(O);
      sig.replaceChildren(); cep.replaceChildren();
      line(sig, { height: 170, table: false, x: { min: 0, max: S.length - 1, name: "x", unit: " px", label: "camera pixel along the row" },
        y: { min: Math.min(...S) * 1.1, max: Math.max(...S) * 1.1, fmt: (v) => num(v, 2) },
        series: [{ name: "S, one copy", color: "var(--axis)", points: S.map((v, i) => [i, v]) },
                 { name: "O, both copies", color: "var(--b)", points: O.map((v, i) => [i, v]) }] });
      const q = 40;
      let best = -1, at = 0;
      for (let k = Math.ceil(floorPx); k <= q; k++) if (c[k] - base[k] > best) { best = c[k] - base[k]; at = k; }
      line(cep, { height: 190, table: false, x: { min: 0, max: q, name: "quefrency", unit: " px", label: "quefrency (camera px)" },
        y: { min: 0, max: Math.max(0.05, ...c.slice(1, q + 1)) * 1.15, fmt: (v) => num(v, 2) },
        bands: [{ x0: 0, x1: floorPx, label: "floor" }], vrefs: [{ x: state.d, label: `d = ${num(state.d, 2)} px` }],
        series: [{ name: "cepstrum of S alone (content)", color: "var(--axis)", points: base.slice(0, q + 1).map((v, i) => [i, v]) },
                 { name: "cepstrum of O", color: "var(--b)", points: c.slice(0, q + 1).map((v, i) => [i, v]) }] });
      ro.replaceChildren(
        el("div", { class: "readout" }, "Strongest rise over the content beyond the floor: ", el("b", { text: `${at} px` })),
        el("div", { class: "readout" }, "True echo: ", el("b", { text: `${num(state.d, 2)} px` }),
          state.d <= floorPx ? el("span", { class: "verdict hold", text: "  inside the floor: unresolved" }) : ""));
    }
    draw();
  }

  /* ---- Hotspot ramp -------------------------------------------------------------------------- */
  function hotspotToy(container, opts) {
    const W = opts.widthPx, k = opts.factor, warn = opts.lampWarn;
    const state = { u: 2, g: 1 };
    const ramps = el("div"), delta = el("div"), ro = readout();
    const su = slider("B moved across by u (projector px)", -4, 4, 0.1, state.u, (v) => num(v, 1), (v) => { state.u = v; draw(); });
    const sg = slider("B's lamp at g of its calibrated brightness", 0.8, 1.0, 0.005, state.g, (v) => `${num(100 * v, 1)}%`, (v) => { state.g = v; draw(); });
    container.append(el("div", { class: "toy" },
      el("div", { class: "toy-title", text: "The blend ramp, moved and dimmed" }),
      el("p", { class: "small muted", text: `A fades out and B fades in across a ${num(W, 0)} px overlap, a(x) + b(x) = 1. Moving B moves its ramp; dimming B scales it.` }),
      el("div", { class: "sliders" }, su.node, sg.node), ramps, delta, ro));
    const ramp = (x) => (x <= 0 ? 0 : x >= W ? 1 : 0.5 - 0.5 * Math.cos((Math.PI * x) / W));
    function draw() {
      const xs = [];
      for (let x = -80; x <= W + 80; x += 2) xs.push(x);
      const a = xs.map((x) => [x, 1 - ramp(x)]), b = xs.map((x) => [x, state.g * ramp(x - state.u)]);
      const sum = xs.map((x, i) => [x, a[i][1] + b[i][1]]);
      const d = xs.map((x, i) => [x, 100 * k * (sum[i][1] - 1)]);
      const shapeOnly = xs.map((x) => [x, 100 * k * (ramp(x - state.u) - ramp(x))]);
      ramps.replaceChildren(); delta.replaceChildren();
      line(ramps, { height: 170, table: false, x: { min: xs[0], max: xs[xs.length - 1], name: "x", unit: " px" },
        y: { min: 0, max: 1.1, ticks: [0, 0.5, 1], fmt: (v) => num(v, 1) },
        series: [{ name: "a(x), projector A", color: "var(--a)", points: a }, { name: "g·b(x − u), projector B", color: "var(--b)", points: b },
                 { name: "a + g·b, the overlap", color: "var(--ov)", points: sum, dash: true }] });
      const lo = Math.min(-0.2, ...d.map((p) => p[1])), hi = Math.max(0.2, ...d.map((p) => p[1]));
      line(delta, { height: 190, table: false, x: { min: xs[0], max: xs[xs.length - 1], name: "x", unit: " px", label: "position across the overlap (projector px)" },
        y: { min: lo * 1.15, max: hi * 1.15, fmt: (v) => `${num(v, 2)}%` },
        refs: [{ y: 0 }],
        series: [{ name: "brightness change seen by the camera", color: "var(--b)", points: d },
                 { name: "from the move alone", color: "var(--a)", points: shapeOnly, dash: true }] });
      const dip = Math.min(...shapeOnly.map((p) => p[1])), bump = Math.max(...shapeOnly.map((p) => p[1]));
      const lampOff = Math.abs(state.g - 1) > warn;
      ro.replaceChildren(
        el("div", { class: "readout" }, "Move signature: ", el("b", { text: `${num(Math.abs(dip) > bump ? dip : bump, 2)}%` }), " at the ramp"),
        el("div", { class: "readout" }, "Lamp: ", el("b", { text: `${num(100 * (state.g - 1), 1)}%` }),
          lampOff ? el("span", { class: "verdict hold", text: `  over LAMP_WARN (${num(100 * warn, 0)}%): a lamp warning, not a YES` }) : ""),
        el("div", { class: "readout" }, "Room light and black level dilute it to ", el("b", { text: `${num(100 * k, 1)}%` })));
    }
    draw();
  }

  /* ---- Boundary boxes --------------------------------------------------------------------- */
  function boundaryToy(container, arrangements, cfg, pitch) {
    const order = ["side_by_side", "stacked", "rotated", "corner", "different_sizes", "large_overlap"];
    const names = [...order.filter((n) => n in arrangements), ...Object.keys(arrangements).filter((n) => !order.includes(n))];
    const state = { name: names[0], tx: 4, ty: 0, rot: 0, kx: 0, ky: 0 };
    const pic = el("div"), ro = readout();
    const select = el("select", { "aria-label": "Arrangement" }, names.map((n) => el("option", { value: n, text: n.replace(/_/g, " ") })));
    select.addEventListener("change", () => { state.name = select.value; draw(); });
    const sliders = [
      slider("B moves right (mm)", -10, 10, 0.1, state.tx, (v) => num(v, 1), (v) => { state.tx = v; draw(); }),
      slider("B moves down (mm)", -10, 10, 0.1, state.ty, (v) => num(v, 1), (v) => { state.ty = v; draw(); }),
      slider("B turns (degrees)", -0.3, 0.3, 0.005, state.rot, (v) => num(v, 3), (v) => { state.rot = v; draw(); }),
      slider("Camera bumped right (mm)", -30, 30, 0.5, state.kx, (v) => num(v, 1), (v) => { state.kx = v; draw(); }),
      slider("Camera bumped down (mm)", -30, 30, 0.5, state.ky, (v) => num(v, 1), (v) => { state.ky = v; draw(); }),
    ];
    container.append(el("div", { class: "toy" },
      el("div", { class: "controls" }, el("label", { class: "small" }, "Arrangement ", select)),
      el("div", { class: "sliders" }, sliders.map((s) => s.node)), pic, ro));
    const EXAG = 25;
    function draw() {
      const g = arrangements[state.name];
      const cb = g.b.reduce((acc, p) => [acc[0] + p[0] / 4, acc[1] + p[1] / 4], [0, 0]);
      const th = (state.rot * Math.PI) / 180, c = Math.cos(th), s = Math.sin(th);
      const moveB = ([x, y]) => [cb[0] + c * (x - cb[0]) - s * (y - cb[1]) + state.tx, cb[1] + s * (x - cb[0]) + c * (y - cb[1]) + state.ty];
      const k = [state.kx, state.ky];
      const seen = (p, owner, ex = 1) => { const m = owner === "b" ? moveB(p) : p; return [p[0] + ex * (m[0] - p[0] + k[0]), p[1] + ex * (m[1] - p[1] + k[1])]; };
      const [W, H] = g.screen, pad = 120;
      const v = svg("svg", { viewBox: `${-pad} ${-pad} ${W + 2 * pad} ${H + 2 * pad}`, role: "img", "aria-label": "The two boxes, their edge pieces and what each piece measures" });
      v.append(svg("rect", { x: 0, y: 0, width: W, height: H, fill: "var(--surface-2)", stroke: "var(--border-2)" }));
      const pts = (poly) => poly.map((p) => p.join(",")).join(" ");
      v.append(svg("polygon", { points: pts(g.overlap), fill: "var(--ov)", "fill-opacity": 0.18, stroke: "none" }));
      v.append(svg("polygon", { points: pts(g.a), fill: "none", stroke: "var(--a)", "stroke-width": 4, opacity: 0.35 }),
        svg("polygon", { points: pts(g.b), fill: "none", stroke: "var(--b)", "stroke-width": 4, opacity: 0.35, "stroke-dasharray": "24 16" }));
      v.append(svg("polygon", { points: pts(g.a.map((p) => seen(p, "a", EXAG))), fill: "none", stroke: "var(--a)", "stroke-width": 9 }),
        svg("polygon", { points: pts(g.b.map((p) => seen(p, "b", EXAG))), fill: "none", stroke: "var(--b)", "stroke-width": 9, "stroke-dasharray": "30 20" }));
      let maxObs = 0;
      for (const piece of g.pieces) {
        const m = piece.mid, owner = piece.owner;
        const moved = seen(m, owner);
        const u = [moved[0] - m[0], moved[1] - m[1]];
        const obs = piece.normal[0] * u[0] + piece.normal[1] * u[1];
        maxObs = Math.max(maxObs, Math.abs(obs));
        const col = owner === "a" ? "var(--a)" : "var(--b)";
        const base = seen(m, owner, EXAG);
        v.append(svg("line", { x1: base[0], y1: base[1], x2: base[0] + piece.normal[0] * obs * EXAG, y2: base[1] + piece.normal[1] * obs * EXAG,
          stroke: col, "stroke-width": 12, "stroke-linecap": "round", opacity: piece.kind === "inner" ? 0.55 : 1 }));
        v.append(svg("circle", { cx: base[0], cy: base[1], r: 13, fill: piece.kind === "inner" ? "var(--surface)" : col, stroke: col, "stroke-width": 5 }));
      }
      pic.replaceChildren(v);
      // B's motion relative to A: the camera's shared shift cancels.
      let off = 0;
      for (const p of g.overlap) { const q = moveB(p); off = Math.max(off, Math.hypot(q[0] - p[0], q[1] - p[1])); }
      const tol = cfg.TOLERANCE_MM;
      const verdict = off > tol ? el("span", { class: "verdict yes", text: "over tolerance: votes YES" }) : el("span", { class: "verdict no", text: "within tolerance: NO" });
      const inner = g.pieces.filter((p) => p.kind === "inner").length;
      ro.replaceChildren(
        el("div", { class: "readout" }, "Relative offset (B vs A): ", el("b", { text: `${num(off, 2)} mm = ${num(off / pitch, 2)} px` }), " ", verdict),
        el("div", { class: "readout" }, "Shared camera term: ", el("b", { text: `${num(Math.hypot(...k), 1)} mm` }), " (cancels)"),
        el("div", { class: "readout" }, "Pieces: ", el("b", { text: `${g.pieces.length - inner} outer, ${inner} inner` }), " · largest n̂·u ", el("b", { text: `${num(maxObs, 2)} mm` })));
    }
    draw();
  }

  /* ---- Fusion and decision: K of N votes, hysteresis, None holds --------------------------------- */
  function decisionToy(container, cfg) {
    const [K, N] = cfg.YES_VOTES, tol = cfg.TOLERANCE_MM, clear = cfg.CLEAR_RATIO * tol;
    let seed = 7;
    const rand = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
    const noise = (s) => (rand() - 0.5) * 2 * s;
    const presets = {
      "B shifts 4 px": (i) => (i < 9 ? 0.15 + noise(0.1) : 4.17 + noise(0.15)),
      "Slow drift": (i) => Math.max(0, 0.09 * i + noise(0.12)),
      "Camera bump": (i) => (i >= 10 && i <= 12 ? null : 0.2 + noise(0.12)),
      "Near the threshold": () => 1.7 + noise(0.35),
      "Comes and goes": (i) => Math.max(0, 1.2 + 1.1 * Math.sin((2 * Math.PI * i) / 14) + noise(0.08)),
    };
    const names = Object.keys(presets);
    let current = names[0];
    const chartBox = el("div"), strip = el("div", { style: { display: "grid", gridTemplateColumns: "repeat(30, 1fr)", gap: "2px", margin: "6px 0 0 54px" } }), ro = readout();
    const seg = window.Demo.segmented(names, 0, (i) => { current = names[i]; seed = 7; draw(); }, "Scenario");
    const again = el("button", { class: "chip", type: "button", text: "New noise" });
    again.addEventListener("click", () => { seed = (seed * 7 + 13) % 2147483648 || 3; draw(); });
    container.append(el("div", { class: "toy" }, el("div", { class: "controls" }, seg, again), chartBox, strip, ro));
    function draw() {
      const offsets = Array.from({ length: 30 }, (_, i) => presets[current](i));
      let answer = "NO";
      const votes = [], answers = [];
      for (const v of offsets) {
        if (v === null) { answers.push(["hold", answer]); continue; }
        votes.push(v > tol);
        if (votes.length > N) votes.shift();
        if (answer === "NO" && votes.filter(Boolean).length >= K) answer = "YES";
        else if (answer === "YES" && v < clear) { answer = "NO"; votes.length = 0; }
        answers.push([answer, answer]);
      }
      chartBox.replaceChildren();
      const pts = offsets.map((v, i) => [i + 1, v]);
      line(chartBox, { height: 210, table: false, x: { min: 0.5, max: 30.5, name: "interval", ticks: [1, 5, 10, 15, 20, 25, 30], fmt: (v) => num(v, 0), label: "30-second interval" },
        y: { min: 0, max: Math.max(5, ...offsets.filter((v) => v !== null)) * 1.05, fmt: (v) => num(v, 1), unit: " mm" },
        refs: [{ y: tol, label: `TOLERANCE_MM ${num(tol, 2)}` }, { y: clear, label: `back to NO below ${num(clear, 2)}` }],
        series: [{ name: "measured offset", color: "var(--b)", points: pts, dots: pts.filter((p) => p[1] !== null) }] });
      strip.replaceChildren(...answers.map(([kind, a], i) => el("div", {
        title: `interval ${i + 1}: ${kind === "hold" ? "can't tell, holds " : ""}${a}`,
        style: { height: "22px", borderRadius: "4px", background: kind === "hold" ? "var(--warn)" : a === "YES" ? "var(--bad)" : "var(--good)", opacity: kind === "hold" ? 0.55 : 0.85 },
      })));
      const firstYes = answers.findIndex((a) => a[1] === "YES");
      ro.replaceChildren(
        el("div", { class: "readout" }, el("span", { class: "dot", style: { background: "var(--good)" } }), "NO  ",
          el("span", { class: "dot", style: { background: "var(--bad)" } }), "YES  ",
          el("span", { class: "dot", style: { background: "var(--warn)" } }), "can't tell (holds the last answer)"),
        el("div", { class: "readout" }, "First YES: ", el("b", { text: firstYes < 0 ? "never" : `interval ${firstYes + 1}` })),
        el("div", { class: "readout" }, "Rule: ", el("b", { text: `${K} of the last ${N} answers over ${num(tol, 2)} mm` })));
    }
    draw();
  }

  /* ---- Tolerance from the viewing distance -------------------------------------------------------- */
  function toleranceToy(container, cfg, pitch) {
    const out = readout();
    const tanArcmin = Math.tan(Math.PI / (180 * 60));
    const s = slider("Closest seat (metres from the screen)", 2, 20, 0.5, 6, (v) => `${num(v, 1)} m`, (v) => draw(v));
    container.append(el("div", { class: "toy" }, el("div", { class: "sliders" }, s.node), out));
    function draw(d) {
      const mm = d * 1000 * tanArcmin;
      out.replaceChildren(
        el("div", { class: "readout" }, "One arcminute there is ", el("b", { text: `${num(mm, 2)} mm` }), " = ", el("b", { text: `${num(mm / pitch, 2)} projector px` })),
        el("div", { class: "readout" }, "The config's TOLERANCE_MM: ", el("b", { text: `${num(cfg.TOLERANCE_MM, 2)} mm` }), ` (one arcminute at ${num(cfg.TOLERANCE_MM / (1000 * tanArcmin), 1)} m)`));
    }
    draw(6);
  }

  window.Toys = { cepstrumToy, hotspotToy, boundaryToy, decisionToy, toleranceToy, cepstrum, shifted };
})();
