/* Shared helpers and components for the demo pages. Data comes from window.DEMO (data/*.js). */
(function () {
  "use strict";
  const D = (window.DEMO = window.DEMO || {});
  const SVG = "http://www.w3.org/2000/svg";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  /* Build an element. Children are nodes or text (always inserted as text, never as HTML). */
  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) node.append(c.nodeType ? c : String(c));
    return node;
  }
  function svg(tag, attrs, ...children) {
    const node = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v !== null && v !== undefined) node.setAttribute(k, v);
    for (const c of children.flat()) if (c) node.append(c.nodeType ? c : document.createTextNode(String(c)));
    return node;
  }
  const get = (path, root = D) => path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), root);

  const num = (v, d = 2) => {
    if (v == null || Number.isNaN(v)) return "–";
    const x = Math.abs(v) < 0.5 * 10 ** -d ? 0 : Number(v); // never "-0.00"
    return x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  };
  const int = (v) => (v == null ? "–" : Math.round(v).toLocaleString("en-US"));
  const pct = (v, d = 2) => (v == null ? "–" : `${num(100 * v, d)}%`);
  const formats = {
    int, f0: (v) => num(v, 0), f1: (v) => num(v, 1), f2: (v) => num(v, 2), f3: (v) => num(v, 3), f4: (v) => num(v, 4),
    pct1: (v) => pct(v, 1), pct2: (v) => pct(v, 2), pct3: (v) => pct(v, 3), text: (v) => String(v),
    list: (v) => (Array.isArray(v) ? v.join(", ") : String(v)),
    date: (v) => String(v).slice(0, 10),
    duration: (s) => (s >= 3600 ? `${num(s / 3600, 1)} h` : s >= 60 ? `${num(s / 60, 0)} min` : `${num(s, 0)} s`),
  };
  const fmt = (v, f) => (v === undefined ? "–" : (formats[f] || formats.text)(v));

  /* Theme: OS preference by default; the toggle stores an explicit choice. */
  function effectiveTheme() {
    const t = document.documentElement.dataset.theme;
    if (t) return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function initTheme() {
    const btn = $(".theme-toggle");
    if (!btn) return;
    const paint = () => {
      const dark = effectiveTheme() === "dark";
      btn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
      btn.replaceChildren(svg("svg", { width: 18, height: 18, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": 2, "stroke-linecap": "round" },
        dark ? [svg("circle", { cx: 12, cy: 12, r: 4.5 }), ...[0, 45, 90, 135, 180, 225, 270, 315].map((a) => {
          const r = (a * Math.PI) / 180;
          return svg("line", { x1: 12 + 7.5 * Math.cos(r), y1: 12 + 7.5 * Math.sin(r), x2: 12 + 9.5 * Math.cos(r), y2: 12 + 9.5 * Math.sin(r) });
        })] : [svg("path", { d: "M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" })]));
    };
    btn.addEventListener("click", () => {
      const next = effectiveTheme() === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("demo-theme", next); } catch (e) { /* storage unavailable: theme lasts this visit */ }
      paint();
      document.dispatchEvent(new CustomEvent("themechange"));
    });
    paint();
  }

  function initNav() {
    const page = document.body.dataset.page;
    for (const a of $$(".site-nav a")) if (a.dataset.page === page) a.setAttribute("aria-current", "page");
  }

  function initStatus() {
    const box = $(".status-strip .wrap");
    const c = D.common;
    if (!box || !c) return;
    const done = c.phases.filter((p) => p.state === "done");
    const next = c.phases.find((p) => p.state === "next");
    const env = c.env || {};
    box.replaceChildren(
      el("span", {}, el("span", { class: "dot done" }), el("strong", { text: "Built: " }),
        `the simulator (phases ${done.map((p) => p.number).join(", ")})`),
      next ? el("span", {}, el("span", { class: "dot next" }), el("strong", { text: "Next: " }),
        `phase ${next.number}, ${next.title.toLowerCase()}. The detector itself is not built yet.`) : null,
      el("span", { class: "muted" }, `Site built ${fmt(env.built, "date")} from ${env.short || "?"}${env.dirty ? " (with uncommitted changes)" : ""}`)
    );
  }

  /* Fill <span data-v="common.counts.frames" data-fmt="int"> from the data. */
  function fillValues(root = document) {
    for (const node of $$("[data-v]", root)) {
      const v = get(node.dataset.v);
      node.textContent = fmt(v, node.dataset.fmt);
    }
  }

  /* Lightbox: click any image with .zoomable to see it large. */
  let dialog = null;
  function lightbox(src, caption, pixelated) {
    if (!dialog) {
      dialog = el("dialog", { class: "lightbox", "aria-label": "Enlarged image" });
      dialog.addEventListener("click", () => dialog.close());
      document.body.append(dialog);
    }
    dialog.replaceChildren(el("img", { src, alt: caption || "", class: pixelated ? "pixelated" : null }), caption ? el("p", { text: caption }) : null);
    dialog.showModal();
  }
  document.addEventListener("click", (e) => {
    const img = e.target.closest && e.target.closest("img.zoomable");
    if (img) lightbox(img.dataset.full || img.src, img.alt, img.classList.contains("pixelated"));
  });

  /* ---- Components ------------------------------------------------------------------------- */
  const LAYERS = [
    { key: "a", label: "Box A", color: "var(--a)", on: true },
    { key: "b", label: "Box B", color: "var(--b)", on: true },
    { key: "overlap", label: "Overlap", color: "var(--ov)", on: true },
    { key: "content", label: "Content rect", color: "#e8e7e1", on: false },
    { key: "markers", label: "Markers", color: "var(--marker)", on: true },
  ];

  function polyPoints(poly) { return poly.map((p) => p.join(",")).join(" "); }

  /* The true geometry drawn over a frame, in camera pixels (viewBox = the camera image). */
  function overlaySvg(L, on) {
    const g = svg("svg", { class: "overlay", viewBox: `-0.5 -0.5 ${L.w} ${L.h}`, preserveAspectRatio: "none", "aria-hidden": "true" });
    const sw = Math.max(L.w, L.h) / 520;
    const layer = (key) => svg("g", { "data-layer": key, style: on[key] === false ? "display:none" : null });
    const ga = layer("a"), gb = layer("b"), go = layer("overlap"), gc = layer("content"), gm = layer("markers");
    if (L.overlap && L.overlap.length) go.append(svg("polygon", { points: polyPoints(L.overlap), fill: "var(--ov)", "fill-opacity": 0.22, stroke: "var(--ov)", "stroke-width": sw }));
    if (L.content) gc.append(svg("polygon", { points: polyPoints(L.content), fill: "none", stroke: "#e8e7e1", "stroke-width": sw * 0.8, "stroke-dasharray": `${sw * 5} ${sw * 4}` }));
    ga.append(svg("polygon", { points: polyPoints(L.a), fill: "none", stroke: "var(--a)", "stroke-width": sw * 1.3 }));
    gb.append(svg("polygon", { points: polyPoints(L.b), fill: "none", stroke: "var(--b)", "stroke-width": sw * 1.3, "stroke-dasharray": `${sw * 7} ${sw * 5}` }));
    for (const m of L.markers || []) {
      gm.append(svg("polygon", { points: polyPoints(m.poly), fill: "none", stroke: "var(--marker)", "stroke-width": sw, "stroke-dasharray": m.visible ? null : `${sw * 2} ${sw * 2}` }));
    }
    for (const h of L.hotspots || []) {
      const [x, y] = h.at, r = sw * 9;
      const col = h.name === "a" ? "var(--a)" : "var(--b)";
      g.append(svg("g", {}, svg("line", { x1: x - r, y1: y, x2: x + r, y2: y, stroke: col, "stroke-width": sw * 1.5 }),
        svg("line", { x1: x, y1: y - r, x2: x, y2: y + r, stroke: col, "stroke-width": sw * 1.5 })));
    }
    for (const lab of L.labels || []) {
      const key = lab.layer;
      const target = key === "a" ? ga : key === "b" ? gb : go;
      target.append(svg("text", { x: lab.at[0], y: lab.at[1], "text-anchor": "middle", "dominant-baseline": "middle",
        "font-size": sw * 8, "font-weight": 700, fill: "#fff", stroke: "rgba(0,0,0,.75)", "stroke-width": sw * 2, "paint-order": "stroke",
        "font-family": "system-ui, sans-serif" }, lab.text));
    }
    g.append(go, gc, ga, gb, gm);
    return g;
  }

  /* A frame with its true geometry and layer toggles. opts: {alt, chips, extraToggle: {labels, srcs}} */
  function overlayFigure(container, img, L, opts = {}) {
    const on = Object.fromEntries(LAYERS.map((l) => [l.key, opts.layersOn ? opts.layersOn.includes(l.key) : l.on]));
    const image = el("img", { src: img.src, width: img.w, height: img.h, alt: opts.alt || "", loading: opts.eager ? null : "lazy", class: "zoomable" });
    const frame = el("div", { class: "frame" }, image);
    let overlay = null;
    if (L) { overlay = overlaySvg(L, on); frame.append(overlay); }
    if (opts.tag) frame.append(el("span", { class: "tag", text: opts.tag }));
    const chips = el("div", { class: "chips", role: "group", "aria-label": "Layers drawn on the picture" });
    if (L && opts.chips !== false) {
      for (const l of LAYERS) {
        if (l.key === "markers" && !(L.markers || []).length) continue;
        const chip = el("button", { class: "chip", type: "button", "aria-pressed": String(on[l.key]) },
          el("span", { class: "swatch", style: { background: l.color } }), l.label);
        chip.addEventListener("click", () => {
          on[l.key] = !on[l.key];
          chip.setAttribute("aria-pressed", String(on[l.key]));
          const g = overlay.querySelector(`[data-layer="${l.key}"]`);
          if (g) g.style.display = on[l.key] ? "" : "none";
        });
        chips.append(chip);
      }
    }
    if (opts.extraToggle) {
      const { labels, srcs } = opts.extraToggle;
      const seg = segmented(labels, 0, (i) => { image.src = srcs[i].src; });
      chips.prepend(seg);
    }
    container.append(chips.childNodes.length ? chips : "", frame);
    return { frame, image, overlay };
  }

  /* A row of buttons, one pressed. */
  function segmented(labels, start, onChange, ariaLabel) {
    const seg = el("div", { class: "seg", role: "group", "aria-label": ariaLabel || "Choose" });
    const buttons = labels.map((label, i) => {
      const b = el("button", { type: "button", "aria-pressed": String(i === start), text: label });
      b.addEventListener("click", () => {
        buttons.forEach((x, j) => x.setAttribute("aria-pressed", String(j === i)));
        onChange(i);
      });
      return b;
    });
    seg.append(...buttons);
    return seg;
  }

  /* Before/after slider over two same-size images. */
  function compare(container, before, after, labels = ["before", "after"], pixelated = false) {
    const box = el("div", { class: "frame compare" });
    const cls = pixelated ? "pixelated" : null;
    const a = el("img", { src: before.src, width: before.w, height: before.h, alt: labels[0], class: cls });
    const bwrap = el("div", { class: "after" }, el("img", { src: after.src, width: after.w, height: after.h, alt: labels[1], class: cls }));
    const range = el("input", { type: "range", min: 0, max: 100, value: 50, "aria-label": `Slide between ${labels[0]} and ${labels[1]}` });
    const set = (v) => box.style.setProperty("--pos", `${v}%`);
    range.addEventListener("input", () => set(range.value));
    box.append(a, bwrap, el("div", { class: "handle" }), el("span", { class: "tag", text: labels[0] }), el("span", { class: "tag right", text: labels[1] }), range);
    set(50);
    container.append(box);
    return box;
  }

  /* Pretty JSON with chosen keys highlighted. Text only: values never become markup. */
  function jsonView(obj, highlight = [], depth = 0, key = null) {
    const pad = "  ".repeat(depth);
    const frag = document.createDocumentFragment();
    const mark = key !== null && highlight.includes(key);
    const wrap = mark ? el("span", { class: "truth" }) : frag;
    if (key !== null) wrap.append(el("span", { class: "k", text: JSON.stringify(key) }), ": ");
    if (obj === null || typeof obj !== "object") {
      const cls = typeof obj === "string" ? "s" : "n";
      wrap.append(el("span", { class: cls, text: JSON.stringify(obj) }));
    } else if (Array.isArray(obj) && obj.every((v) => v === null || typeof v !== "object")) {
      wrap.append(el("span", { class: "n", text: JSON.stringify(obj) }));
    } else {
      const entries = Array.isArray(obj) ? obj.map((v, i) => [null, v]) : Object.entries(obj);
      wrap.append(Array.isArray(obj) ? "[" : "{", "\n");
      entries.forEach(([k, v], i) => {
        wrap.append(pad + "  ", jsonView(v, highlight, depth + 1, k), i < entries.length - 1 ? ",\n" : "\n");
      });
      wrap.append(pad + (Array.isArray(obj) ? "]" : "}"));
    }
    if (mark) frag.append(wrap);
    return frag;
  }

  /* Figures render after load and push content down, so jump to the page's #anchor again once they are in. */
  function restoreHash() {
    if (!location.hash) return;
    const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (target) requestAnimationFrame(() => target.scrollIntoView());
  }

  window.Demo = { D, $, $$, el, svg, get, fmt, num, int, pct, fillValues, lightbox, overlayFigure, overlaySvg, segmented, compare, jsonView, effectiveTheme, restoreHash };

  document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    initNav();
    initStatus();
  });
})();
