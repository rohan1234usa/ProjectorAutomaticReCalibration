/* Tests page: the suite as run at build time, test by test; phases; findings. */
(function () {
  "use strict";
  const { D, $, el, restoreHash, num, int, fillValues } = window.Demo;
  const OUTCOME = { passed: "passed", failed: "failed", error: "error", skipped: "skipped" };

  function header(T, C) {
    const r = T.run, env = C.env || {};
    $("#run-line").textContent = `Ran ${r.command.replace(/"/g, "'")} when the site was built: ${r.summary || "no summary"} (${num(r.wall_s, 0)} s wall clock). `
      + `Python ${env.python}, commit ${env.short}${env.dirty ? " with uncommitted changes" : ""}.`;
    const t = T.totals;
    const bad = t.failed + t.error;
    const tile = (label, value, sub) => el("div", { class: "kpi" }, el("div", { class: "label", text: label }), el("div", { class: "value", text: value }), el("div", { class: "sub", text: sub || "" }));
    $("#test-kpis").append(
      tile("Passed", int(t.passed), `of ${int(t.total)} run`),
      tile("Failed", int(bad), bad ? "see the explorer below" : "none"),
      tile("Skipped", int(t.skipped), ""),
      tile("Slow, not run", int(t.slow_not_run), t.slow_not_run ? "demo-scale; build with \u2011\u2011tests all" : "all run"),
      tile("Time in tests", `${num(t.seconds, 0)} s`, "sum of test durations"));
  }

  function areas(T) {
    window.Charts.bars($("#area-bars"), {
      labelName: "area", valueName: "tests",
      items: T.areas.map((a) => ({ label: a.area, value: a.count, color: a.failed + a.error ? "var(--bad)" : "var(--b)",
        note: `tests · ${a.failed + a.error ? `${a.failed + a.error} failing · ` : ""}${num(a.seconds, 1)} s` })),
    });
    $("#slowest").append(el("div", { class: "table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", { text: "Test" }), el("th", { class: "num", text: "Seconds" }))),
      el("tbody", {}, T.slowest.map((s) => el("tr", {}, el("td", { class: "mono", text: `${s.file}::${s.func}${s.params ? `[${s.params}]` : ""}` }),
        el("td", { class: "num", text: num(s.seconds, 2) })))))));
  }

  function explorer(T) {
    const list = $("#test-list"), search = $("#test-search"), fileSel = $("#test-file"), outSel = $("#test-outcome");
    for (const f of T.files) fileSel.append(el("option", { value: f.file, text: `${f.file} (${f.count})` }));
    const fn = (c) => (T.functions[c.file] || {})[c.func] || {};
    const haystack = T.cases.map((c) => [c.file, c.func, c.params, c.title, fn(c).doc || ""].join(" ").toLowerCase());
    function row(c) {
      const d = el("details");
      const summary = el("summary", {},
        el("span", { class: `badge ${OUTCOME[c.outcome] || ""}`, text: c.outcome }),
        el("span", { class: "t-name" }, el("span", { text: c.title }), " ",
          el("span", { class: "file", text: `${c.file}::${c.func}` }), c.params ? el("span", { class: "params", text: ` [${c.params}]` }) : null,
          c.slow ? el("span", { class: "badge slow", text: "slow", style: { marginLeft: "6px" } }) : null),
        el("span", { class: "t-time", text: `${num(c.seconds, 3)} s` }));
      d.append(summary);
      d.addEventListener("toggle", () => {
        if (!d.open || d.dataset.built) return;
        d.dataset.built = "1";
        const f = fn(c);
        const body = el("div", { class: "t-body" });
        if (f.doc) body.append(el("p", { text: f.doc }));
        if (c.message) body.append(el("pre", { class: "code", text: c.message }));
        if (f.source) body.append(el("pre", { class: "code", text: f.source }), el("p", { class: "small muted", text: `tests/${c.file}.py, line ${f.line}` }));
        d.append(body);
      });
      return d;
    }
    function render() {
      const q = search.value.trim().toLowerCase(), file = fileSel.value, outcome = outSel.value;
      const words = q.split(/\s+/).filter(Boolean);
      const shown = T.cases.filter((c, i) => (!file || c.file === file) && (!outcome || c.outcome === outcome)
        && words.every((w) => haystack[i].includes(w)));
      list.replaceChildren(...shown.map(row));
      if (!shown.length) list.append(el("p", { class: "small muted", style: { padding: "12px" }, text: "No test matches." }));
      $("#test-count").textContent = `${shown.length} of ${T.cases.length}`;
    }
    for (const c of [search, fileSel, outSel]) c.addEventListener("input", render);
    render();
    const slow = T.run.slow_not_run || [];
    if (slow.length) {
      $("#slow-not-run").append(el("details", { class: "more" }, el("summary", { text: `${slow.length} slow tests not run in this build` }),
        el("div", { class: "body" }, el("p", { text: "They render frames at demo scale and take minutes. Rebuild with --tests all to include them." }),
          el("pre", { class: "code", text: slow.join("\n") }))));
    }
  }

  function files(T) {
    const rows = T.files.map((f) => el("tr", {},
      el("td", { class: "mono", text: f.file }), el("td", { text: f.area }),
      el("td", { class: "small", text: f.doc.split("\n\n")[0].replace(/\n/g, " ") }),
      el("td", { class: "num", text: `${int(f.passed)} / ${int(f.count)}` }),
      el("td", { class: "num", text: `${num(f.seconds, 1)} s` })));
    $("#file-table").append(el("div", { class: "table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, ["File", "Area", "What it checks", "Passed", "Time"].map((h, i) => el("th", { class: i > 2 ? "num" : null, text: h })))),
      el("tbody", {}, rows))));
  }

  function phases(C) {
    const html = (s) => { const td = el("td", { class: "small" }); td.innerHTML = s; return td; }; // made by demo/markdown.py, text escaped
    const rows = C.phases.map((p) => el("tr", { class: `state-${p.state}` },
      el("td", {}, el("strong", { text: p.number }), el("div", { class: "small", text: p.title })),
      html(p.scope), html(p.done_condition),
      el("td", { class: "small" }, el("span", { class: `badge ${p.state}`, text: p.state }), p.status ? el("div", { style: { marginTop: "6px" } }) : null)));
    C.phases.forEach((p, i) => { if (p.status) rows[i].lastChild.lastChild.innerHTML = p.status; });
    $("#phase-table").append(el("div", { class: "table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, ["Phase", "Scope", "Done when", "Status"].map((h) => el("th", { text: h })))), el("tbody", {}, rows))));
  }

  function findings(C) {
    const box = $("#findings-list");
    C.findings.forEach((f, i) => {
      const body = el("div", { class: "body" });
      body.innerHTML = f.html; // docs/findings.md through demo/markdown.py, text escaped
      const d = el("details", { class: "more finding", id: f.anchor, open: i === C.findings.length - 1 ? true : null },
        el("summary", {}, el("span", { class: "muted", text: f.date, style: { marginRight: "8px", fontVariantNumeric: "tabular-nums" } }), el("span", {})), body);
      d.querySelector("summary span:last-child").innerHTML = f.title;
      box.append(d);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    fillValues();
    const T = D.tests, C = D.common;
    phases(C);
    findings(C);
    if (!T) {
      $("main").prepend(el("div", { class: "wrap" }, el("p", { class: "note", text: "The test results were not built. Run python -m scripts.make_site." })));
      return;
    }
    header(T, C);
    areas(T);
    explorer(T);
    files(T);
    restoreHash();
  });
})();
