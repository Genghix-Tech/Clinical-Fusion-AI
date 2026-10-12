"use strict";
/* ClinicalFusion dashboard. Dependency-free; all dynamic text goes through textContent. */
const EXP = ["A", "B", "C", "D1", "D2"];
const SHORT = { A: "A Image", B: "B Tabular", C: "C Time-series", D1: "D1 Late fusion", D2: "D2 Attention fusion" };
const MODS = ["image", "tabular", "timeseries"];
const MOD_KEY = { image: "A", tabular: "B", timeseries: "C" };
const TAB_FEATURES = ["age", "sex", "bmi", "sbp", "dbp", "heart_rate_adm", "creatinine", "lactate", "wbc", "hemoglobin", "sodium", "glucose"];
const GRAD = { // [from, to]
  A: ["#2563eb", "#22d3ee"], B: ["#f97316", "#fbbf24"], C: ["#059669", "#34d399"], D1: ["#7c3aed", "#c084fc"], D2: ["#db2777", "#fb7185"],
  acc: ["#4f46e5", "#a855f7"], pos: ["#e11d48", "#fb7185"], neg: ["#2563eb", "#38bdf8"], hw: ["#ffffff", "#fde68a"], hw2: ["#67e8f9", "#ffffff"],
};
const SOLID = { A: "#2f6df0", B: "#f08a1c", C: "#10a37f", D1: "#8a4fe8", D2: "#e0348a", acc: "#6d4af0" };
const lg = (k) => `linear-gradient(135deg, ${GRAD[k][0]}, ${GRAD[k][1]})`;
const $ = (id) => document.getElementById(id);
let state = { runs: [], runId: null, res: null, poll: null };

/* ---------- DOM helpers ---------- */
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") el.className = v; else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
  }
  add(el, kids); return el;
}
function add(el, kids) { for (const k of kids.flat(Infinity)) if (k != null && k !== false) el.append(k.nodeType ? k : document.createTextNode(String(k))); }
const NS = "http://www.w3.org/2000/svg";
function s(tag, attrs, ...kids) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  add(el, kids); return el;
}
const f3 = (x) => (x == null ? "–" : Number(x).toFixed(3));
const pct = (x) => (x == null ? "–" : (100 * x).toFixed(1) + "%");
const ci = (c) => (c ? `[${c[0].toFixed(3)}, ${c[1].toFixed(3)}]` : "");
const sgn = (x) => (Math.abs(x) < 0.0005 ? "±" : x >= 0 ? "+" : "−") + Math.abs(x).toFixed(3);
const chip = (k) => h("span", { class: "chip", style: `background:${lg(k)}` });
const card = (title, ...body) => h("div", { class: "card" }, title ? h("h2", {}, title) : null, body);
function table(head, rows, cls) {
  return h("div", { class: "scroll" }, h("table", { class: cls || "" }, h("thead", {}, h("tr", {}, head.map((x) => h("th", {}, x)))),
    h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c) => h("td", {}, c)))))));
}

/* ---------- graphics: defs, icons ---------- */
function installDefs() {
  const defs = s("defs");
  for (const [k, [a, b]] of Object.entries(GRAD)) defs.append(s("linearGradient", { id: "g-" + k, x1: 0, y1: 0, x2: 1, y2: 0 }, s("stop", { offset: "0%", "stop-color": a }), s("stop", { offset: "100%", "stop-color": b })));
  for (const [k, c] of Object.entries(SOLID)) defs.append(s("linearGradient", { id: "a-" + k, x1: 0, y1: 0, x2: 0, y2: 1 }, s("stop", { offset: "0%", "stop-color": c, "stop-opacity": 0.34 }), s("stop", { offset: "100%", "stop-color": c, "stop-opacity": 0 })));
  document.body.prepend(s("svg", { width: 0, height: 0, style: "position:absolute", "aria-hidden": "true" }, defs));
}
const ICONS = {
  image: '<rect x="3" y="3" width="18" height="18" rx="4"/><ellipse cx="9" cy="13" rx="2.6" ry="4.6"/><ellipse cx="15" cy="13" rx="2.6" ry="4.6"/><path d="M12 6v12"/>',
  tabular: '<rect x="3" y="4" width="18" height="16" rx="3"/><path d="M3 10h18M3 15h18M9 4v16"/>',
  timeseries: '<path d="M2 12h4l3-8 4 16 3-8h6"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.2"/>',
  trend: '<path d="M3 17l6-6 4 4 8-9"/><path d="M15 6h6v6"/>',
  gauge: '<path d="M4 18a9 9 0 1 1 16 0"/><path d="M12 13l4-4"/>',
  users: '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c.6-3.6 3.3-5.5 6.5-5.5s5.9 1.9 6.5 5.5"/><path d="M16 5a3.4 3.4 0 0 1 0 6.4M18 14.8c2 .7 3.3 2.5 3.6 5.2"/>',
  layers: '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>', alert: '<path d="M12 4l9.5 16h-19z"/><path d="M12 10v4.5M12 17.5v.1"/>',
  spark: '<path d="M12 3l1.9 5.6L19.5 10l-5.6 1.9L12 17.5l-1.9-5.6L4.5 10l5.6-1.4z"/>',
};
function icon(name, size = 18) {
  const el = s("svg", { viewBox: "0 0 24 24", width: size, height: size, fill: "none", stroke: "currentColor", "stroke-width": 1.9, "stroke-linecap": "round", "stroke-linejoin": "round" });
  el.innerHTML = ICONS[name]; return el;
}
function kpi(v, l, { ico, k } = {}) {
  return h("div", { class: "kpi", style: k ? `--k:${k}` : null }, ico ? h("div", { class: "ico" }, icon(ico, 18)) : null, h("div", { class: "v" }, v), h("div", { class: "l" }, l));
}

/* ---------- graphics: rings ---------- */
function ringArc(cx, cy, r, stroke, value, key, track = "var(--track)") {
  const C = 2 * Math.PI * r, len = C * Math.max(0, Math.min(1, value));
  return [s("circle", { cx, cy, r, fill: "none", stroke: track, "stroke-width": stroke }),
    s("circle", { class: "ring-arc", cx, cy, r, fill: "none", stroke: `url(#g-${key})`, "stroke-width": stroke, "stroke-linecap": "round", transform: `rotate(-90 ${cx} ${cy})`, "stroke-dasharray": `${len} ${C}`, style: `--len:${len};--C:${C}` })];
}
function ring({ value, size = 120, stroke = 13, key = "acc", text, sub }) {
  const svg = s("svg", { viewBox: `0 0 ${size} ${size}`, width: size, height: size }); svg.append(...ringArc(size / 2, size / 2, (size - stroke) / 2, stroke, value, key));
  svg.append(s("text", { x: size / 2, y: size / 2 + size * 0.06, "text-anchor": "middle", style: `font-size:${size * 0.22}px;font-weight:750` }, text));
  if (sub) svg.append(s("text", { x: size / 2, y: size / 2 + size * 0.2, "text-anchor": "middle", class: "muted", style: `font-size:${size * 0.09}px` }, sub));
  return svg;
}
const ringBox = (opts, label) => h("div", { class: "ringbox" }, ring(opts), h("div", { class: "l" }, label));
function heroRings(f, b) {
  const size = 230, c = size / 2, svg = s("svg", { viewBox: `0 0 ${size} ${size}`, width: size, height: size });
  svg.append(...ringArc(c, c, 98, 16, (f - 0.5) / 0.5, "hw", "rgba(255,255,255,.22)"), ...ringArc(c, c, 72, 16, (b - 0.5) / 0.5, "hw2", "rgba(255,255,255,.22)"));
  svg.append(s("text", { x: c, y: c + 6, "text-anchor": "middle", style: "font-size:38px;font-weight:800" }, f3(f)), s("text", { x: c, y: c + 28, "text-anchor": "middle", style: "font-size:11.5px;opacity:.85" }, "fusion AUROC"));
  return svg;
}

/* ---------- graphics: charts ---------- */
const axisTicks = (lo, hi, n) => Array.from({ length: n + 1 }, (_, i) => lo + ((hi - lo) * i) / n);
function pathD(pts, smooth) {
  if (!smooth || pts.length < 3) return pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
    d += ` C${(p1[0] + (p2[0] - p0[0]) / 6).toFixed(1)},${(p1[1] + (p2[1] - p0[1]) / 6).toFixed(1)} ${(p2[0] - (p3[0] - p1[0]) / 6).toFixed(1)},${(p2[1] - (p3[1] - p1[1]) / 6).toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
  }
  return d;
}
function lineChart(o) {
  const { series, xd = [0, 1], yd = [0, 1], xl = "", yl = "", diag = false, band = 0, smooth = false, w = 520, hgt = 300, dots = true, xt = 5, yt = 5,
    yfmt = (v) => v.toFixed(2), xfmt = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(2)) } = o;
  const m = { l: 54, r: 18, t: 14, b: 46 }, X = (v) => m.l + ((v - xd[0]) / (xd[1] - xd[0])) * (w - m.l - m.r), Y = (v) => hgt - m.b - ((v - yd[0]) / (yd[1] - yd[0])) * (hgt - m.t - m.b);
  const svg = s("svg", { viewBox: `0 0 ${w} ${hgt}`, width: "100%", role: "img", style: `max-width:${w}px` });
  for (const t of axisTicks(yd[0], yd[1], yt)) svg.append(s("line", { class: "grid-line", x1: m.l, x2: w - m.r, y1: Y(t), y2: Y(t) }), s("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end", class: "muted" }, yfmt(t)));
  for (const t of axisTicks(xd[0], xd[1], xt)) svg.append(s("text", { x: X(t), y: hgt - m.b + 18, "text-anchor": "middle", class: "muted" }, xfmt(t)));
  svg.append(s("text", { x: (m.l + w - m.r) / 2, y: hgt - 6, "text-anchor": "middle", class: "muted" }, xl), s("text", { x: 13, y: (m.t + hgt - m.b) / 2, "text-anchor": "middle", class: "muted", transform: `rotate(-90 13 ${(m.t + hgt - m.b) / 2})` }, yl));
  if (band) svg.append(s("polygon", { points: [[0, 0], [0, band], [1 - band, 1], [1, 1], [1, 1 - band], [band, 0]].map((p) => `${X(p[0])},${Y(p[1])}`).join(" "), fill: "rgba(16,163,127,.12)" }));
  if (diag) svg.append(s("line", { x1: X(xd[0]), y1: Y(yd[0]), x2: X(xd[1]), y2: Y(yd[1]), stroke: "#8b93b3", "stroke-dasharray": "5 5", "stroke-width": 1.4 }));
  for (const se of series) {
    const pts = se.points.filter((p) => p[1] != null).map((p) => [X(p[0]), Y(p[1])]);
    if (!pts.length) continue;
    const col = se.color || SOLID[se.key];
    if (se.area && pts.length > 1) svg.append(s("path", { class: "fadein", d: `${pathD(pts, smooth)} L${pts[pts.length - 1][0]},${Y(yd[0])} L${pts[0][0]},${Y(yd[0])}Z`, fill: `url(#a-${se.key})` }));
    svg.append(s("path", { class: "glow fadein", d: pathD(pts, smooth), fill: "none", stroke: col, "stroke-width": 2.8, "stroke-linecap": "round", "stroke-linejoin": "round", "stroke-dasharray": se.dash || "" }));
    if (dots) se.points.filter((p) => p[1] != null).forEach((p) => svg.append(s("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: "#fff", stroke: col, "stroke-width": 2.2 }, s("title", {}, `${xfmt(p[0])} → ${yfmt(p[1])}`))));
  }
  return svg;
}
function barChart(items, { w = 580, min = 0.4, max = 1, ref = null, label = "AUROC", lab = 160 } = {}) {
  const rowH = 40, m = { l: lab, r: 74, t: 6, b: 40 }, hgt = m.t + m.b + items.length * rowH;
  const X = (v) => m.l + ((v - min) / (max - min)) * (w - m.l - m.r), svg = s("svg", { viewBox: `0 0 ${w} ${hgt}`, width: "100%", role: "img", style: `max-width:${w}px` });
  for (const t of axisTicks(min, max, 5)) svg.append(s("line", { class: "grid-line", x1: X(t), x2: X(t), y1: m.t, y2: hgt - m.b + 4 }), s("text", { x: X(t), y: hgt - m.b + 20, "text-anchor": "middle", class: "muted" }, t.toFixed(2)));
  svg.append(s("text", { x: (m.l + w - m.r) / 2, y: hgt - 4, "text-anchor": "middle", class: "muted" }, label));
  if (ref != null) svg.append(s("line", { x1: X(ref), x2: X(ref), y1: m.t, y2: hgt - m.b + 4, stroke: "#8b93b3", "stroke-dasharray": "4 4", "stroke-width": 1.4 }), s("text", { x: X(ref) + 4, y: m.t + 8, class: "muted", style: "font-size:10px" }, "chance"));
  items.forEach((it, i) => {
    const y = m.t + i * rowH + 9;
    svg.append(s("text", { x: m.l - 12, y: y + 16, "text-anchor": "end", style: "font-weight:600" }, it.label), s("rect", { x: X(min), y, width: X(max) - X(min), height: 22, rx: 11, fill: "var(--track)" }));
    if (it.value == null) { svg.append(s("text", { x: m.l + 10, y: y + 16, class: "muted" }, "n/a")); return; }
    svg.append(s("rect", { class: "grow", x: X(min), y, width: Math.max(X(Math.max(it.value, min)) - X(min), 4), height: 22, rx: 11, fill: `url(#g-${it.key})`, style: `animation-delay:${i * 70}ms` }));
    if (it.lo != null) svg.append(s("g", { stroke: "rgba(20,25,54,.7)", "stroke-width": 1.6, class: "fadein" }, s("line", { x1: X(it.lo), x2: X(it.hi), y1: y + 11, y2: y + 11 }), s("line", { x1: X(it.lo), x2: X(it.lo), y1: y + 5, y2: y + 17 }), s("line", { x1: X(it.hi), x2: X(it.hi), y1: y + 5, y2: y + 17 })));
    svg.append(s("text", { x: w - m.r + 12, y: y + 16, style: "font-weight:750;font-size:13px" }, it.value.toFixed(3)));
  });
  return svg;
}
function hBars(entries, { w = 480, key = "acc", signed = false, lab = 124 } = {}) {
  const rowH = 26, m = { l: lab, r: 52, t: 4, b: 4 }, hgt = m.t + m.b + entries.length * rowH, mx = Math.max(...entries.map((e) => Math.abs(e[1])), 1e-9);
  const svg = s("svg", { viewBox: `0 0 ${w} ${hgt}`, width: "100%", style: `max-width:${w}px` });
  const half = (w - m.l - m.r) / 2, zero = signed ? m.l + half : m.l, scale = (signed ? half - 38 : w - m.l - m.r) / mx;
  entries.forEach(([k, v], i) => {
    const y = m.t + i * rowH + 4, len = Math.max(Math.abs(v) * scale, 2), x = v >= 0 ? zero : zero - len;
    svg.append(s("text", { x: m.l - 8, y: y + 12, "text-anchor": "end" }, k), s("rect", { class: "grow", x, y, width: len, height: 15, rx: 7.5, style: `transform-origin:${v >= 0 ? "left" : "right"} center;animation-delay:${i * 40}ms`, fill: `url(#g-${signed ? (v >= 0 ? "pos" : "neg") : key})` }),
      s("text", { x: v >= 0 ? zero + len + 6 : zero - len - 6, y: y + 12, "text-anchor": v >= 0 ? "start" : "end", class: "muted" }, v.toFixed(2)));
  });
  if (signed) svg.append(s("line", { x1: zero, x2: zero, y1: 0, y2: hgt, stroke: "#8b93b3" }));
  return svg;
}
function forest(rows, { w = 580 } = {}) {
  const m = { l: 222, r: 62, t: 14, b: 38 }, rowH = 44, hgt = m.t + m.b + rows.length * rowH;
  const lo = Math.min(...rows.map((r) => r.lo), 0) - 0.02, hi = Math.max(...rows.map((r) => r.hi), 0) + 0.02;
  const X = (v) => m.l + ((v - lo) / (hi - lo)) * (w - m.l - m.r), svg = s("svg", { viewBox: `0 0 ${w} ${hgt}`, width: "100%", style: `max-width:${w}px` });
  const ticks = []; for (let t = Math.ceil(lo / 0.05) * 0.05; t <= hi + 1e-9; t += 0.05) ticks.push(Math.round(t * 100) / 100);
  for (const t of ticks) svg.append(s("line", { class: "grid-line", x1: X(t), x2: X(t), y1: m.t, y2: hgt - m.b }), s("text", { x: X(t), y: hgt - m.b + 16, "text-anchor": "middle", class: "muted" }, (Math.abs(t) < 1e-9 ? "" : t > 0 ? "+" : "−") + Math.abs(t).toFixed(2)));
  svg.append(s("line", { x1: X(0), x2: X(0), y1: m.t, y2: hgt - m.b, stroke: "#141936", "stroke-width": 1.6 }), s("text", { x: (m.l + w - m.r) / 2, y: hgt - 3, "text-anchor": "middle", class: "muted" }, "Δ AUROC (right = first model better)"));
  rows.forEach((r, i) => {
    const y = m.t + i * rowH + rowH / 2, col = r.sig ? "#7c3aed" : "#8b93b3";
    svg.append(s("text", { x: m.l - 12, y: y + 4, "text-anchor": "end", style: "font-weight:600" }, r.label),
      s("g", { class: "fadein", stroke: col, "stroke-width": 3, "stroke-linecap": "round" }, s("line", { x1: X(r.lo), x2: X(r.hi), y1: y, y2: y })),
      s("path", { class: "fadein glow", d: `M${X(r.diff)},${y - 9} L${X(r.diff) + 9},${y} L${X(r.diff)},${y + 9} L${X(r.diff) - 9},${y}Z`, fill: r.sig ? "url(#g-acc)" : "#b4bad6", stroke: "#fff", "stroke-width": 2 }),
      s("text", { x: w - m.r + 12, y: y + 4, style: "font-weight:750" }, sgn(r.diff)));
  });
  return svg;
}
const legend = (keys) => h("div", { class: "legend" }, keys.map((k) => h("span", {}, chip(k), SHORT[k])));

/* ---------- sections ---------- */
function pipeline(cfg) {
  const svg = s("svg", { class: "pipeline", viewBox: "0 0 1000 318", role: "img", "aria-label": "Pipeline: three inputs, three encoders, fusion, outputs" });
  const box = (x, y, w, hh, fill, rx = 16) => s("rect", { x, y, width: w, height: hh, rx, fill });
  const txt = (x, y, t, st = "", anchor = "start") => s("text", { x, y, "text-anchor": anchor, style: st }, t);
  const ysC = [55, 155, 255], enc = [`CNN encoder`, `${cfg.tab_backend === "catboost" ? "CatBoost" : "XGBoost"} + MLP`, cfg.ts_arch === "lstm" ? "BiLSTM encoder" : "Temporal CNN"];
  const labels = [["Chest X-ray", "64×64 grayscale"], ["Clinical values", "12 labs & vitals"], ["Vital-sign series", "48 h · HR, SpO₂"]];
  [["INPUTS", 10], ["ENCODERS", 262], ["FUSION", 540], ["OUTPUTS", 836]].forEach(([t, x]) => svg.append(txt(x, 12, t, "font-size:10.5px;letter-spacing:.14em;font-weight:700;fill:#8b93b3")));
  ysC.forEach((yc, i) => { // connectors first, so boxes sit on top
    svg.append(s("path", { class: "flow", d: `M180,${yc} L250,${yc}`, stroke: "#b9bdf0", "stroke-width": 2.6, fill: "none" }),
      s("path", { class: "flow", d: `M440,${yc} C490,${yc} 480,155 530,155`, stroke: "#b9bdf0", "stroke-width": 2.6, fill: "none" }));
  });
  [38, 108, 178, 248].forEach((yy) => svg.append(s("path", { class: "flow", d: `M720,155 C765,155 755,${yy + 27} 800,${yy + 27}`, stroke: "#b9bdf0", "stroke-width": 2.6, fill: "none" })));
  ysC.forEach((yc, i) => {
    const k = ["A", "B", "C"][i], mod = MODS[i];
    svg.append(s("rect", { x: 10, y: yc - 36, width: 170, height: 72, rx: 18, fill: "#fff", stroke: "rgba(20,25,54,.1)" }));
    const g = s("g", { transform: `translate(24 ${yc - 14}) scale(1.17)`, fill: "none", stroke: SOLID[k], "stroke-width": 1.9, "stroke-linecap": "round", "stroke-linejoin": "round" }); g.innerHTML = ICONS[mod];
    svg.append(g, txt(62, yc - 4, labels[i][0], "font-weight:700;font-size:13px"), txt(62, yc + 14, labels[i][1], "font-size:10.5px;fill:#5b6384"));
    svg.append(s("rect", { x: 250, y: yc - 36, width: 190, height: 72, rx: 18, fill: `url(#g-${k})` }), txt(270, yc - 3, enc[i], "fill:#fff;font-weight:700;font-size:14px"), txt(270, yc + 16, "pretrained → fine-tuned", "fill:rgba(255,255,255,.85);font-size:10.5px"));
  });
  svg.append(s("rect", { x: 530, y: 30, width: 190, height: 250, rx: 22, fill: "url(#g-D2)" }), s("rect", { x: 530, y: 30, width: 190, height: 250, rx: 22, fill: "none", stroke: "rgba(255,255,255,.5)" }));
  svg.append(txt(625, 120, "Attention fusion", "fill:#fff;font-weight:750;font-size:16px", "middle"), txt(625, 142, "learned query over", "fill:rgba(255,255,255,.9);font-size:11.5px", "middle"), txt(625, 158, "modality tokens", "fill:rgba(255,255,255,.9);font-size:11.5px", "middle"),
    txt(625, 190, "missing-aware", "fill:#fff;font-weight:700;font-size:12px", "middle"), txt(625, 206, "+ modality dropout", "fill:rgba(255,255,255,.9);font-size:11px", "middle"), txt(625, 250, "also: late-fusion stacker", "fill:rgba(255,255,255,.8);font-size:10.5px", "middle"));
  [["Risk estimate", "calibrated probability"], ["Uncertainty", "MC dropout, logit σ"], ["Explanations", "Grad-CAM · SHAP · occlusion"], ["Missing-data report", "AUROC per modality subset"]].forEach(([a, b], i) => {
    const yy = 38 + i * 70;
    svg.append(s("rect", { x: 800, y: yy, width: 190, height: 54, rx: 15, fill: "#fff", stroke: "rgba(20,25,54,.1)" }), s("rect", { x: 800, y: yy + 12, width: 5, height: 30, rx: 2.5, fill: "url(#g-acc)" }), txt(818, yy + 23, a, "font-weight:700;font-size:13px"), txt(818, yy + 40, b, "font-size:10.5px;fill:#5b6384"));
  });
  return svg;
}
function renderOverview(r) {
  const cc = r.comparison.complete_cases, best = r.comparison.best_unimodal_by_validation, d = r.comparison.paired_differences?.[`D2_minus_${best}`], sp = r.meta.split;
  const f = cc.D2.auroc, b = cc[best].auroc, synth = r.meta.synthetic_data, el = $("s-overview");
  const hero = h("div", { class: "hero" }, h("i", { class: "blob b1" }), h("i", { class: "blob b2" }), h("i", { class: "blob b3" }),
    h("div", {}, h("span", { class: "eyebrow" }, icon("spark", 14), "Multimodal research platform"),
      h("h2", {}, "Does fusing medical data sources ", h("em", {}, "really"), " beat any single one?"),
      h("p", {}, "Image, tabular and vital-sign models are trained side by side, then compared with their fusion on the same held-out patients, with honest confidence intervals."),
      h("div", { class: "pills" }, h("span", { class: "pill" }, synth ? "Synthetic demo data" : "Real cohort"), h("span", { class: "pill" }, `${sp.test_complete_cases} complete test patients`),
        d ? h("span", { class: "pill solid" }, `Δ AUROC ${sgn(d.diff)} · 95% CI ${ci(d.ci)}`) : null)),
    h("div", { class: "hero-panel" }, heroRings(f, b), h("div", { class: "legend-row" },
      h("div", { class: "lg" }, h("b", {}, f3(f)), h("span", {}, h("i", { style: "background:#fde68a" }), "Attention fusion")),
      h("div", { class: "lg" }, h("b", {}, f3(b)), h("span", {}, h("i", { style: "background:#67e8f9" }), "Best single: " + r.experiments[best].name)),
      h("div", { class: "lg" }, h("b", {}, "0.500"), h("span", {}, h("i", { style: "background:rgba(255,255,255,.3)" }), "Chance level")))));
  const exps = h("div", { class: "exp-grid" }, EXP.map((k, i) => h("div", { class: "exp-card", style: `--g:${lg(k)};animation-delay:${i * 70}ms` },
    h("span", { class: "k" }, k), h("div", { class: "big" }, f3(cc[k].auroc)), h("div", { class: "sm" }, h("b", {}, r.experiments[k].name)), h("div", { class: "sm" }, `AUROC ${ci(cc[k].auroc_ci)}`))));
  const tot = sp.train + sp.val + sp.test, seg = (n, c, dly) => h("div", { style: `width:${(100 * n) / tot}%;background:${c};animation-delay:${dly}ms` });
  el.replaceChildren(hero,
    h("div", { class: "kpis" },
      kpi(f3(f), "Attention-fusion AUROC", { ico: "target", k: lg("D2") }), kpi(f3(b), "Best single modality: " + r.experiments[best].name, { ico: "layers", k: lg(MOD_KEY[r.experiments[best].modalities[0]] || "A") }),
      kpi(d ? sgn(d.diff) : "–", d ? "Fusion gain, 95% CI " + ci(d.ci) : "Fusion gain", { ico: "trend", k: lg("C") }), kpi(f3(cc.D2.ece), "Calibrated expected calibration error", { ico: "gauge", k: lg("B") }),
      kpi(`${sp.test_complete_cases}/${sp.test}`, "Complete / total held-out test patients", { ico: "users", k: lg("acc") })),
    card("How the platform works", pipeline(r.meta.config)),
    exps,
    card("What the experiments say", h("ul", { class: "findings" }, r.findings.map((x, i) => h("li", { class: synth && i === 0 ? "warn" : "" }, h("span", { class: "fi" }, icon(synth && i === 0 ? "alert" : "check", 16)), h("span", {}, x))))),
    card("Setup", h("div", { class: "splitbar" }, seg(sp.train, "linear-gradient(90deg,#4f46e5,#7c3aed)", 0), seg(sp.val, "linear-gradient(90deg,#f59e0b,#fbbf24)", 150), seg(sp.test, "linear-gradient(90deg,#10b981,#34d399)", 300)),
      h("div", { class: "legend" }, h("span", {}, h("span", { class: "chip", style: "background:#6d4af0" }), `Train ${sp.train}`), h("span", {}, h("span", { class: "chip", style: "background:#f59e0b" }), `Validation ${sp.val}`), h("span", {}, h("span", { class: "chip", style: "background:#10b981" }), `Held-out test ${sp.test}`)),
      h("p", { class: "muted" }, `Models are fit on train, early-stopped and calibrated on validation, and reported on test only. ${synth ? "Data is synthetic and generated to exercise the pipeline. " : ""}Run took ${r.meta.runtime_seconds}s.`)));
}
function renderCompare(r) {
  const c = r.comparison, el = $("s-compare");
  const rows = (set, keys) => keys.map((k) => { const x = set[k]; return [h("span", {}, chip(k), SHORT[k]), x.n, x.auroc == null ? "–" : `${f3(x.auroc)} ${ci(x.auroc_ci)}`, f3(x.auprc), f3(x.brier), `${f3(x.raw_ece)} → ${f3(x.ece)}`]; });
  const head = ["Experiment", "n", "AUROC [95% CI]", "AUPRC", "Brier (cal.)", "ECE raw → cal."];
  const forestRows = Object.entries(c.paired_differences || {}).map(([k, v]) => { const [a, bb] = k.split("_minus_"); return { label: `${SHORT[a] || a} − ${SHORT[bb] || bb}`, diff: v.diff, lo: v.ci[0], hi: v.ci[1], sig: v.ci_excludes_zero }; });
  el.replaceChildren(
    card("Single modality vs fusion: complete cases", h("p", { class: "muted" }, "Same held-out patients for every experiment (all three modalities present), so the comparison is like for like. Bars show AUROC with bootstrap 95% CIs."),
      h("div", { class: "grid two-wide" }, barChart(EXP.map((k) => ({ label: SHORT[k], key: k, value: c.complete_cases[k].auroc, lo: c.complete_cases[k].auroc_ci?.[0], hi: c.complete_cases[k].auroc_ci?.[1] })), { ref: 0.5, w: 470, lab: 128 }), table(head, rows(c.complete_cases, EXP), "nw"))),
    card("Paired AUROC differences", h("p", { class: "muted" }, "Same patients, paired bootstrap. A violet diamond means the 95% interval excludes zero."), h("div", { class: "grid two" }, forest(forestRows), h("div", {}, h("p", { class: "hint" }, `The best single modality (${r.experiments[c.best_unimodal_by_validation].name}) was chosen on the validation set, not the test set, to avoid selection bias. A fusion model that does not beat it is a valid finding.`)))),
    h("div", { class: "grid two" }, card("All available test patients", h("p", { class: "muted" }, "Unimodal models only score patients who have that modality, so their groups differ; fusion scores everyone."), table(head.slice(0, 4), rows(c.available_cases, EXP).map((x) => x.slice(0, 4)))),
      card("Patients with incomplete data", table(head.slice(0, 4), rows(c.incomplete_cases, ["D1", "D2"]).map((x) => x.slice(0, 4))))),
    card("Training curves (validation loss)", legend(["A", "C", "D2"]), lineChart({
      series: ["A", "C", "D2"].map((k) => ({ key: k, points: (r.training[k] || []).map((e) => [e.epoch, e.val_loss]), area: k === "D2" })),
      xd: [0, Math.ceil(Math.max(...["A", "C", "D2"].map((k) => (r.training[k] || []).length), 4) / 4) * 4], xt: 4, yd: [0, Math.ceil(Math.max(...["A", "C", "D2"].flatMap((k) => (r.training[k] || []).map((e) => e.val_loss)), 0.8) * 5) / 5], xl: "epoch", yl: "validation BCE", dots: false, w: 760, hgt: 280 })));
}
const MIN_BIN = 10; // reliability bins with fewer patients than this are too noisy to plot
function renderTrust(r) {
  const el = $("s-calib"), cc = r.comparison.complete_cases, u = r.uncertainty, d = r.disagreement;
  const sel = h("select", { id: "calMode" }, h("option", { value: "calibrated" }, "After Platt recalibration (fit on validation)"), h("option", { value: "raw" }, "Raw model probabilities"));
  const holder = h("div", {});
  const draw = () => holder.replaceChildren(legend(EXP), lineChart({ diag: true, band: 0.1, xl: "mean predicted probability", yl: "observed fraction positive", w: 540, hgt: 330,
    series: EXP.map((k) => ({ key: k, points: (r.calibration.complete_cases[k][sel.value] || []).filter((b) => b.n >= MIN_BIN).map((b) => [b.mean_pred, b.frac_pos]) })) }));
  sel.addEventListener("change", draw);
  const sp = u.selective_prediction, last = sp[sp.length - 1];
  el.replaceChildren(
    h("div", { class: "grid two" },
      card("Reliability diagram", h("p", { class: "muted" }, "A prediction of 0.85 is only trustworthy if about 85% of such patients are truly positive. The dashed line is perfect calibration; the green band is ±0.10. Bins with fewer than 10 patients are hidden."), sel, holder),
      card("Calibration metrics", table(["Experiment", "Brier raw → cal.", "ECE raw → cal."], EXP.map((k) => [h("span", {}, chip(k), SHORT[k]), `${f3(cc[k].raw_brier)} → ${f3(cc[k].brier)}`, `${f3(cc[k].raw_ece)} → ${f3(cc[k].ece)}`])),
        h("p", { class: "hint" }, "Expected calibration error uses 10 equal-width bins. On a small test set, ECE is itself noisy, so treat small differences with caution."))),
    card("Selective prediction", h("p", { class: "muted" }, u.method + ". Keep only the patients the model is most sure about; if uncertainty is informative, accuracy rises as coverage falls."),
      h("div", { class: "grid two" }, lineChart({ xd: [0.3, 1], yd: [0.5, 1], xl: "coverage (fraction kept, most certain first)", yl: "accuracy", smooth: true, w: 540, hgt: 290, xt: 7,
        series: [{ key: "D2", area: true, points: sp.map((x) => [x.coverage, x.accuracy]) }], xfmt: (v) => `${Math.round(v * 100)}%` }),
        h("div", {}, table(["Coverage", "n", "Accuracy", "AUROC"], sp.map((x) => [pct(x.coverage), x.n, pct(x.accuracy), f3(x.auroc)])), h("p", { class: "hint" }, "AUROC can fall as coverage shrinks even when uncertainty is useful: the retained patients are mostly confident negatives, so there is little left to rank. Accuracy is the cleaner read.")))),
    card("Does uncertainty flag mistakes?", h("div", { class: "rings" },
      ringBox({ value: ((u.error_detection_auroc || 0.5) - 0.5) / 0.5, key: "D2", text: f3(u.error_detection_auroc), sub: "AUROC", size: 150 }, "Error-detection AUROC (0.5 = useless)"),
      ringBox({ value: last.accuracy, key: "C", text: pct(last.accuracy), sub: `top ${Math.round(last.coverage * 100)}%`, size: 150 }, `Accuracy on the ${Math.round(last.coverage * 100)}% most certain patients`),
      ringBox({ value: sp[0].accuracy, key: "A", text: pct(sp[0].accuracy), sub: "all patients", size: 150 }, "Accuracy with no filtering"),
      h("div", { class: "kpis flat", style: "grid-template-columns:1fr;min-width:200px;align-self:center;margin:0" }, kpi(f3(u.mean_std_when_correct), "Mean logit σ when correct"), kpi(f3(u.mean_std_when_wrong), "Mean logit σ when wrong"))),
      h("p", { class: "hint" }, "MC-dropout spread approximates model uncertainty. It is not a calibrated confidence interval.")),
    card("Disagreement between single-modality models", h("p", { class: "muted" }, d.definition), d.low_disagreement_half ? table(["Group", "n", "AUROC", "Accuracy"], [["Low-disagreement half", d.low_disagreement_half.n, f3(d.low_disagreement_half.auroc), pct(d.low_disagreement_half.accuracy)], ["High-disagreement half", d.high_disagreement_half.n, f3(d.high_disagreement_half.auroc), pct(d.high_disagreement_half.accuracy)]]) : h("p", {}, "Not enough complete cases.")));
  draw();
}
const heat = (w) => `rgba(124,58,237,${Math.min(0.08 + w * 1.6, 0.92).toFixed(2)})`;
function renderMissing(r) {
  const el = $("s-missing"), sc = r.missing_modality.scenarios_on_complete_cases, names = Object.keys(sc), allV = sc.all.D2.auroc, byN = r.missing_modality.by_number_of_available_modalities;
  const cards = names.map((n, i) => {
    const v = sc[n], drop = allV - v.D2.auroc, t = Math.min(1, Math.max(0, drop / 0.2)), hue = 150 - 150 * t;
    return h("div", { class: "scen", style: `--s:linear-gradient(90deg,hsl(${hue} 70% 46%),hsl(${hue + 22} 85% 58%));animation-delay:${i * 60}ms` },
      h("div", { class: "mods" }, MODS.map((m) => { const on = v.modalities.includes(m), k = MOD_KEY[m]; return h("span", { class: "m" + (on ? " on" : ""), style: `--g:${lg(k)};--g1:${GRAD[k][0]}`, title: m }, icon(m, 18)); })),
      h("div", { class: "nm" }, n.replace(/_/g, " ")), h("div", { class: "auc" }, f3(v.D2.auroc)),
      h("div", { class: "delta", style: `color:hsl(${hue} 60% 38%)` }, n === "all" ? "all modalities" : `${sgn(-drop)} vs all`),
      h("div", { class: "meter" }, h("div", { style: `width:${Math.max(4, 100 * (v.D2.auroc - 0.5) / (allV - 0.5))}%;animation-delay:${i * 60 + 200}ms` })),
      h("div", { class: "sub" }, h("span", {}, "D1 late fusion"), h("b", {}, f3(v.D1.auroc))));
  });
  const attn = h("div", { class: "scroll" }, h("table", {}, h("thead", {}, h("tr", {}, ["Scenario", "Image", "Tabular", "Time-series", "None-token"].map((x) => h("th", {}, x)))),
    h("tbody", {}, names.map((n) => h("tr", {}, h("td", {}, n.replace(/_/g, " ")), sc[n].D2_attention.map((w) => h("td", { class: "heat", style: `background:${heat(w)};color:${w > 0.3 ? "#fff" : "#141936"}` }, w.toFixed(2))))))));
  el.replaceChildren(
    card("What happens when a modality is taken away?", h("p", { class: "muted" }, "Complete test patients, with modalities hidden at inference time. The attention model trained with random modality dropout; the late-fusion model zeroes the missing logit and sets its availability flag. Lit icons are the modalities still present; the card colour shows the AUROC lost."), h("div", { class: "scen-grid" }, cards)),
    h("div", { class: "grid two" },
      card("Attention-model AUROC with 95% CI", barChart(names.map((n) => ({ label: n.replace(/_/g, " "), key: "D2", value: sc[n].D2.auroc, lo: sc[n].D2.auroc_ci?.[0], hi: sc[n].D2.auroc_ci?.[1] })), { ref: 0.5, w: 520, lab: 130 })),
      card("Where does the attention model look?", attn, h("p", { class: "hint" }, "Attention weights show how the model combines inputs, not which input is medically important. Masked modalities necessarily receive weight 0."))),
    card("Patients by how many modalities they naturally have", table(["Modalities available", "D1 AUROC", "D2 AUROC", "n"], Object.keys(byN).map((n) => [n, f3(byN[n].D1?.auroc), f3(byN[n].D2?.auroc), byN[n].D2?.n ?? 0]))));
}
function patientCard(p, i) {
  const ts = p.timeseries, T = ts.importance.length, w = 340, hgt = 130, mx = Math.max(...ts.importance.map(Math.abs), 1e-9), svg = s("svg", { viewBox: `0 0 ${w} ${hgt}`, width: "100%" });
  svg.append(s("rect", { x: 0, y: 0, width: w, height: hgt, rx: 12, fill: "#f7f8ff" }));
  ts.importance.forEach((v, j) => svg.append(s("rect", { x: (j * w) / T, y: 0, width: w / T + 0.5, height: hgt, fill: v >= 0 ? "#e11d48" : "#2563eb", opacity: Math.min(Math.abs(v) / mx, 1) * 0.28 })));
  ts.channels.forEach((c, k) => {
    const vals = ts.values[k], ok = vals.filter((x) => x != null), lo = Math.min(...ok), hi = Math.max(...ok), segs = []; let seg = [];
    vals.forEach((v, j) => { if (v == null) { if (seg.length) segs.push(seg), (seg = []); } else seg.push([(j * w) / T + w / T / 2, hgt - 12 - ((v - lo) / (hi - lo + 1e-9)) * (hgt - 24)]); });
    if (seg.length) segs.push(seg);
    segs.forEach((sg) => svg.append(s("path", { d: pathD(sg, true), fill: "none", stroke: k === 0 ? SOLID.B : SOLID.C, "stroke-width": 2.2, "stroke-linecap": "round", class: "glow" })));
  });
  const probs = EXP.filter((k) => p.probability[k] != null).map((k) => h("span", { class: "risk" }, chip(k), p.probability[k].toFixed(2)));
  const tagc = p.role.includes("false") ? "warn" : "";
  return h("div", { class: "pcard", style: `animation-delay:${i * 70}ms` },
    h("div", { class: "cap" }, h("b", {}, p.patient_id), h("span", { class: "tag " + tagc }, p.role.replace(/ \(.*\)/, "")), h("span", { class: "tag" }, "label " + p.y_true)),
    h("div", { class: "imgs" }, h("figure", {}, h("img", { src: p.image, alt: "synthetic chest X-ray" }), h("figcaption", {}, "X-ray")), h("figure", {}, h("img", { src: p.image_gradcam, alt: "Grad-CAM overlay" }), h("figcaption", {}, "Grad-CAM (model attention)"))),
    h("h3", {}, "Predicted risk (calibrated)"), h("div", { class: "risk-row" }, probs),
    h("p", { class: "muted" }, `MC-dropout logit σ ${p.uncertainty_logit_std.toFixed(2)} · unimodal disagreement ${p.disagreement == null ? "–" : p.disagreement.toFixed(2)}`),
    h("h3", {}, "Tabular contributions (log-odds)"), hBars(p.tabular_top_contributions.map((c) => [`${c.feature}${c.value == null ? " (missing)" : "=" + +c.value.toFixed(1)}`, c.contribution_logodds]), { w: 340, signed: true, lab: 130 }),
    h("h3", {}, "Vitals · shading = occlusion importance"), svg, h("div", { class: "legend", style: "margin-top:6px" }, h("span", {}, h("span", { class: "chip", style: `background:${SOLID.B}` }), "heart rate"), h("span", {}, h("span", { class: "chip", style: `background:${SOLID.C}` }), "SpO₂")));
}
function renderExplain(r) {
  const g = r.explainability.global, el = $("s-explain"), loc = g.image_localisation_vs_synthetic_lesion, tsimp = g.timeseries_importance_by_hour, att = g.mean_attention_full_availability;
  const lo = tsimp ? Math.min(0, ...(tsimp.negative_class || []), ...(tsimp.positive_class || [])) : 0, hi = tsimp ? Math.max(...(tsimp.positive_class || [0]), ...(tsimp.negative_class || [0]), 0.01) : 1;
  el.replaceChildren(
    h("div", { class: "hint", style: "margin:0 0 18px" }, g.caveat + " Heatmaps and importances can highlight shortcuts and artefacts as readily as real signal."),
    h("div", { class: "grid two" },
      card("Tabular: mean |contribution| (SHAP-style)", hBars(Object.entries(g.tabular_mean_abs_contribution || {}), { key: "B", lab: 110 })),
      card("Time-series: importance by hour", h("p", { class: "muted" }, "Mean drop in logit when a 6-hour window is blanked (0 = oldest hour, 47 = most recent)."),
        tsimp ? [h("div", { class: "legend" }, h("span", {}, h("span", { class: "chip", style: `background:${SOLID.D2}` }), "positive patients"), h("span", {}, h("span", { class: "chip", style: `background:${SOLID.A}` }), "negative patients")),
          lineChart({ xd: [0, 48], yd: [Math.floor(lo * 20) / 20, Math.ceil(hi * 20) / 20], xl: "hour", yl: "importance", smooth: true, dots: false, w: 520, hgt: 280, xt: 6,
            series: [{ key: "A", points: (tsimp.negative_class || []).map((v, i) => [i, v]) }, { key: "D2", area: true, points: (tsimp.positive_class || []).map((v, i) => [i, v]) }] })] : "–"),
      card("Fusion attention (all modalities present)", hBars(g.attention_order.map((k, i) => [k, att[i]]), { key: "acc", lab: 90 })),
      card("Image localisation sanity check", loc ? [h("div", { class: "rings" },
        ringBox({ value: loc.pointing_game_hit_rate, key: "D2", text: pct(loc.pointing_game_hit_rate), size: 130 }, "Grad-CAM peak inside true lesion"),
        ringBox({ value: loc.cam_mass_in_lesion, key: "A", text: pct(loc.cam_mass_in_lesion), size: 130 }, "CAM mass inside lesion"),
        ringBox({ value: loc.lesion_area_fraction, key: "C", text: pct(loc.lesion_area_fraction), size: 130 }, "Lesion share of image (chance)")),
        h("p", { class: "hint" }, `Only possible on synthetic data, where the lesion location is known (n=${loc.n} positive test X-rays). Real data has no such ground truth.`)] : h("p", { class: "muted" }, "No ground-truth lesion masks for this cohort."))),
    h("h2", { style: "margin:6px 0 14px" }, "Individual patients"), h("div", { class: "patients" }, r.explainability.local.map(patientCard)));
}
function renderPredict(r) {
  const el = $("s-predict"), fields = {}, local = r.explainability.local;
  const grid = h("div", { class: "field-grid" }, TAB_FEATURES.map((f) => h("label", {}, f, (fields[f] = h("input", { type: "number", step: "any", placeholder: "missing" }))))), out = h("div", { class: "result" });
  const tsSel = h("select", {}, h("option", { value: "" }, "no time-series"), h("option", { value: "stable" }, "stable vitals"), h("option", { value: "deteriorating" }, "deteriorating vitals (last 16 h)"));
  const imgSel = h("select", {}, h("option", { value: "" }, "no image"), local.map((p, i) => h("option", { value: i }, `X-ray of ${p.patient_id} (${p.role})`)));
  const examples = { "typical": [60, 0, 27, 125, 75, 80, 1.0, 1.2, 8, 13, 140, 105], "high-risk profile": [74, 1, 29, 95, 55, 112, 2.6, 4.5, 16, 10.5, 137, 150] };
  const fill = (name) => TAB_FEATURES.forEach((f, i) => (fields[f].value = examples[name][i]));
  const series = (kind) => { const hr = [], sp = []; for (let t = 0; t < 48; t++) { const ramp = kind === "deteriorating" ? Math.max(0, (t - 32) / 16) : 0; hr.push(+(78 + 14.4 * ramp + Math.sin(t) * 1.5).toFixed(1)); sp.push(+(96.5 - 2.4 * ramp + Math.cos(t) * 0.4).toFixed(1)); } return { heart_rate: hr, spo2: sp }; };
  const loadImg = (i) => new Promise((res) => { const im = new Image(); im.onload = () => { const c = document.createElement("canvas"); c.width = c.height = 64; const x = c.getContext("2d"); x.drawImage(im, 0, 0, 64, 64); const d = x.getImageData(0, 0, 64, 64).data, rows = []; for (let y = 0; y < 64; y++) { const row = []; for (let xx = 0; xx < 64; xx++) row.push(d[(y * 64 + xx) * 4] / 255); rows.push(row); } res(rows); }; im.src = local[i].image; });
  const go = h("button", { class: "primary" }, "Estimate risk");
  go.addEventListener("click", async () => {
    const body = { run_id: state.runId, mc_samples: 30 }, tab = {};
    TAB_FEATURES.forEach((f) => { if (fields[f].value !== "") tab[f] = Number(fields[f].value); });
    if (Object.keys(tab).length) body.tabular = tab;
    if (tsSel.value) body.timeseries = series(tsSel.value);
    if (imgSel.value !== "") body.image = await loadImg(Number(imgSel.value));
    go.disabled = true;
    try { out.replaceChildren(renderPrediction(await api("/predict", { method: "POST", body: JSON.stringify(body) }))); } catch (e) { out.replaceChildren(h("p", { class: "bad" }, "Could not predict: " + e.message)); }
    go.disabled = false;
  });
  el.replaceChildren(card("Try a patient with any subset of modalities", h("p", { class: "muted" }, "Leave fields blank to treat them as missing. Remove whole modalities to see how the estimate and its uncertainty move. Research demo only: not for patient care."),
    h("div", { class: "row", style: "justify-content:flex-start;margin-bottom:12px" }, Object.keys(examples).map((n) => h("button", { onclick: () => fill(n) }, "Fill: " + n)), h("button", { class: "ghost", onclick: () => TAB_FEATURES.forEach((f) => (fields[f].value = "")) }, "Clear")),
    grid, h("div", { class: "grid two", style: "margin:14px 0" }, h("label", { class: "muted" }, "Vitals ", tsSel), h("label", { class: "muted" }, "Image ", imgSel)), go, out));
}
function renderPrediction(p) {
  const u = p.uncertainty;
  return h("div", { class: "grid two" }, h("div", {}, h("div", { class: "bigp" }, pct(p.risk.fusion_attention)),
    h("p", { class: "muted" }, `MC-dropout 95% range ${pct(u.interval_95[0])}–${pct(u.interval_95[1])} · logit σ ${u.mc_logit_std.toFixed(2)}`),
    h("p", { class: "muted" }, "Modalities used: " + p.modalities_used.join(", ") + ` · late-fusion estimate ${pct(p.risk.fusion_late)}`),
    p.disagreement != null ? h("p", { class: "muted" }, `Disagreement between single-modality models: ${p.disagreement.toFixed(2)}`) : null,
    p.warnings.map((w) => h("p", { class: "bad" }, "⚠ " + w)), h("p", { class: "hint" }, p.disclaimer)),
    h("div", {}, Object.keys(p.unimodal).length ? [h("h3", {}, "Single-modality estimates"), hBars(Object.entries(p.unimodal), { w: 420, key: "B", lab: 90 })] : null, h("h3", {}, "Attention"), hBars(Object.entries(p.attention), { w: 420, lab: 90 })));
}
function renderData(r) {
  const v = r.validation, el = $("s-data"), av = v.modality_availability;
  el.replaceChildren(
    card("Validation summary", h("p", {}, v.ok ? h("span", { class: "good" }, "✓ No issues found") : h("span", { class: "bad" }, "Issues found")), v.issues.length ? h("ul", {}, v.issues.map((x) => h("li", {}, x))) : null,
      h("div", { class: "rings", style: "margin-top:10px" },
        ...MODS.map((m) => ringBox({ value: av[m].pct / 100, key: MOD_KEY[m], text: av[m].pct.toFixed(0) + "%", size: 130 }, `${m} available (${av[m].n})`)),
        ringBox({ value: v.complete_cases_pct / 100, key: "acc", text: v.complete_cases_pct.toFixed(0) + "%", size: 130 }, `complete cases (${v.complete_cases})`),
        ringBox({ value: v.prevalence, key: "D2", text: pct(v.prevalence), size: 130 }, "outcome prevalence"))),
    h("div", { class: "grid two" },
      card("Completeness patterns", table(["Pattern", "Patients"], Object.entries(v.completeness_patterns))),
      card("Missing values within available data", h("h3", {}, "Tabular (% missing)"), hBars(Object.entries(v.tabular_missing_pct), { w: 440, key: "B", lab: 120 }), h("h3", {}, "Time-series (% missing hours)"), table(["Channel", "% missing"], Object.entries(v.timeseries_missing_pct)))));
}

/* ---------- API + run management ---------- */
async function api(path, opts = {}) {
  const key = sessionStorage.getItem("cf_key");
  const r = await fetch("/api" + path, { ...opts, headers: { "Content-Type": "application/json", ...(key ? { "X-API-Key": key } : {}) } });
  if (r.status === 401) { const k = prompt("This server requires an API key:"); if (k) { sessionStorage.setItem("cf_key", k); return api(path, opts); } }
  if (!r.ok) { let m = r.statusText; try { const j = await r.json(); m = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (e) {} throw new Error(m); }
  return r.status === 204 ? null : r.json();
}
function showReport(r) {
  state.res = r; $("report").hidden = false; $("empty").hidden = true;
  renderOverview(r); renderCompare(r); renderTrust(r); renderMissing(r); renderExplain(r); renderPredict(r); renderData(r);
}
function setupTabs() {
  const secs = [...document.querySelectorAll("section[data-tab]")], nav = $("tabs"); nav.replaceChildren();
  const show = (name) => { secs.forEach((x) => x.classList.toggle("on", x.dataset.tab === name)); [...nav.children].forEach((b) => b.setAttribute("aria-selected", b.textContent === name)); history.replaceState(null, "", "#" + encodeURIComponent(name)); };
  secs.forEach((x) => nav.append(h("button", { role: "tab", onclick: () => show(x.dataset.tab) }, x.dataset.tab)));
  const want = decodeURIComponent(location.hash.slice(1)); show(secs.some((x) => x.dataset.tab === want) ? want : "Overview");
}
async function loadRun(id) {
  state.runId = id; clearTimeout(state.poll);
  const run = await api("/runs/" + id), st = $("status");
  if (run.status === "done") { st.hidden = true; showReport(await api(`/runs/${id}/results`)); return; }
  $("report").hidden = true; st.hidden = false;
  if (run.status === "failed") { st.replaceChildren(h("h2", { class: "bad" }, "Run failed"), h("p", {}, run.error || "unknown error")); return; }
  st.replaceChildren(h("h2", {}, "Run in progress"), h("div", { class: "progress" }, h("div", { style: `width:${Math.round(run.progress * 100)}%` })), h("p", { class: "muted" }, `${run.message} (${Math.round(run.progress * 100)}%)`));
  state.poll = setTimeout(() => loadRun(id).then(refreshRuns), 1500);
}
async function refreshRuns() {
  state.runs = await api("/runs"); const sel = $("runSelect");
  sel.replaceChildren(...state.runs.map((r) => h("option", { value: r.id }, `${r.id} · ${r.status}${r.config.n_patients ? " · n=" + r.config.n_patients : ""}`)));
  if (state.runId) sel.value = state.runId;
  $("empty").hidden = state.runs.length > 0;
}
async function init() {
  installDefs(); setupTabs();
  $("runSelect").addEventListener("change", (e) => loadRun(e.target.value));
  const open = () => $("newRunDialog").showModal();
  $("newRunBtn").addEventListener("click", open); $("emptyStart").addEventListener("click", open);
  $("newRunForm").addEventListener("submit", async (e) => {
    if (e.submitter?.value !== "go") return;
    const f = Object.fromEntries(new FormData(e.target)), body = { preset: f.preset };
    for (const k of ["n_patients", "seed"]) if (f[k] !== "") body[k] = Number(f[k]);
    if (f.ts_arch) body.ts_arch = f.ts_arch;
    try { const r = await api("/runs", { method: "POST", body: JSON.stringify(body) }); await refreshRuns(); await loadRun(r.id); } catch (err) { alert("Could not start run: " + err.message); }
  });
  try {
    await refreshRuns();
    const pick = state.runs.find((r) => r.status === "running") || state.runs.find((r) => r.status === "done") || state.runs[0];
    if (pick) await loadRun(pick.id);
  } catch (e) { $("status").hidden = false; $("status").replaceChildren(h("p", { class: "bad" }, "Cannot reach the API: " + e.message)); }
}
init();
