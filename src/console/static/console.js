"use strict";
const $ = (s, r = document) => r.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (v, d = 1) => (v === null || v === undefined ? "–" : num(v * 100, d) + "%");
const words = (s) => { const t = String(s ?? "").replace(/^view_/, "").replace(/_/g, " "); return t.charAt(0).toUpperCase() + t.slice(1); };
const clock = (ts) => new Date(ts * 1000).toLocaleTimeString("en-GB");
const api = async (path) => { const r = await fetch("/api/" + path); if (!r.ok) throw new Error(r.status); return r.json(); };
const LABEL = { pass: "PASS", fail: "REJECT", not_inspected: "NOT INSPECTED", incomplete: "INCOMPLETE", missing: "NO IMAGE" };
const TABS = [["live", "Live"], ["trace", "Trace a part"], ["process", "Process control"], ["quality", "Quality"], ["audit", "Audit trail"]];

const S = { tab: "live", state: null, parts: [], part: null, key: null, heat: true, signal: "hold_pressure.active_median", failedOnly: true, busy: false };

/* ---------------------------------------------------------------- charts */
const tip = $("#tip");
function showTip(e, html) { tip.innerHTML = html; tip.style.opacity = 1; const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.max(8, Math.min(innerWidth - w - 8, e.clientX + 14)) + "px"; tip.style.top = Math.max(8, Math.min(innerHeight - h - 8, e.clientY + 14)) + "px"; }
const hideTip = () => { tip.style.opacity = 0; };
const registry = new Map(); let chartId = 0;
function nice(lo, hi, n = 4) { const span = (hi - lo) || 1, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw)), f = raw / mag, step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.abs(v) < step / 1e6 ? 0 : v); return { ticks: out, d: Math.max(0, -Math.floor(Math.log10(step))) }; }
/* One line chart: points, optional band, reference lines and marked points, with a crosshair. */
function line(o) {
  const W = o.w || 900, H = o.h || 240, m = { l: 46, r: o.right || 92, t: 10, b: 24 };
  const ys = [...o.y, ...(o.lo || []), ...(o.hi || []), ...(o.refs || []).map((r) => r.v)].filter(Number.isFinite);
  if (!ys.length) return '<div class="empty">No data yet.</div>';
  let lo = o.min ?? Math.min(...ys), hi = o.max ?? Math.max(...ys); const pad = (hi - lo || 1) * 0.08; if (o.min === undefined) lo -= pad; if (o.max === undefined) hi += pad;
  const n = o.y.length, X = (i) => m.l + (W - m.l - m.r) * (n < 2 ? 0 : i / (n - 1)), Y = (v) => m.t + (H - m.t - m.b) * (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo || 1));
  const path = (a) => a.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  const t = nice(lo, hi); let s = "";
  for (const v of t.ticks) s += `<line class="gridline" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${o.fmt ? o.fmt(v) : num(v, t.d)}</text>`;
  if (o.lo) s += `<path class="bandf" d="${path(o.hi)}${o.lo.map((v, i) => [v, i]).reverse().map(([v, i]) => `L${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("")}Z"/>`;
  if (o.mid) s += `<path class="mid" d="${path(o.mid)}"/>`;
  let last = -99; for (const r of [...(o.refs || [])].sort((a, b) => Y(a.v) - Y(b.v))) { const y = Y(r.v); s += `<line class="${r.cls || "ref"}" x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}"/>`; const ly = Math.max(y + 4, last + 13); last = ly; s += `<text x="${W - m.r + 8}" y="${ly}">${esc(r.label)}</text>`; }
  s += `<path class="l1" d="${path(o.y)}"/>`;
  (o.marks || []).forEach((i) => { s += `<circle class="dot bad" r="4.5" cx="${X(i)}" cy="${Y(o.y[i])}"/>`; });
  (o.x || []).forEach(([i, label]) => { s += `<text x="${X(i)}" y="${H - 4}" text-anchor="middle">${esc(label)}</text>`; });
  const id = "ch" + ++chartId; registry.set(id, { o, X, W, H, m, n, Y });
  return `<svg data-chart="${id}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria || "Line chart")}">${s}<g class="hover"></g></svg>`;
}
document.addEventListener("pointermove", (e) => {
  const t = e.target.closest?.("[data-tip]"); if (t) return showTip(e, t.dataset.tip);
  const svg = e.target.closest?.("svg[data-chart]"); if (!svg) return hideTip();
  const c = registry.get(svg.dataset.chart); if (!c || c.n < 2) return;
  const box = svg.getBoundingClientRect(), px = (e.clientX - box.left) * c.W / box.width;
  const i = Math.max(0, Math.min(c.n - 1, Math.round((px - c.m.l) / (c.W - c.m.l - c.m.r) * (c.n - 1))));
  $(".hover", svg).innerHTML = `<line class="cross" x1="${c.X(i)}" x2="${c.X(i)}" y1="${c.m.t}" y2="${c.H - c.m.b}"/><circle class="dot" r="5" cx="${c.X(i)}" cy="${c.Y(c.o.y[i])}"/>`;
  svg.dataset.i = i; showTip(e, c.o.tip ? c.o.tip(i) : num(c.o.y[i], 2));
});
document.addEventListener("click", (e) => { const svg = e.target.closest?.("svg[data-chart]"); const c = svg && registry.get(svg.dataset.chart); if (c?.o.pick && svg.dataset.i !== undefined) c.o.pick(+svg.dataset.i); });
const spark = (c) => line({ y: c.history, h: 120, w: 360, right: 46, min: Math.min(-2, ...c.history), max: Math.max(c.act + 1, ...c.history), refs: [{ v: c.watch, label: "watch", cls: "warnl" }, { v: c.act, label: "act", cls: "limit" }], aria: `${c.component} wear index`, tip: (i) => `<b>${esc(c.component)}</b>${num(c.history[i], 1)}σ from healthy` });

/* ---------------------------------------------------------------- pieces */
const partName = (p) => (p.number !== null && p.number !== undefined ? "Part " + p.number : p.cycle_id ? esc(p.cycle_id) : "Unlinked part");
function strip(parts, n = 80) {
  return `<div class="strip">${parts.slice(-n).map((p) => `<button class="${esc(p.verdict)} ${p.key && p.key === S.key ? "sel" : ""}" data-part="${esc(p.key)}" aria-label="${partName(p)} ${LABEL[p.verdict] || ""}" data-tip="<b>${partName(p)}</b>${LABEL[p.verdict] || ""}${p.defects.length ? " · " + esc(p.defects.map(words).join(", ")) : ""}<br>${clock(p.ts)}"></button>`).join("")}</div>`;
}
const partKey = `<div class="key"><span><i style="background:var(--good)"></i>Pass</span><span><i style="background:var(--crit)"></i>Reject</span><span><i style="background:var(--warn)"></i>Not fully inspected</span></div>`;
function viewTiles(d) {
  return `<div class="views">${d.views.map((v) => { const src = S.heat && v.heat ? v.heat : v.image;
    return `<div class="view ${esc(v.verdict)}">${src ? `<img alt="${esc(words(v.camera))} view" src="${esc(src)}">` : ""}<span class="tag">${LABEL[v.verdict] || esc(v.verdict)}</span><div class="cap"><b>${esc(words(v.camera))}</b><span data-tip="Anomaly score. 1.00 is the calibrated reject limit.">${v.score === null ? "–" : num(v.score, 2)}</span></div></div>`; }).join("")}</div>`;
}
const verdictHead = (d) => `<div class="verdict"><span class="big ${d.verdict === "pass" ? "pass" : d.verdict === "fail" ? "fail" : "other"}">${LABEL[d.verdict] || "–"}</span><span>Part ${d.sequence}${d.defects.length ? " · " + esc(d.defects.map(words).join(", ")) : ""}</span><span class="muted">${clock(d.ts)}</span></div>`;
const heatBtn = (d) => (d.views.some((v) => v.heat) ? `<span class="seg"><button data-heat="1" aria-pressed="${S.heat}">Heatmap</button><button data-heat="0" aria-pressed="${!S.heat}">Image</button></span>` : "");
function alarms(list) {
  return list.length ? list.map((a) => `<div class="alarm ${a.level}"><span class="ic" aria-hidden="true">${a.level === "critical" ? "■" : "▲"}</span><div><b>${esc(a.title)}</b><p>${esc(a.text)}</p></div></div>`).join("") : '<div class="empty">No active alarms.</div>';
}
function healthCards(list) {
  if (!list.length) return '<div class="empty">Learning the healthy baseline: needs 20 machine cycles.</div>';
  return `<div class="hl">${list.map((c) => `<div class="comp"><div class="n">${esc(c.component)} <span class="chip ${c.status === "OK" ? "good" : c.status === "WATCH" ? "warn" : "crit"}">${c.status === "OK" ? "Healthy" : c.status === "WATCH" ? "Watch" : "Act"}</span></div><div class="h">${num(c.health, 0)}<small> / 100</small></div>${spark(c)}<div class="muted">${esc(c.signal)}${c.status === "WATCH" && c.cycles_to_act ? ` · about ${c.cycles_to_act} cycles to act` : ""}</div></div>`).join("")}</div>`;
}

/* ---------------------------------------------------------------- views */
function liveView() {
  const s = S.state, k = s.kpi, d = S.part;
  const model = s.model || {};
  return `<div class="grid">
  <div class="kpis c12">
    <div class="kpi"><div class="l">Parts inspected</div><div class="v">${num(k.parts, 0)}</div><div class="s">last 8 hours</div></div>
    <div class="kpi"><div class="l">Yield</div><div class="v">${pct(k.yield)}</div><div class="s">last 30 parts: ${pct(k.recent_yield)}</div></div>
    <div class="kpi"><div class="l">Rejects</div><div class="v">${num(k.fail, 0)}</div><div class="s">worst view decides the part</div></div>
    <div class="kpi"><div class="l">Not fully inspected</div><div class="v">${num(k.not_inspected + k.incomplete, 0)}</div><div class="s">never counted as a pass</div></div>
    <div class="kpi"><div class="l">Linked to a machine cycle</div><div class="v">${pct(k.linked, 1)}</div><div class="s">part to cycle</div></div>
    <div class="kpi"><div class="l">Throughput</div><div class="v">${num(k.rate_per_min, 1)}<small> /min</small></div><div class="s">last 10 minutes</div></div>
  </div>
  <section class="card c8"><h2>Latest part, every face ${d ? `<span class="end">${heatBtn(d)}</span>` : ""}</h2>${d ? verdictHead(d) + viewTiles(d) : '<div class="empty">Waiting for the first part.</div>'}</section>
  <div class="c4" style="display:grid;gap:14px;align-content:start">
    <section class="card"><h2>Alarms <span class="end">${s.alarms.length || ""}</span></h2>${alarms(s.alarms)}</section>
    <section class="card"><h2>Station</h2><div class="rows">
      <div class="row"><span>Cameras</span><span>${s.cameras.filter((c) => c.online).length} of ${s.cameras.length} delivering</span></div>
      <div class="row"><span>PLC output</span><span>${!s.plc.online ? "–" : s.plc.hold_reason ? "Heartbeat held" : s.plc.simulated ? "Heartbeat running (simulated PLC)" : "Heartbeat running"}</span></div>
      <div class="row"><span>Last verdict sent</span><span>${s.plc.last_verdict ? `part ${s.plc.last_verdict.part}, flag ${s.plc.last_verdict.flag}` : "–"}</span></div>
      <div class="row"><span>Audit trail</span><span>${s.audit.ok === null ? (s.audit.checking ? "Checking…" : "–") : s.audit.ok ? `Intact, ${num(s.audit.records, 0)} records` : "Broken"}</span></div>
      <div class="row"><span>False-reject limit</span><span>${model.part_frr_target ? `${pct(model.part_frr_target, 0)} per part across ${model.n_views} views${model.guaranteed ? "" : " (not guaranteed)"}` : "–"}</span></div>
      <div class="row"><span>Model</span><span class="mono">${esc(model.digest || "–")}</span></div></div></section>
  </div>
  <section class="card c12"><h2>Last 80 parts <span class="end">Select a part to trace it</span></h2>${strip(S.parts)}<div style="margin-top:9px">${partKey}</div></section>
  <section class="card c12"><h2>Machine health, read from the same cycles <span class="end">σ from each component's healthy baseline</span></h2>${healthCards(s.health)}</section></div>`;
}

function cause(d) {
  const hits = d.deviations.filter((x) => x.abnormal), comps = [...new Set(hits.map((x) => x.points_to).filter(Boolean))];
  if (!hits.length) return `<div class="cause"><b>No machine signal was abnormal in this cycle</b><p>${d.verdict === "fail" ? "The defect is not explained by the process data: check material, handling and tooling." : "Every tracked signal sat within 4σ of its healthy baseline."}</p></div>`;
  const first = hits[0];
  return `<div class="cause"><b>${comps.length ? "Consistent with: " + esc(comps[0]) + (comps.length > 1 ? ". Also off: " + esc(comps.slice(1).join(", ")) : "") : esc(first.label) + " was abnormal"}</b><p>${esc(first.label)} was ${num(Math.abs(first.value - first.expected), 1)} ${esc(first.unit)} ${first.value < first.expected ? "below" : "above"} its healthy level (${num(Math.abs(first.z), 0)}σ)${hits.length > 1 ? `, with ${hits.length - 1} other signal${hits.length > 2 ? "s" : ""} off as well` : ""}. Confirm on the machine.</p></div>`;
}
function traceView() {
  const d = S.part, list = S.parts.filter((p) => p.key && (!S.failedOnly || p.verdict !== "pass")).slice().reverse();
  const picker = `<section class="card c3" style="align-self:start"><h2>Parts <span class="end seg"><button data-only="1" aria-pressed="${S.failedOnly}">Rejects</button><button data-only="0" aria-pressed="${!S.failedOnly}">All</button></span></h2><div class="scroll" style="max-height:1010px"><table><tbody>${list.map((p) => `<tr class="click" data-part="${esc(p.key)}" tabindex="0"><td style="${p.key === S.key ? "font-weight:700" : ""}">${partName(p)}</td><td>${p.defects.length ? esc(p.defects.map(words).join(", ")) : LABEL[p.verdict]}</td><td class="n muted">${clock(p.ts)}</td></tr>`).join("") || '<tr><td class="muted">No rejects in this window.</td></tr>'}</tbody></table></div></section>`;
  if (!d) return `<div class="grid">${picker}<section class="card c9"><div class="empty">Select a part to see its images, the machine cycle that made it, and what was abnormal.</div></section></div>`;
  const max = Math.max(8, ...d.deviations.map((x) => Math.abs(x.z)));
  const rows = d.deviations.map((x) => { const w = Math.min(50, Math.abs(x.z) / max * 50); return `<tr><td>${esc(x.label)}</td><td class="n">${num(x.value, 1)} ${esc(x.unit)}</td><td class="n muted">${num(x.expected, 1)}</td><td><div class="bar" data-tip="<b>${esc(x.label)}</b>${num(x.z, 1)}σ from healthy"><i class="${x.abnormal ? "hot" : ""}" style="width:${w}%;${x.z < 0 ? `right:50%` : `left:50%`}"></i></div></td><td class="n">${x.z > 0 ? "+" : ""}${num(x.z, 1)}σ</td></tr>`; }).join("");
  const curves = d.curves.filter((c) => ["hold_pressure", "cavity_pressure", "mold_temp", "melt_temp"].includes(c.name)).map((c) => `<section class="card c6"><h2>${esc(c.label)} through the cycle <span class="end">${esc(c.unit)}</span></h2>${line({ y: c.values, lo: c.lo, hi: c.hi, mid: c.mid, w: 620, h: 190, right: 14, x: [0, 40, 80, 120, 159].map((i) => [i, num(i / 159 * c.seconds, 0) + " s"]), aria: c.label, tip: (i) => `<b>${num(i / 159 * c.seconds, 1)} s</b>This cycle ${num(c.values[i], 1)} ${esc(c.unit)}${c.mid ? `<br>Healthy ${num(c.mid[i], 1)}` : ""}` })}<div class="key"><span><i style="background:var(--s1)"></i>This part's cycle</span><span><i style="background:var(--ink-3)"></i>Healthy median</span><span><i style="background:var(--band)"></i>98% of healthy cycles</span></div></section>`).join("");
  const av = d.audit.find((a) => a.record === "part_verdict") || d.audit[0];
  return `<div class="grid">${picker}<div class="c9 grid" style="align-content:start">
   <section class="card c7"><h2>1. What the cameras saw <span class="end">${heatBtn(d)}</span></h2>${verdictHead(d)}${viewTiles(d)}</section>
   <section class="card c5"><h2>2. What the machine did <span class="end">cycle ${d.sequence} · link confidence ${pct(d.link_confidence, 0)}</span></h2>${cause(d)}<table class="dev"><thead><tr><th>Signal</th><th class="n">This cycle</th><th class="n">Healthy</th><th></th><th class="n">Deviation</th></tr></thead><tbody>${rows}</tbody></table></section>
   ${curves}
   <section class="card c12"><h2>3. The evidence <span class="end">${d.audit.length} signed records for this part</span></h2>${av ? `<div class="rows"><div class="row"><span>Machine, cycle, material lot</span><span>${esc(d.machine)} · ${d.sequence} · ${esc(d.scalars.material_lot || "–")}</span></div><div class="row"><span>Process check before the camera saw it</span><span>${d.risk ? ({ ok: "Normal", watch: "Watch", high: "Abnormal" }[d.risk.decision] || esc(d.risk.decision)) + (d.risk.model_kind === "supervised" ? `, defect risk ${pct(d.risk.risk, 0)}` : ", deviation model") : "–"}</span></div><div class="row"><span>Record hash, chained to</span><span class="mono">${esc(av.hash)} ← ${esc(av.prev)}</span></div></div>` : '<div class="empty">No audit records for this part.</div>'}</section></div></div>`;
}

async function processView() {
  const [spc, risk, rx] = await Promise.all([api("spc?signal=" + encodeURIComponent(S.signal) + (S.machine ? "&machine=" + encodeURIComponent(S.machine) : "")), api("risk"), api("prescriptions")]);
  const sup = risk.length && risk[risk.length - 1].kind === "supervised";
  const p = spc.points, open = (i) => p[i] && select(p[i].cycle_id + "#0", "trace");
  const refs = [{ v: spc.ucl, label: "upper limit", cls: "limit" }, { v: spc.center, label: "centre" }, { v: spc.lcl, label: "lower limit", cls: "limit" }];
  const ticks = (a) => [0, .25, .5, .75, 1].map((f) => Math.round(f * (a.length - 1))).filter((i) => a[i]).map((i) => [i, a[i].n]);
  return `<div class="grid"><section class="card c12"><h2>Control chart <span class="end">3σ limits from the first ${spc.baseline || 0} cycles since the last baseline reset</span></h2>
   <div class="seg" style="margin-bottom:10px">${(spc.signals || []).map((x) => `<button data-signal="${esc(x.id)}" aria-pressed="${x.id === S.signal}">${esc(x.label)}</button>`).join("")}</div>
   ${S.state.machines.length > 1 ? `<div class="seg" style="margin-bottom:10px">${S.state.machines.map((m) => `<button data-machine="${esc(m)}" aria-pressed="${m === (S.machine || S.state.machines[0])}">${esc(m)}</button>`).join("")}</div>` : ""}
   ${p.length ? line({ y: p.map((x) => x.v), h: 300, w: 1500, refs, marks: p.map((x, i) => (x.failed ? i : -1)).filter((i) => i >= 0), x: ticks(p), aria: spc.label + " control chart", pick: open, tip: (i) => `<b>Cycle ${p[i].n}</b>${num(p[i].v, 2)} ${esc(spc.unit)}${p[i].rule ? "<br>" + esc(p[i].rule) : ""}${p[i].failed ? "<br>Part rejected" : ""}<br><span class="muted">Click to trace</span>` }) : '<div class="empty">Learning: the baseline needs 20 cycles.</div>'}
   <div class="key" style="margin-top:8px"><span><i style="background:var(--s1)"></i>${esc(spc.label)} (${esc(spc.unit)})</span><span><i style="background:var(--crit);border-radius:50%"></i>Cycle whose part was rejected</span><span>${num(spc.violations, 0)} of ${p.length} cycles break a control rule</span>${spc.cpk !== null && spc.cpk !== undefined ? `<span>Cpk ${num(spc.cpk, 2)} against the process window</span>` : ""}</div></section>
   <section class="card c8"><h2>${sup ? "Defect risk" : "Process deviation"} per cycle, scored before the camera sees the part <span class="end">${sup ? "learned from labelled parts" : "0% = like a healthy cycle. Becomes defect risk once enough rejects exist"}</span></h2>${risk.length ? line({ y: risk.map((x) => x.risk), h: 240, w: 1000, min: 0, max: 1, fmt: (v) => Math.round(v * 100) + "%", right: 20, marks: risk.map((x, i) => (x.failed ? i : -1)).filter((i) => i >= 0), x: ticks(risk), aria: "Process score per cycle", pick: (i) => select(risk[i].cycle_id + "#0", "trace"), tip: (i) => `<b>Cycle ${risk[i].n}</b>${sup ? "Risk" : "Deviation"} ${pct(risk[i].risk, 0)}${risk[i].likely ? "<br>Likely " + esc(words(risk[i].likely)) : ""}${risk[i].failed ? "<br>Part rejected" : ""}` }) : '<div class="empty">No risk model yet.</div>'}<div class="key" style="margin-top:8px"><span><i style="background:var(--s1)"></i>${sup ? "Risk" : "Deviation"}</span><span><i style="background:var(--crit);border-radius:50%"></i>Cycle whose part was rejected</span></div></section>
   <section class="card c4"><h2>Recommended corrections <span class="end">need approval</span></h2>${rx.length ? rx.slice(0, 5).map((r) => `<div class="alarm"><span class="ic" aria-hidden="true">→</span><div><p>${esc(r.text)}</p></div></div>`).join("") : '<div class="empty">None open. Corrections are suggested here and never written to the machine.</div>'}</section>
   <section class="card c12"><h2>Machine health</h2>${healthCards(S.state.health)}</section></div>`;
}

async function qualityView() {
  const q = await api("quality"), m = S.state.model || {};
  const bars = (rows, label, value, txt) => { const max = Math.max(1, ...rows.map(value)); return rows.map((r) => `<div style="display:grid;grid-template-columns:110px 1fr 90px;gap:10px;align-items:center;padding:5px 0"><span>${esc(label(r))}</span><div data-tip="<b>${esc(label(r))}</b>${esc(txt(r))}" style="height:16px"><div style="height:16px;width:${value(r) / max * 100}%;background:var(--s1);border-radius:0 4px 4px 0"></div></div><span class="muted">${esc(txt(r))}</span></div>`).join(""); };
  const c = q.scores.counts, e = q.scores.edges, W = 1000, H = 220, mx = Math.max(1, ...c), bw = (W - 60) / (c.length || 1), log = (v) => Math.log10(v + 1) / Math.log10(mx + 1);
  const hist = c.length ? `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Anomaly score distribution"><line class="axis" x1="40" x2="${W - 20}" y1="${H - 24}" y2="${H - 24}"/>${c.map((v, i) => v ? `<rect class="barf" style="${e[i] >= 1 ? "fill:var(--crit)" : ""}" rx="2" x="${40 + i * bw + 1}" width="${bw - 2}" y="${H - 24 - log(v) * (H - 50)}" height="${log(v) * (H - 50)}" data-tip="<b>${i === c.length - 1 ? "2.45 and above" : num(e[i], 2) + " to " + num(e[i + 1], 2)}</b>${v} views"/>` : "").join("")}<line class="limit" x1="${40 + 20 * bw}" x2="${40 + 20 * bw}" y1="10" y2="${H - 24}"/><text x="${46 + 20 * bw}" y="20">reject limit</text>${[0, .5, 1, 1.5, 2, 2.5].map((v) => `<text x="${40 + v / 2.5 * (W - 60)}" y="${H - 6}" text-anchor="middle">${num(v, 1)}${v === 2.5 ? "+" : ""}</text>`).join("")}</svg>` : '<div class="empty">No scores yet.</div>';
  return `<div class="grid"><section class="card c8"><h2>Yield, rolling ${q.window} parts</h2>${q.rolling.length ? line({ y: q.rolling.map((r) => r.yield), h: 240, w: 1000, right: 20, max: 1, min: Math.min(.8, ...q.rolling.map((r) => r.yield)), fmt: (v) => Math.round(v * 100) + "%", aria: "Rolling yield", tip: (i) => `<b>${pct(q.rolling[i].yield)}</b>to ${clock(q.rolling[i].ts)}` }) : '<div class="empty">Needs 30 judged parts.</div>'}</section>
   <section class="card c4"><h2>Why parts were rejected <span class="end">parts</span></h2>${q.pareto.length ? bars(q.pareto, (r) => words(r.type), (r) => r.parts, (r) => String(r.parts)) : '<div class="empty">No rejects.</div>'}<p class="muted" style="margin:10px 0 0">"Anomaly" means unlike every good part. Naming the defect needs labelled examples.</p></section>
   <section class="card c8"><h2>How close each view came to the reject limit <span class="end">${num(q.scores.n, 0)} views · log scale</span></h2>${hist}<div class="key"><span><i style="background:var(--s1)"></i>Accepted</span><span><i style="background:var(--crit)"></i>Rejected</span></div></section>
   <section class="card c4"><h2>Rejects by camera view</h2>${bars(q.views, (r) => words(r.camera), (r) => r.rejects, (r) => `${r.rejects} of ${r.judged}`)}</section>
   <section class="card c12"><h2>Inspection model</h2><div class="rows"><div class="row"><span>Method</span><span>${esc(m.kind || "–")} ${m.backbone ? "on " + esc(m.backbone) : ""}, trained on good parts only</span></div><div class="row"><span>Reject limit</span><span>${num(m.threshold, 4)}, set from ${num(m.n_holdout, 0)} held-out good parts</span></div><div class="row"><span>False rejects</span><span>${m.part_frr_target ? `calibrated to at most ${pct(m.part_frr_target, 0)} of good parts across ${m.n_views} views` : "–"}</span></div><div class="row"><span>Model digest</span><span class="mono">${esc(m.digest || "–")}</span></div></div></section></div>`;
}

async function auditView() {
  const a = await api("audit"), ok = a.chain.ok;
  return `<div class="grid"><section class="card c12"><h2>Tamper-evident record of every verdict <span class="end"><button class="btn" id="csv">Export these records</button></span></h2><div class="verdict"><span class="big ${ok ? "pass" : ok === false ? "fail" : "other"}">${ok ? "INTACT" : ok === false ? "BROKEN" : a.chain.checking ? "CHECKING" : "NO RECORDS"}</span><span>${ok ? `${num(a.chain.records, 0)} records checked, every link and signature valid` : esc(a.chain.error || "")}</span><span class="muted">${a.chain.checked ? "checked " + clock(a.chain.checked) : ""}</span></div>
   <div class="scroll"><table><thead><tr><th>Time</th><th>Record</th><th>Cycle</th><th>View</th><th>Verdict</th><th class="n" style="padding-right:26px">Score</th><th>Image SHA-256</th><th>Model</th><th>Hash</th><th>Chained to</th></tr></thead><tbody>${a.records.map((r) => `<tr class="${r.cycle_id ? "click" : ""}" ${r.cycle_id ? `data-part="${esc(r.cycle_id)}#0" tabindex="0"` : ""}><td>${r.ts ? clock(r.ts) : "–"}</td><td>${r.record === "part_verdict" ? "Part verdict" : r.record === "view" ? "View" : esc(words(r.record))}</td><td>${esc(r.cycle_id || "–")}</td><td>${r.camera ? esc(words(r.camera)) : "All"}</td><td>${esc(words(r.verdict || "–"))}</td><td class="n" style="padding-right:26px">${num(r.score, 2)}</td><td class="mono">${esc(r.image || "–")}</td><td class="mono">${esc(r.model || "–")}</td><td class="mono">${esc(r.hash)}</td><td class="mono muted">${esc(r.prev)}</td></tr>`).join("")}</tbody></table></div></section></div>`;
}

/* ---------------------------------------------------------------- app */
function chrome() {
  const s = S.state;
  $("#nav").innerHTML = TABS.map(([id, t]) => `<button role="tab" data-tab="${id}" aria-selected="${S.tab === id}">${t}</button>`).join("");
  $("#clock").textContent = new Date().toLocaleTimeString("en-GB");
  if (!s) return;
  $("#where").textContent = `${s.line} · ${s.machine}`;
  const sim = $("#sim"); sim.textContent = s.simulated ? `Simulated station${s.time_scale !== 1 ? ` · ${s.time_scale}× speed` : ""}` : "";
  const l = $("#link"); const held = s.online && s.plc.hold_reason;
  l.className = "chip " + (!s.online ? "crit" : held ? "warn" : "good"); l.textContent = !s.online ? "■ Offline" : held ? "▲ Heartbeat held" : "● Inspecting";
}
let lastHtml = "", drawing = 0;
async function draw() {
  if (!S.state) return; const token = ++drawing; let html;
  registry.clear(); chartId = 0;
  try { html = S.tab === "live" ? liveView() : S.tab === "trace" ? traceView() : S.tab === "process" ? await processView() : S.tab === "quality" ? await qualityView() : await auditView(); }
  catch (err) { html = '<div class="card"><div class="empty">This view could not load. Retrying.</div></div>'; }
  if (token !== drawing || html === lastHtml) return;
  const boxes = [...document.querySelectorAll(".scroll")].map((b) => b.scrollTop); $("#view").innerHTML = lastHtml = html;
  document.querySelectorAll(".scroll").forEach((b, i) => { b.scrollTop = boxes[i] || 0; });
}
async function loadPart(key) { if (!key) return; const [id, cav] = key.split("#"); try { const d = await api(`part/${encodeURIComponent(id)}?cavity=${cav || 0}`); if (S.key === key) S.part = d; } catch { /* not stored yet */ } }
async function select(key, tab) { S.key = key; if (tab) S.tab = tab; hideTip(); history.replaceState(null, "", "#" + S.tab + (S.tab === "trace" ? "/" + key : "")); await loadPart(key); chrome(); draw(); }
async function tick() {
  if (S.busy) return; S.busy = true;
  try {
    const [state, parts] = await Promise.all([api("state"), api("parts?limit=600")]); S.state = state; S.parts = parts;
    if (S.tab === "live") S.key = state.latest;
    if (S.tab === "trace" && !S.key) S.key = (parts.filter((p) => p.verdict === "fail" && p.key).pop() || {}).key || null;
    if (S.key && (S.tab === "live" || !S.part || S.part.cycle_id !== S.key.split("#")[0])) await loadPart(S.key);
  } catch { if (S.state) S.state.online = false; }
  S.busy = false; chrome(); draw();
}
document.addEventListener("click", (e) => {
  const t = e.target.closest("[data-tab],[data-part],[data-heat],[data-only],[data-signal],[data-machine],#theme,#csv"); if (!t) return;
  if (t.dataset.tab) { S.tab = t.dataset.tab; history.replaceState(null, "", "#" + S.tab); lastHtml = ""; if (S.tab === "trace") { const f = S.parts.filter((p) => p.verdict === "fail" && p.key).pop(); if (f) return select(f.key); } chrome(); draw(); }
  else if (t.dataset.part !== undefined) { if (t.dataset.part) select(t.dataset.part, "trace"); }
  else if (t.dataset.heat) { S.heat = t.dataset.heat === "1"; draw(); }
  else if (t.dataset.only) { S.failedOnly = t.dataset.only === "1"; draw(); }
  else if (t.dataset.signal) { S.signal = t.dataset.signal; draw(); }
  else if (t.dataset.machine) { S.machine = t.dataset.machine; draw(); }
  else if (t.id === "theme") { const r = document.documentElement; r.dataset.theme = r.dataset.theme === "dark" ? "light" : "dark"; }
  else if (t.id === "csv") { const rows = [...document.querySelectorAll("#view table tr")].map((r) => [...r.children].map((c) => '"' + c.textContent.replace(/"/g, '""') + '"').join(",")); const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([rows.join("\n")], { type: "text/csv" })); a.download = "qualyxus-audit.csv"; a.click(); }
});
document.addEventListener("keydown", (e) => { if (e.key === "Enter" && e.target.matches?.("tr[data-part]")) select(e.target.dataset.part, "trace"); });
const [h, k] = location.hash.slice(1).split("/"); if (TABS.some(([id]) => id === h)) { S.tab = h; if (k) S.key = decodeURIComponent(k) + (location.hash.split("#")[2] ? "#" + location.hash.split("#")[2] : ""); }
chrome(); tick(); setInterval(tick, 2500); setInterval(() => { $("#clock").textContent = new Date().toLocaleTimeString("en-GB"); }, 1000);
