'use strict';
// eeg_tvns control plane client.
//
// Two message kinds arrive on one socket: "frame" (EEG + decode, at publish_hz)
// and "state" (control plane, low rate). Frames are only requested while the
// Monitor tab is visible, so the other tabs do not pay for a 16-channel buffer.

const $ = (id) => document.getElementById(id);
const state = {
  frame: null,      // last EEG/decode snapshot
  ctl: null,        // last control-plane state
  ws: null,
  view: 'monitor',
  lastFireN: -1,
  headHits: {},     // canvasId -> [{name,x,y,r}]
  clockOffset: null,  // serverUnix = performance.now()/1000 + clockOffset
  cal: null,        // active cue-timeline runner
  latSeries: [],
};

// ---------------------------------------------------------------- utilities
async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  const text = await r.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (e) { data = { detail: text }; }
  if (!r.ok) throw new Error((data && (data.detail || data.message)) || `HTTP ${r.status}`);
  return data;
}

const fmt = (v, n = 2) => (v === null || v === undefined || Number.isNaN(v))
  ? '\u2014' : Number(v).toFixed(n);

function say(id, msg, cls) {
  const el = $(id);
  if (!el) return;
  el.innerHTML = msg || '';
  el.style.color = cls === 'bad' ? 'var(--bad)'
    : cls === 'good' ? 'var(--good)' : '';
}

function fitCanvas(c) {
  const r = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.floor(r.width * dpr) || c.height !== Math.floor(r.height * dpr)) {
    c.width = Math.floor(r.width * dpr);
    c.height = Math.floor(r.height * dpr);
  }
  return dpr;
}

// ------------------------------------------------------------------ drawing
// Stacked multi-channel trace renderer, shared by the live strip chart and the
// frozen trigger window. Each channel is auto-scaled to its own row.
function drawTraces(canvasId, eeg, chNames, opts) {
  opts = opts || {};
  const c = $(canvasId); const ctx = c.getContext('2d'); const dpr = fitCanvas(c);
  ctx.clearRect(0, 0, c.width, c.height);
  if (!eeg || !eeg.length) return;
  const nCh = eeg.length, nSamp = eeg[0].length;
  if (nSamp < 2) return;
  const w = c.width, h = c.height;
  const rowH = h / nCh;
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1 * dpr;
  for (let i = 1; i < nCh; i++) {
    ctx.beginPath(); ctx.moveTo(0, i * rowH); ctx.lineTo(w, i * rowH); ctx.stroke();
  }
  const stroke = opts.color || 'rgba(157,214,255,0.85)';
  ctx.font = `${11 * dpr}px "JetBrains Mono", monospace`;
  ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  for (let ch = 0; ch < nCh; ch++) {
    const row = eeg[ch];
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < row.length; i++) { const v = row[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
    const span = Math.max(1e-6, mx - mn);
    const mid = (mn + mx) / 2;
    const yMid = (ch + 0.5) * rowH;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.1 * dpr;
    ctx.beginPath();
    for (let i = 0; i < nSamp; i++) {
      const x = (i / (nSamp - 1)) * w;
      const y = yMid - ((row[i] - mid) / span) * (rowH * 0.9);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = 'rgba(216,224,234,0.55)';
    ctx.fillText((chNames && chNames[ch]) ? chNames[ch] : `CH${ch + 1}`,
                 6 * dpr, ch * rowH + 14 * dpr);
  }
  // Vertical event markers, positioned as fractions of the x axis. Each carries
  // its own colour so a suppressed GO event never looks like a stimulation.
  for (const m of (opts.markers || [])) {
    ctx.strokeStyle = m.color || 'rgba(255,90,90,0.9)';
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath(); ctx.moveTo(m.at * w, 0); ctx.lineTo(m.at * w, h); ctx.stroke();
  }
}

const STIM_COLOR = 'rgba(255,90,90,0.9)';
const GO_COLOR = 'rgba(255,180,84,0.85)';

function drawEEG() {
  const d = state.frame; if (!d) return;
  const tNow = d.latest.t;
  const bufS = d.buffer_s || 6;
  const stim = new Set(d.stim_times || []);
  const markers = (d.go_events || [])
    .filter((t) => tNow - t >= 0 && tNow - t <= bufS)
    .map((t) => ({ at: 1 - (tNow - t) / bufS,
                   color: stim.has(t) ? STIM_COLOR : GO_COLOR }));
  drawTraces('eeg', d.eeg, d.ch_names, { markers });
}

function drawTrigger() {
  const d = state.frame; if (!d) return;
  const trig = d.trigger;
  if (!trig) {
    const c = $('trig'); const ctx = c.getContext('2d'); fitCanvas(c);
    ctx.clearRect(0, 0, c.width, c.height);
    return;
  }
  // The GO event lands at the end of the window that produced it.
  drawTraces('trig', trig.eeg, d.ch_names, {
    color: trig.stimulated ? 'rgba(255,140,140,0.85)' : 'rgba(255,200,130,0.8)',
    markers: [{ at: 1.0, color: trig.stimulated ? STIM_COLOR : GO_COLOR }],
  });
}

function drawSeries(canvasId, values, opts) {
  const c = $(canvasId); const ctx = c.getContext('2d'); const dpr = fitCanvas(c);
  ctx.clearRect(0, 0, c.width, c.height);
  const w = c.width, h = c.height;
  const pad = 24 * dpr;
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1 * dpr;
  ctx.beginPath(); ctx.moveTo(pad, h - pad); ctx.lineTo(w, h - pad); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, h - pad); ctx.stroke();
  const yMin = opts.yMin, yMax = opts.yMax;
  ctx.fillStyle = 'rgba(216,224,234,0.55)';
  ctx.font = `${10 * dpr}px "JetBrains Mono", monospace`;
  ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  ctx.fillText(String(yMax), 4 * dpr, pad + 3 * dpr);
  ctx.fillText(String(yMin), 4 * dpr, h - pad + 3 * dpr);
  if (opts.threshold !== undefined) {
    const y = h - pad - ((opts.threshold - yMin) / (yMax - yMin)) * (h - 2 * pad);
    ctx.strokeStyle = 'rgba(255,180,84,0.5)';
    ctx.setLineDash([4 * dpr, 4 * dpr]);
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w, y); ctx.stroke();
    ctx.setLineDash([]);
  }
  const n = values.length;
  if (n < 2) return;
  ctx.strokeStyle = opts.color || 'rgba(89,209,160,0.9)';
  ctx.lineWidth = 1.4 * dpr;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = pad + (i / (n - 1)) * (w - pad);
    const y = h - pad - ((values[i] - yMin) / (yMax - yMin)) * (h - 2 * pad);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// ----------------------------------------------------------------- head map
const STATUS_FILL = { good: '#35d07f', ok: '#e8c34a', bad: '#ff5a5a', unknown: '#3a4452' };

function qualitySources() {
  // Live metrics come from the decode loop; impedance from the last explicit
  // check. Either may be absent, and the map says which it is using.
  const ctl = state.ctl || {};
  const sc = ctl.signal_check || {};
  const live = (state.frame && state.frame.quality && state.frame.quality.length)
    ? state.frame.quality : (sc.results || []);
  const imp = sc.impedance || (state.frame && state.frame.impedance) || {};
  const scalp = (state.frame && state.frame.scalp) || ctl.scalp || {};
  const unplaced = (state.frame && state.frame.unplaced) || ctl.unplaced || [];
  return { live, imp, scalp, unplaced };
}

function statusFor(q, zk) {
  if (q) return q.status;
  if (zk != null) return zk <= 5 ? 'good' : (zk <= 20 ? 'ok' : 'bad');
  return 'unknown';
}

// Top-down scalp map. Positions come from the server (standard_1020, azimuthal
// projection) and colours from measured values -- nothing here is decorative.
function drawHead(canvasId, metaId, noteId) {
  const c = $(canvasId);
  if (!c || !c.offsetParent) return;
  const { live, imp, scalp, unplaced } = qualitySources();
  const dpr = fitCanvas(c), ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  const names = Object.keys(scalp);
  const cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 16 * dpr;
  if (R <= 0) return;

  ctx.strokeStyle = '#2a3340';
  ctx.lineWidth = 1.5 * dpr;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
  // nose (up) and ears, so left/right is unambiguous
  ctx.beginPath();
  ctx.moveTo(cx - 7 * dpr, cy - R); ctx.lineTo(cx, cy - R - 10 * dpr);
  ctx.lineTo(cx + 7 * dpr, cy - R); ctx.stroke();
  [-1, 1].forEach((s) => {
    ctx.beginPath();
    ctx.ellipse(cx + s * R, cy, 4 * dpr, 11 * dpr, 0, 0, Math.PI * 2);
    ctx.stroke();
  });

  // The closest 10-20 pair on this layout (FT8/FC6) sits 0.192 head-radii apart,
  // so cap the marker at half that to guarantee they never overlap at any size.
  const r = Math.max(4 * dpr, Math.min(12.5 * dpr, R * 0.192 * 0.5));

  const byName = {};
  live.forEach((q) => { byName[q.name] = q; });
  const impCh = imp.channels || {};
  const hits = [];

  names.forEach((n) => {
    const p = scalp[n];
    // Projection is nose-up (+y); canvas y grows downward, so flip it.
    const x = cx + p[0] * R, y = cy - p[1] * R;
    const status = statusFor(byName[n], impCh[n]);
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = STATUS_FILL[status] || STATUS_FILL.unknown;
    ctx.globalAlpha = status === 'unknown' ? 0.5 : 0.85;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = 'rgba(0,0,0,0.45)'; ctx.lineWidth = 1 * dpr; ctx.stroke();
    ctx.fillStyle = status === 'unknown' ? '#8b97a8' : '#0b0e13';
    ctx.font = `600 ${Math.max(7 * dpr, r * 0.8)}px ui-monospace, monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(n, x, y);
    hits.push({ name: n, x: x / dpr, y: y / dpr, r: r / dpr });
  });
  state.headHits[canvasId] = hits;

  if (metaId) {
    const nBad = live.filter((q) => q.status === 'bad').length;
    const nOk = live.filter((q) => q.status === 'ok').length;
    $(metaId).innerHTML = !live.length ? 'awaiting data'
      : nBad ? `<b>${nBad}</b> bad` + (nOk ? `, ${nOk} marginal` : '')
      : nOk ? `<b>${nOk}</b> marginal, rest good`
      : 'all channels good';
  }
  if (noteId && $(noteId)) {
    const note = $(noteId);
    if (imp.timestamp) {
      const age = (Date.now() / 1000 - imp.timestamp) / 60;
      note.innerHTML = `Impedance from a check <b>${age.toFixed(0)} min</b> ago `
        + `(not live \u2014 measuring it injects current and suspends EEG).`;
    } else {
      note.textContent = 'Colour is from live amplitude and mains noise. For true '
        + 'impedance in k\u03A9, run a contact check on the Hardware tab.';
    }
    if (unplaced.length) {
      note.innerHTML += `<br>No 10-20 position for: ${unplaced.join(', ')} `
        + `\u2014 not shown on the map.`;
    }
  }
}

function headTipFor(name) {
  const { live, imp } = qualitySources();
  const q = live.find((x) => x.name === name);
  const zk = (imp.channels || {})[name];
  if (!q && zk == null) return `<b>${name}</b>: no measurement yet.`;
  const bits = [];
  if (zk != null) bits.push(`impedance <b>${zk.toFixed(1)} k\u03A9</b>`);
  if (q) {
    bits.push(`RMS <b>${q.rms_uv.toFixed(1)} \u00B5V</b>`);
    bits.push(`mains <b>${(q.line_ratio * 100).toFixed(0)}%</b> of power`);
  }
  let s = `<b>${name}</b> \u2014 ` + bits.join(' \u00B7 ');
  if (q && q.reason) s += `<br>${q.reason}`;
  return s;
}

function bindHeadHover(canvasId, tipId) {
  const c = $(canvasId);
  if (!c) return;
  const idle = 'Hover an electrode for its measured values.';
  c.addEventListener('mousemove', (e) => {
    const rect = c.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const hit = (state.headHits[canvasId] || []).find(
      (h) => (mx - h.x) ** 2 + (my - h.y) ** 2 <= (h.r + 3) ** 2);
    $(tipId).innerHTML = hit ? headTipFor(hit.name) : idle;
  });
  c.addEventListener('mouseleave', () => { $(tipId).innerHTML = idle; });
}

// ----------------------------------------------------------------- word bars
function renderWord() {
  const d = state.frame; if (!d) return;
  const hero = $('wordHero'), bars = $('wordBars'), note = $('wordNote');
  if (!d.word_model_path) {
    hero.className = 'word-hero idle';
    hero.textContent = 'no word model';
    bars.innerHTML = '';
    note.textContent = 'Train one on the Train tab, or assign one on the Models tab.';
    return;
  }
  const names = Object.keys(d.word_names || {})
    .sort((a, b) => Number(a) - Number(b))
    .map((k) => d.word_names[k]);
  const probs = (d.word && d.word.probs) ? d.word.probs : {};
  const hasProbs = Object.keys(probs).length > 0;

  if (hasProbs) {
    hero.className = 'word-hero';
    hero.textContent = d.word.name || '-';
  } else {
    hero.className = 'word-hero idle';
    hero.textContent = 'awaiting GO';
  }

  let topName = null, topVal = -1;
  for (const n of names) {
    const v = probs[n] || 0;
    if (v > topVal) { topVal = v; topName = n; }
  }
  bars.innerHTML = names.map((n) => {
    const v = probs[n] || 0;
    const cls = (hasProbs && n === topName) ? 'bar top' : 'bar';
    return `<div class="${cls}">
      <div class="lbl">${n}</div>
      <div class="track"><div class="fill" style="width:${(v * 100).toFixed(1)}%"></div></div>
      <div class="pct">${(v * 100).toFixed(0)}%</div>
    </div>`;
  }).join('');

  note.textContent = hasProbs
    ? `decoded in ${(d.word.latency_ms || 0).toFixed(2)} ms \u2014 display only, never gates tVNS`
    : 'decoded only while GO is active \u2014 display only, never gates tVNS';
}

// ------------------------------------------------------------ monitor render
function renderMonitor() {
  const d = state.frame;
  if (!d) return;
  $('bufLen').textContent = (d.buffer_s || 6).toFixed(0);
  $('pgo').textContent = d.latest.p_go.toFixed(3);
  $('thresh').textContent = (d.go_threshold ?? 0.5).toFixed(2);
  $('goEvents').textContent = d.latest.n_go_events;
  $('stims').textContent = d.latest.n_stimulations;
  const armed = !!(state.ctl && state.ctl.arm && state.ctl.arm.armed);
  $('goNote').innerHTML = armed
    ? '<span style="color:var(--danger)">armed \u2014 gating tVNS</span>'
    : 'disarmed \u2014 not gating';
  $('goStatNote').textContent = armed
    ? `${d.latest.n_decisions} decisions so far. Every GO event past the `
      + 'refractory window is triggering the stimulator.'
    : `${d.latest.n_decisions} decisions so far. GO events are being detected and `
      + 'logged but suppressed; arm on the Hardware tab to stimulate.';
  const go = $('goInd');
  if (d.latest.go) { go.classList.add('on'); go.textContent = 'GO'; }
  else { go.classList.remove('on'); go.textContent = 'IDLE'; }
  // Flash only on real stimulation: flashing for a suppressed event would read
  // as though the patient had been stimulated.
  const nStim = (d.stim_times || []).length;
  if (nStim > state.lastFireN) {
    if (state.lastFireN >= 0) {
      const f = $('fireFlash'); f.style.opacity = '1';
      setTimeout(() => (f.style.opacity = '0'), 200);
    }
    state.lastFireN = nStim;
  }
  const ls = d.latency_stats || {};
  $('lat').textContent = `latency last / median / p95: ${fmt(ls.last_ms)} / `
    + `${fmt(ls.median_ms)} / ${fmt(ls.p95_ms)} ms`;

  const trig = d.trigger;
  $('trigMeta').innerHTML = trig
    ? `${(trig.window_s || 0).toFixed(1)} s window &middot; p(go)=<b>${trig.p_go.toFixed(3)}</b> \u2265 ${trig.threshold.toFixed(2)}`
      + (trig.word && trig.word.name ? ` &middot; word=<b>${trig.word.name}</b>` : '')
      + ` &middot; ${(d.latest.t - trig.t).toFixed(1)} s ago &middot; `
      + (trig.stimulated
          ? '<b style="color:var(--danger)">stimulated</b>'
          : '<b style="color:var(--go)">suppressed (disarmed)</b>')
    : 'awaiting first GO event';

  if (state.view !== 'monitor') return;
  drawEEG();
  drawTrigger();
  renderWord();
  drawHead('head', 'headMeta', 'impNote');
  drawSeries('pgo_chart', d.p_history || [], {
    yMin: 0, yMax: 1, threshold: d.go_threshold ?? 0.5, color: 'rgba(89,209,160,0.95)' });
  drawSeries('lat_chart', state.latSeries, {
    yMin: 0, yMax: 30, color: 'rgba(157,214,255,0.9)' });
}

// -------------------------------------------------------------- header/banner
const STATE_PILL = {
  idle: '', probing: 'busy', signal_check: 'busy',
  calibrating: 'busy', decoding: 'good', error: 'bad',
};

function renderHeader() {
  const ctl = state.ctl;
  const d = state.frame;
  if (ctl) {
    const s = ctl.session || {};
    const cls = STATE_PILL[s.state] || '';
    $('sessionState').innerHTML =
      `<span class="pill ${cls}">${(s.state || 'idle').replace('_', ' ')}</span>`;
    $('source').textContent = s.port ? `port: ${s.port}` : 'no board';
    const go = (ctl.slots && ctl.slots.go) || {};
    const wd = (ctl.slots && ctl.slots.word) || {};
    $('model').textContent = 'go: ' + (go.name || 'none');
    $('wordModel').textContent = 'word: ' + (wd.name || 'none');
    $('channels').textContent = (ctl.channel_names || []).length
      ? `${ctl.channel_names.length} ch` : '- ch';
  }
  if (d) {
    $('channels').textContent =
      `${d.ch_names.length} ch @ ${d.display_sfreq.toFixed(0)} Hz disp`;
  }

  const banner = $('banner');
  const problems = [];
  if (ctl) {
    const s = ctl.session || {};
    if (s.state === 'error' || (s.detail && s.error)) {
      problems.push(`<b>Acquisition stopped.</b> ${s.detail || ''}`);
    }
    if (ctl.arm && ctl.arm.armed) {
      problems.push('<b>ARMED.</b> tVNS will fire when the GO decoder crosses '
        + 'threshold. Disarm on the Hardware tab when you are done.');
    }
  } else if (d && d.status && d.status.state === 'error') {
    problems.push(`<b>Acquisition stopped.</b> ${d.status.detail || ''}`);
  }
  if (problems.length) {
    banner.classList.add('show');
    banner.innerHTML = problems.join(' &nbsp;|&nbsp; ');
  } else {
    banner.classList.remove('show');
  }
}

// -------------------------------------------------------------- hardware tab
let hwTouchedChannels = false;

function renderHardware() {
  const ctl = state.ctl; if (!ctl) return;
  const s = ctl.session || {};
  const busy = s.state !== 'idle';
  $('hwState').textContent = s.state + (s.detail ? ` \u2014 ${s.detail}` : '');

  // Ports: preserve the user's selection across rescans.
  const sel = $('hwPort'); const cur = sel.value;
  const ports = ctl.ports || [];
  const opts = ports.map((p) =>
    `<option value="${p.device}">${p.device}${p.description ? ' \u2014 ' + p.description : ''}</option>`);
  const joined = opts.join('');
  if (sel.dataset.sig !== joined) {
    sel.dataset.sig = joined;
    sel.innerHTML = joined || '<option value="">(no serial ports found)</option>';
    if (cur && ports.some((p) => p.device === cur)) sel.value = cur;
    else if (s.port) sel.value = s.port;
  }

  const bsel = $('hwBoard');
  if (!bsel.dataset.filled) {
    bsel.innerHTML = (ctl.boards || []).map((b) =>
      `<option value="${b.id}">${b.label}</option>`).join('');
    bsel.dataset.filled = '1';
    if (s.board) bsel.value = s.board;
  }

  if (!hwTouchedChannels && (ctl.channel_names || []).length) {
    $('hwChannels').value = ctl.channel_names.join(', ');
  }

  $('hwStart').disabled = busy;
  $('hwStop').disabled = s.state !== 'decoding';
  $('hwProbe').disabled = busy;
  $('scRun').disabled = busy;

  // ARM
  const arm = ctl.arm || {};
  const box = $('armBox');
  box.classList.toggle('on', !!arm.armed);
  $('armState').textContent = arm.armed ? 'ARMED' : 'DISARMED';
  $('armState').classList.toggle('safe', !arm.armed);
  $('armBtn').disabled = !!arm.armed;
  $('disarmBtn').disabled = !arm.armed;
  const go = (ctl.slots && ctl.slots.go) || {};
  $('armModel').textContent = go.name || 'none';
  const v = go.verdict || {};
  $('armVerdict').innerHTML = v.level
    ? `<span class="pill ${v.level === 'ok' ? 'good' : v.level === 'unknown' ? 'warn' : 'bad'}">${v.label}</span>`
    : '\u2014';
  const why = [];
  if (arm.armed && arm.expires_in_s != null) {
    why.push(`Auto-disarms in ${Math.max(0, Math.round(arm.expires_in_s))} s.`);
  } else if (!arm.armed) {
    why.push('While disarmed the loop decodes and displays but never triggers the stimulator.');
  }
  if (!arm.can_arm && arm.reason) why.push(arm.reason);
  $('armWhy').textContent = why.join(' ');
  $('armNote').innerHTML = v.detail || '';

  // Signal check
  const sc = ctl.signal_check || {};
  $('scProgWrap').hidden = !sc.running;
  if (sc.running && sc.total) {
    $('scProg').style.width = `${(100 * sc.done / sc.total).toFixed(0)}%`;
  }
  if (sc.running) {
    say('scStatus', sc.stage || `measuring channel ${sc.done + 1} of ${sc.total}\u2026`);
  } else if (sc.error) {
    say('scStatus', sc.error, 'bad');
  } else if ((sc.results || []).length) {
    const nBad = sc.results.filter((q) => q.status === 'bad').length;
    say('scStatus', nBad
      ? `${nBad} channel(s) need attention \u2014 reseat or re-gel, then re-run.`
      : 'All channels look usable.', nBad ? 'bad' : 'good');
  }
  renderQualityTable(sc.results || []);
  drawHead('headBig', 'headMetaBig', null);
}

function renderQualityTable(rows) {
  const host = $('scTable');
  if (!rows.length) {
    host.innerHTML = '<div class="empty">No measurements yet. Run a contact check.</div>';
    return;
  }
  const body = rows.map((q) => `<tr>
    <td class="num">${q.index + 1}</td>
    <td>${q.name}</td>
    <td class="num">${q.impedance_kohm == null ? '\u2014' : q.impedance_kohm.toFixed(1) + ' k\u03A9'}</td>
    <td class="num">${q.rms_uv.toFixed(1)} \u00B5V</td>
    <td class="num">${(q.line_ratio * 100).toFixed(0)}%</td>
    <td><span class="pill ${q.status === 'good' ? 'good' : q.status === 'ok' ? 'warn' : q.status === 'bad' ? 'bad' : ''}">${q.status}</span></td>
    <td>${q.reason || ''}</td></tr>`).join('');
  host.innerHTML = `<table><thead><tr><th>Ch</th><th>Site</th><th>Impedance</th>
    <th>RMS</th><th>Mains</th><th>Status</th><th>Notes</th></tr></thead>
    <tbody>${body}</tbody></table>`;
}

function hwParams() {
  const names = $('hwChannels').value.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean);
  return { port: $('hwPort').value, board: $('hwBoard').value, channel_names: names };
}

function wireHardware() {
  $('hwChannels').addEventListener('input', () => { hwTouchedChannels = true; });
  $('hwRescan').onclick = async () => {
    try { await api('POST', '/api/ports/rescan'); } catch (e) { say('hwProbeOut', e.message, 'bad'); }
  };
  $('hwProbe').onclick = async () => {
    say('hwProbeOut', 'Probing\u2026');
    try {
      const r = await api('POST', '/api/board/probe', hwParams());
      say('hwProbeOut', `${r.board_label}: <b>${r.n_eeg}</b> EEG channels at `
        + `<b>${r.sfreq.toFixed(0)} Hz</b>. ${r.note || ''}`, 'good');
    } catch (e) { say('hwProbeOut', e.message, 'bad'); }
  };
  $('hwStart').onclick = async () => {
    say('hwProbeOut', 'Starting\u2026');
    try {
      await api('POST', '/api/session/decode/start', hwParams());
      say('hwProbeOut', 'Closed loop running. Switch to Monitor to watch it.', 'good');
    } catch (e) { say('hwProbeOut', e.message, 'bad'); }
  };
  $('hwStop').onclick = async () => {
    try { await api('POST', '/api/session/stop'); say('hwProbeOut', 'Stopped.'); }
    catch (e) { say('hwProbeOut', e.message, 'bad'); }
  };
  $('armBtn').onclick = async () => {
    const ctl = state.ctl || {};
    const go = (ctl.slots && ctl.slots.go) || {};
    if (!go.name) { say('armNote', 'No GO model is assigned.', 'bad'); return; }
    const v = go.verdict || {};
    let ack = false;
    if (v.level && v.level !== 'ok') {
      ack = confirm(`${v.label}\n\n${(v.detail || '').replace(/<[^>]*>/g, '')}\n\n`
        + 'Arm anyway?');
      if (!ack) return;
    }
    const typed = prompt(`Type the GO model filename to arm stimulation:\n\n${go.name}`);
    if (typed === null) return;
    try {
      await api('POST', '/api/arm',
                { armed: true, confirm: typed.trim(), acknowledge_unvalidated: ack });
      say('armNote', '');
    } catch (e) { say('armNote', e.message, 'bad'); }
  };
  $('disarmBtn').onclick = async () => {
    try { await api('POST', '/api/arm', { armed: false }); }
    catch (e) { say('armNote', e.message, 'bad'); }
  };
  $('scRun').onclick = async () => {
    const p = hwParams();
    p.line_freq = Number($('scLine').value);
    p.impedance = $('scImp').value === '1';
    p.impedance_input = $('scInput').value;
    say('scStatus', 'Starting check\u2026');
    try { await api('POST', '/api/signal_check/start', p); }
    catch (e) { say('scStatus', e.message, 'bad'); }
  };
  bindHeadHover('head', 'headTip');
  bindHeadHover('headBig', 'headTipBig');
}

// ------------------------------------------------------------- calibrate tab
// The server owns the timeline; the browser pre-schedules every cue against a
// synced clock so no per-cue network jitter reaches the recording, and reports
// back when each cue was actually painted so the true lag is measured.
async function syncClock(samples = 7) {
  let best = { rtt: Infinity, offset: 0 };
  for (let i = 0; i < samples; i++) {
    const t0 = performance.now();
    const j = await api('GET', '/api/time');
    const t1 = performance.now();
    const rtt = t1 - t0;
    // The server read its clock at roughly the midpoint of the round trip.
    const offset = j.server_time - ((t0 + t1) / 2) / 1000;
    if (rtt < best.rtt) best = { rtt, offset };
  }
  state.clockOffset = best.offset;
  $('calClock').textContent = `clock synced \u00b1${(best.rtt / 2).toFixed(1)} ms`;
  return best;
}

const serverNow = () => performance.now() / 1000 + (state.clockOffset || 0);

function showCue(kind, text, count) {
  const box = $('cueBox');
  box.className = 'cue ' + kind;
  $('cueKind').textContent = kind === 'action' ? 'now' : kind;
  $('cueText').textContent = text;
  $('cueCount').textContent = count || '';
}

function startCueRunner(timeline) {
  const events = [];
  timeline.trials.forEach((t, i) => {
    const count = `trial ${i + 1} of ${timeline.trials.length}`;
    events.push({ at: t.prepare_at, kind: 'prepare', text: 'get ready', count });
    events.push({ at: t.action_at, kind: 'action', text: t.label, count, index: i });
    events.push({ at: t.iti_at, kind: 'rest', text: 'relax', count });
  });
  events.sort((a, b) => a.at - b.at);
  if (timeline.settle_until) {
    events.unshift({ at: 0, kind: 'settle',
                     text: 'sit still \u2014 amplifier settling', count: '' });
  }

  const runner = { i: 0, events, paints: [], stop: false, total: timeline.trials.length };
  state.cal = runner;

  function flush(final) {
    if (!runner.paints.length && !final) return;
    const batch = runner.paints.splice(0, runner.paints.length);
    if (!batch.length) return;
    api('POST', '/api/calibration/display', { paints: batch }).catch(() => {
      // Losing a lag report degrades the diagnostic only: the server keeps its
      // planned onset for those trials and flags them.
    });
  }

  function tick() {
    if (runner.stop) return;
    const now = serverNow();
    while (runner.i < events.length && events[runner.i].at <= now) {
      const ev = events[runner.i++];
      showCue(ev.kind, ev.text, ev.count);
      if (ev.kind === 'action') {
        const idx = ev.index;
        // The next frame's timestamp is when this paint became visible.
        requestAnimationFrame((ts) => {
          runner.paints.push({ index: idx, shown_at: ts / 1000 + (state.clockOffset || 0) });
          if (runner.paints.length >= 4) flush(false);
        });
        $('calProg').style.width = `${(100 * (idx + 1) / runner.total).toFixed(0)}%`;
      }
    }
    if (runner.i < events.length) requestAnimationFrame(tick);
    else { flush(true); showCue('settle', 'finishing\u2026', ''); }
  }
  requestAnimationFrame(tick);
}

function stopCueRunner() {
  if (state.cal) state.cal.stop = true;
  state.cal = null;
}

function renderCalibrate() {
  const ctl = state.ctl; if (!ctl) return;
  const cal = ctl.calibration || {};
  const s = ctl.session || {};
  const running = s.state === 'calibrating';
  $('calState').textContent = running ? `recording \u2014 trial ${cal.trial || 0}/${cal.total || 0}` : s.state;
  $('calStart').disabled = s.state !== 'idle';
  $('calAbort').disabled = !running;
  $('calProgWrap').hidden = !running;

  const n = Number($('calTrials').value) || 40;
  const a = Number($('calAction').value) || 2.5;
  const mins = (n * 2 * (1.0 + a + 2.0)) / 60;
  $('calEstimate').textContent = `${n} attempt + ${n} rest trials, about `
    + `${mins.toFixed(1)} min.`;

  if (!running) { $('cueBox').className = 'cue'; }
  if (cal.message) say('calMsg', cal.message, cal.error ? 'bad' : '');
  if (cal.lag && cal.lag.n) {
    $('calLag').innerHTML = `Cue display lag: median <b>${(cal.lag.median_ms).toFixed(1)} ms</b>, `
      + `max <b>${(cal.lag.max_ms).toFixed(1)} ms</b> over ${cal.lag.n} trials`
      + (cal.lag.flagged ? ` \u2014 <span style="color:var(--bad)">${cal.lag.flagged} trial(s) exceeded tolerance and kept their planned onset</span>` : '');
  }

  // A timeline appears once the server has opened the board and planned the run.
  if (cal.timeline && (!state.cal || state.cal.id !== cal.timeline.id)) {
    startCueRunner(cal.timeline);
    state.cal.id = cal.timeline.id;
  }
  if (!running && state.cal) stopCueRunner();

  renderRecordings(cal.recordings || []);
}

function renderRecordings(rows) {
  const host = $('calList');
  if (!rows.length) {
    host.innerHTML = '<div class="empty">No recordings yet.</div>';
    return;
  }
  host.innerHTML = `<table><thead><tr><th>File</th><th>Subject</th><th>Session</th>
    <th>Attempt</th><th>Rest</th><th>Rate</th><th>Channels</th></tr></thead><tbody>`
    + rows.map((r) => `<tr><td>${r.name}</td><td class="num">${r.subject}</td>
      <td class="num">${r.session}</td><td class="num">${r.n_attempt}</td>
      <td class="num">${r.n_rest}</td><td class="num">${r.sfreq.toFixed(0)} Hz</td>
      <td class="num">${r.n_channels}</td></tr>`).join('')
    + '</tbody></table>';
}

function wireCalibrate() {
  ['calTrials', 'calAction'].forEach((id) =>
    $(id).addEventListener('input', renderCalibrate));
  $('calStart').onclick = async () => {
    say('calMsg', 'Syncing clock\u2026');
    try {
      await syncClock();
      const p = hwParams();
      p.subject = Number($('calSubject').value);
      p.session = Number($('calSession').value);
      p.trials_per_class = Number($('calTrials').value);
      p.action_s = Number($('calAction').value);
      await api('POST', '/api/calibration/start', p);
      say('calMsg', 'Recording. Follow the cues.', 'good');
    } catch (e) { say('calMsg', e.message, 'bad'); stopCueRunner(); }
  };
  $('calAbort').onclick = async () => {
    stopCueRunner();
    try { await api('POST', '/api/session/stop'); say('calMsg', 'Aborted; nothing saved.'); }
    catch (e) { say('calMsg', e.message, 'bad'); }
  };
}

// ----------------------------------------------------------------- train tab
function renderTrain() {
  const ctl = state.ctl; if (!ctl) return;
  const tr = ctl.training || {};
  const running = tr.state === 'running';
  $('trainState').textContent = tr.state || 'idle';
  $('trStart').disabled = running || (ctl.session || {}).state === 'decoding';
  $('trCancel').disabled = !running;
  $('trProgWrap').hidden = !running;
  $('trProg').style.width = `${(100 * (tr.progress || 0)).toFixed(0)}%`;
  $('trStage').textContent = running ? (tr.stage || '')
    : (ctl.session || {}).state === 'decoding'
      ? 'Training is blocked while the closed loop runs, so a heavy fit never '
        + 'competes with the stimulation gate for CPU.'
      : (tr.error || '');
  if (tr.error) say('trStage', tr.error, 'bad');

  const log = $('trLog');
  if ((tr.log || []).length) {
    const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 24;
    log.textContent = tr.log.join('\n');
    if (atBottom) log.scrollTop = log.scrollHeight;
  }
  renderTrainResult(tr.result);
}

function verdictPill(v) {
  if (!v || !v.level) return '';
  const cls = v.level === 'ok' ? 'good' : v.level === 'unknown' ? 'warn' : 'bad';
  return `<span class="pill ${cls}">${v.label}</span>`;
}

function renderTrainResult(res) {
  const host = $('trResults');
  if (!res) { host.innerHTML = '<div class="empty">No results yet.</div>'; return; }
  const m = res.metrics || {};
  const perm = m.permutation || {};
  const chance = 1 / Math.max(2, (m.n_classes || 2));
  const rows = [
    ['Task', m.task],
    ['Trials / subjects', `${m.n_trials} / ${m.n_subjects}`],
    ['Channels', m.n_channels],
    ['Cross-subject LOSO', `${fmt(m.cross_subject_bacc_mean, 3)} \u00b1 ${fmt(m.cross_subject_bacc_std, 3)}`],
    ['Within-subject CV', `${fmt(m.within_subject_bacc_mean, 3)} \u00b1 ${fmt(m.within_subject_bacc_std, 3)}`],
    ['Chance level', fmt(chance, 3)],
    ['Permutation', perm.n_permutations
      ? `observed ${fmt(perm.observed, 3)} vs null ${fmt(perm.null_mean, 3)}\u00b1${fmt(perm.null_std, 3)}, p = ${fmt(perm.p_value, 4)}`
      : 'not run \u2014 no empirical chance baseline'],
    ['Decode latency', `median ${fmt((m.latency || {}).median_ms)} ms, p95 ${fmt((m.latency || {}).p95_ms)} ms`],
  ];
  const plots = (res.plots || []).map((p) =>
    `<img src="/outputs/${p}" style="max-width:100%;border:1px solid var(--line);border-radius:4px;margin-top:10px" />`).join('');
  host.innerHTML = `<div class="grid2"><div><table><tbody>`
    + rows.map(([k, v]) => `<tr><th style="width:45%">${k}</th><td>${v}</td></tr>`).join('')
    + `</tbody></table>${res.verdict ? '<div style="margin-top:10px">' + verdictPill(res.verdict) + '</div>' : ''}
       <div class="disclaimer">${res.verdict && res.verdict.detail ? res.verdict.detail : ''}</div>
       </div><div>${plots}</div></div>`;
  $('trResultNote').textContent = res.model_path ? `saved ${res.model_path}` : '';
}

function wireTrain() {
  const syncSource = () => {
    const src = $('trSource').value;
    $('trPath').value = src === 'calibration' ? 'calib/*.npz' : './ds003626';
    $('trTask').value = src === 'calibration' ? 'go' : 'word';
    $('trCond').disabled = src === 'calibration';
  };
  $('trSource').onchange = syncSource;
  syncSource();
  $('trStart').onclick = async () => {
    const body = {
      source: $('trSource').value,
      path: $('trPath').value.trim(),
      task: $('trTask').value,
      condition: $('trCond').value,
      classifier: $('trClf').value,
      permutations: Number($('trPerm').value),
      align: $('trAlign').checked,
    };
    say('trStage', 'Starting\u2026');
    try { await api('POST', '/api/train/start', body); }
    catch (e) { say('trStage', e.message, 'bad'); }
  };
  $('trCancel').onclick = async () => {
    try { await api('POST', '/api/train/cancel'); }
    catch (e) { say('trStage', e.message, 'bad'); }
  };
}

// ---------------------------------------------------------------- models tab
function renderModels() {
  const ctl = state.ctl; if (!ctl) return;
  const models = ctl.models || [];
  const slots = ctl.slots || {};
  $('mdGo').textContent = (slots.go && slots.go.name) || 'none';
  $('mdWord').textContent = (slots.word && slots.word.name) || 'none';
  $('mdNote').textContent = `${models.length} found in outputs/`;

  const host = $('mdTable');
  if (!models.length) {
    host.innerHTML = '<div class="empty">No .joblib models found. Train one on the '
      + 'Train tab, or import one below.</div>';
    return;
  }
  host.innerHTML = `<table><thead><tr><th>File</th><th>Task</th><th>Ch</th><th>Rate</th>
    <th>LOSO</th><th>Gating verdict</th><th>Assign</th></tr></thead><tbody>`
    + models.map((m) => {
      const isGo = slots.go && slots.go.path === m.path;
      const isWd = slots.word && slots.word.path === m.path;
      const btns = [];
      if (m.task === 'go') {
        btns.push(`<button class="btn" data-assign="go" data-path="${m.path}" ${isGo ? 'disabled' : ''}>${isGo ? 'in GO slot' : 'to GO'}</button>`);
      } else {
        btns.push(`<button class="btn" data-assign="word" data-path="${m.path}" ${isWd ? 'disabled' : ''}>${isWd ? 'in word slot' : 'to word'}</button>`);
      }
      return `<tr class="${isGo || isWd ? 'sel' : ''}">
        <td>${m.name}${m.error ? ` <span style="color:var(--bad)">(${m.error})</span>` : ''}</td>
        <td>${m.task || '?'}</td><td class="num">${m.n_channels ?? '\u2014'}</td>
        <td class="num">${m.sfreq ? m.sfreq.toFixed(0) + ' Hz' : '\u2014'}</td>
        <td class="num">${m.loso == null ? '\u2014' : m.loso.toFixed(3)}</td>
        <td>${verdictPill(m.verdict)}</td>
        <td>${btns.join(' ')}</td></tr>`;
    }).join('') + '</tbody></table>';

  host.querySelectorAll('button[data-assign]').forEach((b) => {
    b.onclick = async () => {
      try {
        await api('POST', '/api/models/assign',
                  { slot: b.dataset.assign, path: b.dataset.path });
        say('mdUploadOut', '');
      } catch (e) { say('mdUploadOut', e.message, 'bad'); }
    };
  });
}

function wireModels() {
  $('mdUpload').onclick = async () => {
    const f = $('mdFile').files[0];
    if (!f) { say('mdUploadOut', 'Choose a .joblib file first.', 'bad'); return; }
    say('mdUploadOut', `Uploading ${f.name}\u2026`);
    try {
      // Raw body upload: avoids a multipart dependency for a single binary file.
      const r = await fetch(`/api/models/import?name=${encodeURIComponent(f.name)}`,
                            { method: 'PUT', body: f });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || `HTTP ${r.status}`);
      say('mdUploadOut', `Imported as ${j.name}. ${j.note || ''}`, 'good');
    } catch (e) { say('mdUploadOut', e.message, 'bad'); }
  };
}

// ---------------------------------------------------------------------- tabs
function setView(name) {
  state.view = name;
  document.querySelectorAll('.view').forEach((v) =>
    v.classList.toggle('active', v.id === `view-${name}`));
  document.querySelectorAll('#tabs button').forEach((b) =>
    b.classList.toggle('active', b.dataset.view === name));
  // Only stream the EEG buffer when something is going to draw it.
  if (state.ws && state.ws.readyState === 1) {
    state.ws.send(JSON.stringify({ eeg: name === 'monitor' }));
  }
  renderAll();
}

function renderAll() {
  renderHeader();
  if (state.view === 'monitor') renderMonitor();
  else if (state.view === 'hardware') renderHardware();
  else if (state.view === 'calibrate') renderCalibrate();
  else if (state.view === 'train') renderTrain();
  else if (state.view === 'models') renderModels();
}

// ----------------------------------------------------------------- websocket
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;
  ws.onopen = () => {
    $('connDot').classList.add('live'); $('conn').textContent = 'live';
    ws.send(JSON.stringify({ eeg: state.view === 'monitor' }));
  };
  ws.onclose = () => {
    $('connDot').classList.remove('live'); $('conn').textContent = 'disconnected';
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.kind === 'frame') {
        state.frame = msg.data;
        const last = (msg.data.latency_stats || {}).last_ms;
        if (last != null) {
          state.latSeries.push(last);
          if (state.latSeries.length > 300) state.latSeries.shift();
        }
        if (state.view === 'monitor') { renderHeader(); renderMonitor(); }
      } else if (msg.kind === 'state') {
        state.ctl = msg.data;
        renderAll();
      }
    } catch (e) { console.error(e); }
  };
}

window.addEventListener('resize', renderAll);
document.getElementById('tabs').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-view]');
  if (b) setView(b.dataset.view);
});

wireHardware();
wireCalibrate();
wireTrain();
wireModels();
connect();
