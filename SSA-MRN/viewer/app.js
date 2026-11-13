'use strict';
// Result viewer: loads data/index.json and per-scene int16 arrays written by export.py.

const PEAK = 2047;            // common/metrics.rr_metrics PSNR peak
const WINDOWS = [16, 32, 64, 128];
const MIN_CELL = 150;         // narrower than this and the layer grid wraps
const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v; else if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else if (v !== false && v != null) e.setAttribute(k, v === true ? '' : v);
  }
  for (const k of kids.flat(3)) if (k != null && k !== false) e.append(k instanceof Node ? k : document.createTextNode(k));
  return e;
};

const S = {
  index: null, ds: null, scene: 0, visible: [], ref: null, cand: null, comp: 'RGB', win: 64, x: 96, y: 96,
  errBand: 'mean', errScale: 'ref', errManual: 30, zoomScale: 'scene', zoomStretch: 'scene', inputs: false, popOpen: false, theme: 'auto',
};
const cache = new Map();
let current = null;           // loaded scene
let tiles = [];               // layer columns on screen
let factor = 8;

// ---------- data ----------
async function loadArray(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const q = new Int16Array(await r.arrayBuffer()), f = new Float32Array(q.length);
  for (let i = 0; i < q.length; i++) f[i] = q[i] / factor;
  return f;
}

async function loadScene(dsId, i) {
  const key = `${dsId}/${i}`;
  if (cache.has(key)) return cache.get(key);
  const ds = dataset(dsId), dir = `data/${dsId}/s${String(i).padStart(2, '0')}/`;
  const ids = Object.keys(ds.results);
  const [gt, pan, ms, ...preds] = await Promise.all(['gt', 'pan', 'ms', ...ids].map(n => loadArray(dir + n + '.i16')));
  const sc = { key, ds: dsId, i, C: ds.bands, H: ds.size, W: ds.size, h: ds.ms_size, gt, pan, ms, pred: Object.fromEntries(ids.map((id, k) => [id, preds[k]])), err: {}, stretch: {}, stat: {} };
  cache.set(key, sc);
  if (cache.size > 12) cache.delete(cache.keys().next().value);
  return sc;
}

const dataset = id => S.index.datasets.find(d => d.id === id);
const model = id => S.index.models.find(m => m.id === id);
const modelsIn = ds => S.index.models.filter(m => ds.results && ds.results[m.id]);

// ---------- numeric helpers ----------
function percentile(arr, p) {
  const a = Float32Array.from(arr).sort(), pos = (p / 100) * (a.length - 1), lo = Math.floor(pos), hi = Math.ceil(pos);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}
function sub(arr, H, W, band, r0, c0, h, w) {     // band plane region as Float32Array
  const out = new Float32Array(h * w), base = band * H * W;
  for (let y = 0; y < h; y++) out.set(arr.subarray(base + (r0 + y) * W + c0, base + (r0 + y) * W + c0 + w), y * w);
  return out;
}
function mean(arr) { let s = 0; for (let i = 0; i < arr.length; i++) s += arr[i]; return s / arr.length; }
function memo(sc, key, fn) { if (!(key in sc.stat)) sc.stat[key] = fn(); return sc.stat[key]; }
function stretchOf(sc, bands, region) {           // GT 1-99% per band, scene or region
  const key = bands.join(',') + (region ? `@${region.join(',')}` : '');
  if (!sc.stretch[key]) {
    const [r0, c0, h, w] = region || [0, 0, sc.H, sc.W];
    const v = bands.map(b => sub(sc.gt, sc.H, sc.W, b, r0, c0, h, w));
    sc.stretch[key] = { lo: v.map(a => percentile(a, 1)), hi: v.map(a => percentile(a, 99)) };
  }
  return sc.stretch[key];
}
function panStretch(sc, region) {
  const key = 'pan' + (region ? `@${region.join(',')}` : '');
  if (!sc.stretch[key]) {
    const [r0, c0, h, w] = region || [0, 0, sc.H, sc.W], a = sub(sc.pan, sc.H, sc.W, 0, r0, c0, h, w);
    sc.stretch[key] = { lo: percentile(a, 1), hi: percentile(a, 99) };
  }
  return sc.stretch[key];
}
function errMap(sc, id) {                         // |pred - gt|, band mean or one band (DN)
  const key = `${id}:${S.errBand}`;
  if (!sc.err[key]) {
    const { C, H, W, gt } = sc, p = sc.pred[id], N = H * W, out = new Float32Array(N);
    if (S.errBand === 'mean') {
      for (let b = 0; b < C; b++) for (let i = 0, o = b * N; i < N; i++, o++) out[i] += Math.abs(p[o] - gt[o]);
      for (let i = 0; i < N; i++) out[i] /= C;
    } else {
      const o = +S.errBand * N;
      for (let i = 0; i < N; i++) out[i] = Math.abs(p[o + i] - gt[o + i]);
    }
    sc.err[key] = out;
  }
  return sc.err[key];
}
function gainMap(sc, id) {                        // reference error - model error (>0: model better)
  const key = `gain:${S.ref}:${id}:${S.errBand}`;
  if (!sc.err[key]) {
    const a = errMap(sc, S.ref), b = errMap(sc, id), out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
    sc.err[key] = out;
  }
  return sc.err[key];
}
function regionMetrics(sc, id, r0, c0, w) {
  const { C, H, W, gt } = sc, p = sc.pred[id], N = H * W;
  const mse = new Float64Array(C), mu = new Float64Array(C);
  let sam = 0, n = 0, mae = 0;
  for (let y = r0; y < r0 + w; y++) for (let x = c0; x < c0 + w; x++) {
    let dot = 0, na = 0, nb = 0;
    for (let b = 0, o = y * W + x; b < C; b++, o += N) {
      const g = gt[o], f = p[o], d = f - g;
      mse[b] += d * d; mu[b] += g; mae += Math.abs(d); dot += g * f; na += g * g; nb += f * f;
    }
    const norm = Math.sqrt(na * nb);
    if (norm > 0) { sam += Math.acos(Math.min(1, Math.max(-1, dot / norm))); n++; }
  }
  const P = w * w;
  let psnr = 0, erg = 0;
  for (let b = 0; b < C; b++) {
    const m = mse[b] / P, g = mu[b] / P;
    psnr += 10 * Math.log10(PEAK * PEAK / Math.max(m, 1e-30)); erg += m / Math.max(g * g, 1e-30);
  }
  return { PSNR: psnr / C, SAM: n ? sam / n * 180 / Math.PI : 0, ERGAS: 25 * Math.sqrt(erg / C), MAE: mae / (P * C) };
}
function bestWindow(sc, sign) {                   // window with the largest mean gain (sign 1) or loss (sign -1)
  const g = gainMap(sc, S.cand), { H, W } = sc, w = S.win, I = new Float64Array((H + 1) * (W + 1));
  for (let y = 0; y < H; y++) { let row = 0; for (let x = 0; x < W; x++) { row += g[y * W + x]; I[(y + 1) * (W + 1) + x + 1] = I[y * (W + 1) + x + 1] + row; } }
  let best = -Infinity, at = [0, 0];
  for (let y = 0; y + w <= H; y++) for (let x = 0; x + w <= W; x++) {
    const s = sign * (I[(y + w) * (W + 1) + x + w] - I[y * (W + 1) + x + w] - I[(y + w) * (W + 1) + x] + I[y * (W + 1) + x]);
    if (s > best) { best = s; at = [y, x]; }
  }
  return at;
}

// ---------- painting ----------
function paintRGB(cv, arr, sc, bands, st, r0, c0, h, w) {
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(w, h), d = img.data, { H, W } = sc;
  for (let k = 0; k < 3; k++) {
    const base = bands[k] * H * W, lo = st.lo[k], s = 255 / Math.max(st.hi[k] - lo, 1e-8);
    for (let y = 0; y < h; y++) for (let x = 0, o = base + (r0 + y) * W + c0, q = y * w * 4 + k; x < w; x++, o++, q += 4) d[q] = (arr[o] - lo) * s;
  }
  for (let q = 3; q < d.length; q += 4) d[q] = 255;
  ctx.putImageData(img, 0, 0);
}
function paintLR(cv, sc, bands, st, r0, c0, h, w) {
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(w, h), d = img.data, ratio = sc.H / sc.h, hh = sc.h;
  for (let k = 0; k < 3; k++) {
    const base = bands[k] * hh * hh, lo = st.lo[k], s = 255 / Math.max(st.hi[k] - lo, 1e-8);
    for (let y = 0; y < h; y++) for (let x = 0, q = y * w * 4 + k; x < w; x++, q += 4)
      d[q] = (sc.ms[base + Math.floor((r0 + y) / ratio) * hh + Math.floor((c0 + x) / ratio)] - lo) * s;
  }
  for (let q = 3; q < d.length; q += 4) d[q] = 255;
  ctx.putImageData(img, 0, 0);
}
function paintGray(cv, arr, W, st, r0, c0, h, w) {
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(w, h), d = img.data, s = 255 / Math.max(st.hi - st.lo, 1e-8);
  for (let y = 0; y < h; y++) for (let x = 0, q = y * w * 4; x < w; x++, q += 4) {
    const v = (arr[(r0 + y) * W + c0 + x] - st.lo) * s; d[q] = d[q + 1] = d[q + 2] = v; d[q + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
// Error: black -> red -> yellow -> white (same as common/result_images._heat). Gain: black centre, blue = better, orange = worse.
const BETTER = [70, 160, 255], WORSE = [255, 104, 56];
function paintMap(cv, map, W, vmax, kind, r0, c0, h, w) {
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(w, h), d = img.data;
  for (let y = 0; y < h; y++) for (let x = 0, q = y * w * 4; x < w; x++, q += 4) {
    const v = map[(r0 + y) * W + c0 + x] / vmax;
    if (kind === 'heat') { d[q] = 382.5 * v; d[q + 1] = 382.5 * v - 127.5; d[q + 2] = 765 * v - 510; }
    else { const t = Math.min(1, Math.abs(v)), c = v >= 0 ? BETTER : WORSE; d[q] = c[0] * t; d[q + 1] = c[1] * t; d[q + 2] = c[2] * t; }
    d[q + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// ---------- small view helpers ----------
const num = (v, dg) => v.toFixed(dg);
const dn = v => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1));
function delta(v, ref, higherBetter, digits, cls = 'd') {
  const d = v - ref, kind = Math.abs(d) < 0.5 * 10 ** -digits ? 'flat' : (d > 0) === higherBetter ? 'up' : 'down';
  return el('span', { class: `${cls} ${kind}` }, (d >= 0 ? '+' : '−') + Math.abs(d).toFixed(digits));
}
const signed = (v, dg) => el('b', { class: Math.abs(v) < 0.5 * 10 ** -dg ? 'flat' : v > 0 ? 'up' : 'down' }, (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(dg));
const scaleEl = (kind, text) => el('span', { class: 'scale' }, el('i', { class: kind }), text);
const setCap = (node, left, right) => node.replaceChildren(el('span', {}, left), right || '');
function sceneLabel(ds, i) { return (ds.scene_labels && ds.scene_labels[i]) || ''; }
function seg(options, cur, onpick) {
  return el('span', { class: 'seg' }, options.map(([v, text]) => el('button', { type: 'button', 'aria-pressed': String(String(v) === String(cur)), onclick: () => onpick(v) }, text)));
}

// ---------- layout ----------
function layer(t) {
  const frame = zoom => {
    const f = el('div', { class: 'frame ' + (zoom ? 'zoom' : 'full') }), cv = el('canvas');
    f.append(cv, el('i', { class: 'vl' }), el('i', { class: 'hl' }));
    if (!zoom) f.append(el('div', { class: 'box' }));
    return [f, cv];
  };
  [t.fullFrame, t.full] = frame(false); [t.zoomFrame, t.zoom] = frame(true);
  t.cap1 = el('div', { class: 'cap first' }); t.cap2 = el('div', { class: 'cap' });
  return el('div', { class: 'layer' }, el('h3', {}, t.title, t.tag ? el('span', { class: 'tag' }, t.tag) : null, t.sub ? el('small', {}, t.sub) : null),
    t.fullFrame, t.cap1, t.zoomFrame, t.cap2);
}

function build() {
  const ds = dataset(S.ds), app = $('#app'), all = modelsIn(ds), vis = S.visible.map(model), ref = model(S.ref), others = vis.filter(m => m.id !== S.ref), cand = S.cand && model(S.cand);
  document.querySelectorAll('#dsTabs button').forEach(b => b.setAttribute('aria-pressed', b.dataset.id === S.ds));
  const opt = (v, t, cur) => el('option', { value: v, selected: String(v) === String(cur) }, t);

  // scene strip
  const thumbs = ds.thumbnails ? `background-image:url(data/${ds.id}/${ds.thumbnails});` : '';
  const strip = el('div', { class: 'strip', id: 'strip', style: `--n:${ds.scenes}` }, Array.from({ length: ds.scenes }, (_, i) => {
    const r = ds.results, lab = sceneLabel(ds, i), std = (ds.standard_scenes || []).includes(i);
    const tip = [`장면 ${i}${lab ? ' · ' + lab : ''}`, ...vis.map(m => `${m.short} ${num(r[m.id].scene_metrics[i].PSNR, 2)} dB`)].join('\n');
    return el('button', { class: 'shot', type: 'button', title: tip, 'data-i': i, onclick: () => setScene(i) },
      el('span', { class: 'th', style: `--i:${i};${thumbs}` }, el('span', { class: 'no' + (std ? ' std' : '') }, i)),
      cand ? delta(r[cand.id].scene_metrics[i].PSNR, r[ref.id].scene_metrics[i].PSNR, true, 2, 'dv') : el('span', { class: 'dv' }, num(r[ref.id].scene_metrics[i].PSNR, 2)));
  }));
  const stripNote = (cand ? `숫자는 PSNR 차이 (${cand.short} − ${ref.short}, dB)` : `숫자는 ${ref.short} PSNR (dB)`) + ((ds.standard_scenes || []).length ? ' · 밑줄 번호는 표준 결과 이미지 장면' : '');

  // toolbar
  const settings = el('details', { class: 'pop', id: 'pop', open: S.popOpen, ontoggle: e => { S.popOpen = e.target.open; } }, el('summary', {}, '표시 설정'),
    el('div', { class: 'pop-body' },
      el('span', { class: 'lab' }, '비교 기준'), el('select', { onchange: e => { S.ref = e.target.value; if (S.cand === S.ref) S.cand = S.visible.find(v => v !== S.ref) || null; build(); } }, vis.map(m => opt(m.id, m.label, S.ref))),
      others.length > 1 ? [el('span', { class: 'lab' }, '비교 대상'), el('select', { onchange: e => { S.cand = e.target.value; build(); } }, others.map(m => opt(m.id, m.label, S.cand)))] : null,
      el('span', { class: 'lab' }, '오차 밴드'), el('select', { onchange: e => { S.errBand = e.target.value; renderAll(); } }, opt('mean', '전체 밴드 평균', S.errBand), ds.band_names.map((n, b) => opt(b, `${b} · ${n}`, S.errBand))),
      el('span', { class: 'lab' }, '오차 색 범위'), el('div', { class: 'pair' },
        el('select', { onchange: e => { S.errScale = e.target.value; build(); } }, opt('ref', '기준 모델 99%', S.errScale), opt('max', '모델 중 최대 99%', S.errScale), opt('manual', '직접 입력', S.errScale)),
        S.errScale === 'manual' ? el('input', { type: 'number', min: 1, step: 1, value: S.errManual, title: 'DN', style: 'width:84px', onchange: e => { S.errManual = Math.max(0.1, +e.target.value || 1); renderAll(); } }) : null),
      el('span', { class: 'lab' }, '확대 색 범위'), el('select', { onchange: e => { S.zoomScale = e.target.value; renderZoom(); } }, opt('scene', '전체 장면과 같게', S.zoomScale), opt('region', '확대 영역에 맞춤', S.zoomScale)),
      el('span', { class: 'lab' }, '확대 밝기'), el('select', { onchange: e => { S.zoomStretch = e.target.value; renderZoom(); } }, opt('scene', '전체 장면과 같게', S.zoomStretch), opt('region', '확대 영역에 맞춤', S.zoomStretch)),
      el('span', { class: 'lab' }, '입력 영상'), el('label', { class: 'check' }, el('input', { type: 'checkbox', checked: S.inputs, onchange: e => { S.inputs = e.target.checked; build(); } }), 'LR MS · PAN 함께 보기')));
  const bar = el('div', { class: 'bar' },
    el('div', { class: 'scene-title', id: 'sceneTitle' }),
    el('span', { class: 'group' }, all.map(m => el('label', { class: 'check' }, el('input', { type: 'checkbox', checked: S.visible.includes(m.id), onchange: e => toggleModel(m.id, e.target.checked) }), m.label))),
    el('span', { class: 'spacer' }),
    Object.keys(ds.composites).length > 1 ? el('span', { class: 'group' }, el('span', { class: 'lab' }, '밴드'), seg(Object.keys(ds.composites).map(k => [k, k]), S.comp, v => { S.comp = v; build(); })) : null,
    el('span', { class: 'group' }, el('span', { class: 'lab' }, '확대 영역'), seg(WINDOWS.map(w => [w, w]), S.win, setWindow)),
    cand ? el('span', { class: 'group' }, el('span', { class: 'lab' }, '영역 찾기'),
      el('button', { class: 'btn', type: 'button', title: `${cand.short}의 오차가 ${ref.short}보다 가장 많이 줄어든 곳`, onclick: () => autoWindow(1) }, '개선 최대'),
      el('button', { class: 'btn', type: 'button', title: `${cand.short}의 오차가 ${ref.short}보다 가장 많이 늘어난 곳`, onclick: () => autoWindow(-1) }, '악화 최대')) : null,
    settings);

  // layers: one column per image, whole scene on top and the selected region below
  tiles = [];
  if (S.inputs) tiles.push({ kind: 'lr', title: 'LR MS', sub: '입력 · 최근접 확대' }, { kind: 'pan', title: 'PAN', sub: '입력' });
  vis.forEach(m => tiles.push({ kind: 'pred', id: m.id, title: m.label, tag: m.id === S.ref && vis.length > 1 ? '기준' : null }));
  tiles.push({ kind: 'gt', title: '정답', sub: 'GT' });
  vis.forEach(m => tiles.push({ kind: 'err', id: m.id, title: `${m.short} 오차`, sub: '|예측 − 정답|' }));
  others.forEach(m => tiles.push({ kind: 'gain', id: m.id, title: `${m.short} 개선`, sub: `${ref.short} 오차 − ${m.short} 오차` }));
  const layers = el('div', { class: 'layers', id: 'layers' }, tiles.map(layer));

  // about
  const about = el('details', { class: 'about' }, el('summary', {}, '모델과 수치 설명'),
    el('dl', {}, all.map(m => [el('dt', {}, m.label), el('dd', {}, m.description || '', el('br'), el('code', {}, ds.results[m.id].checkpoint || ds.results[m.id].predictions || ''))])),
    el('ul', {},
      el('li', {}, '장면과 데이터셋 수치는 export.py가 원래 예측으로 계산했습니다(common/metrics.rr_metrics, PSNR은 peak 2047의 밴드별 평균).'),
      el('li', {}, `확대 영역 수치는 브라우저가 1/${factor} DN 단위로 저장한 값으로 계산합니다. 장면 전체에서 두 계산의 차이는 PSNR 0.0004 dB 이내였습니다.`),
      el('li', {}, 'seed 42 가중치 하나의 결과입니다. 실험 판정은 연구 문서의 시드 평균을 따릅니다.'),
      el('li', {}, `데이터 생성 ${S.index.generated}`)));

  app.replaceChildren(
    el('p', { class: 'ds-note' }, el('b', {}, ds.title), ds.note),
    el('div', { class: 'strip-head' }, el('span', {}, `장면 ${ds.scenes}개`), el('span', {}, stripNote)), strip,
    bar, layers,
    el('p', { class: 'hint' }, '영상을 누르거나 끌어서 확대 영역을 옮깁니다 · 방향키 1픽셀, Shift+방향키 8픽셀 · [ ] 이전·다음 장면'),
    el('div', { id: 'readout' }),
    el('div', { class: 'scroll', id: 'numbers' }),
    about);
  wire(); layoutCols(); renderAll();
}

function layoutCols() {
  const grid = $('#layers'); if (!grid) return;
  const width = grid.parentElement.clientWidth - 40;
  let cols = tiles.length;
  while (cols > 2 && width / cols < MIN_CELL) cols = Math.ceil(cols / 2);
  grid.style.setProperty('--cols', cols);
}
addEventListener('resize', layoutCols);

// ---------- rendering ----------
function errVmax(sc) {
  if (S.errScale === 'manual') return S.errManual;
  if (S.errScale === 'max') return Math.max(...S.visible.map(id => memo(sc, `p99:${id}:${S.errBand}`, () => percentile(errMap(sc, id), 99))));
  return memo(sc, `p99:${S.ref}:${S.errBand}`, () => percentile(errMap(sc, S.ref), 99));
}
function gainVmax(sc, region) {
  const ids = S.visible.filter(v => v !== S.ref);
  const one = id => { const g = gainMap(sc, id); return percentile(Float32Array.from(region ? sub(g, sc.H, sc.W, 0, ...region) : g, Math.abs), 99); };
  return Math.max(1e-6, ...ids.map(id => (region ? one(id) : memo(sc, `g99:${S.ref}:${id}:${S.errBand}`, () => one(id)))));
}

function renderAll() {
  const sc = current; if (!sc) return;
  const ds = dataset(S.ds), bands = ds.composites[S.comp], st = stretchOf(sc, bands), { H, W } = sc, res = ds.results;
  const vmax = errVmax(sc), gmax = gainVmax(sc), refPsnr = res[S.ref].scene_metrics[S.scene].PSNR;
  for (const t of tiles) {
    if (t.kind === 'lr') { paintLR(t.full, sc, bands, st, 0, 0, H, W); setCap(t.cap1, `${sc.h}×${sc.h}`); }
    else if (t.kind === 'pan') { paintGray(t.full, sc.pan, W, panStretch(sc), 0, 0, H, W); setCap(t.cap1, `${W}×${H}`); }
    else if (t.kind === 'gt') { paintRGB(t.full, sc.gt, sc, bands, st, 0, 0, H, W); setCap(t.cap1, `${W}×${H} · ${S.comp} ${bands.join(' ')}`); }
    else if (t.kind === 'pred') {
      paintRGB(t.full, sc.pred[t.id], sc, bands, st, 0, 0, H, W);
      const p = res[t.id].scene_metrics[S.scene].PSNR;
      setCap(t.cap1, ['장면 ', el('b', {}, num(p, 2)), ' dB'], t.id === S.ref ? null : delta(p, refPsnr, true, 2, ''));
    } else if (t.kind === 'err') {
      const m = errMap(sc, t.id); paintMap(t.full, m, W, vmax, 'heat', 0, 0, H, W);
      setCap(t.cap1, ['평균 ', el('b', {}, num(memo(sc, `mean:${t.id}:${S.errBand}`, () => mean(m)), 1)), ' DN'], scaleEl('heat', `0-${dn(vmax)}`));
    } else {
      const g = gainMap(sc, t.id); paintMap(t.full, g, W, gmax, 'div', 0, 0, H, W);
      setCap(t.cap1, ['평균 ', signed(memo(sc, `gmean:${S.ref}:${t.id}:${S.errBand}`, () => mean(g)), 1), ' DN'], scaleEl('div', `±${dn(gmax)}`));
    }
  }
  const lab = sceneLabel(ds, S.scene);
  $('#sceneTitle').replaceChildren(`장면 ${S.scene}`, lab ? el('span', {}, lab) : '');
  document.querySelectorAll('#strip .shot').forEach(b => b.setAttribute('aria-current', +b.dataset.i === S.scene));
  const cur = $(`#strip .shot[data-i="${S.scene}"]`), strip = $('#strip');
  if (cur && (cur.offsetLeft < strip.scrollLeft || cur.offsetLeft + cur.offsetWidth > strip.scrollLeft + strip.clientWidth)) strip.scrollLeft = cur.offsetLeft - strip.clientWidth / 2;
  renderZoom();
}

let zoomQueued = false;
function renderZoom() {
  if (zoomQueued) return; zoomQueued = true;
  const run = () => { if (!zoomQueued) return; zoomQueued = false; drawZoom(); };
  requestAnimationFrame(run); setTimeout(run, 60);   // the timer covers tabs where frames are paused
}
function drawZoom() {
  const sc = current; if (!sc || !tiles.length) return;
  const ds = dataset(S.ds), bands = ds.composites[S.comp], { H, W } = sc, w = S.win, r0 = S.y, c0 = S.x, reg = [r0, c0, w, w];
  const fit = S.zoomStretch === 'region', st = fit ? stretchOf(sc, bands, reg) : stretchOf(sc, bands);
  const own = S.zoomScale === 'region';
  const vmax = own ? Math.max(1e-6, percentile(sub(errMap(sc, S.ref), H, W, 0, ...reg), 99)) : errVmax(sc);
  const gmax = own ? gainVmax(sc, reg) : gainVmax(sc);
  const rm = Object.fromEntries(S.visible.map(id => [id, regionMetrics(sc, id, r0, c0, w)]));
  for (const t of tiles) {
    if (t.kind === 'lr') { paintLR(t.zoom, sc, bands, st, r0, c0, w, w); setCap(t.cap2, ''); }
    else if (t.kind === 'pan') { paintGray(t.zoom, sc.pan, W, fit ? panStretch(sc, reg) : panStretch(sc), r0, c0, w, w); setCap(t.cap2, ''); }
    else if (t.kind === 'gt') { paintRGB(t.zoom, sc.gt, sc, bands, st, r0, c0, w, w); setCap(t.cap2, `행 ${r0}-${r0 + w - 1} · 열 ${c0}-${c0 + w - 1}`); }
    else if (t.kind === 'pred') {
      paintRGB(t.zoom, sc.pred[t.id], sc, bands, st, r0, c0, w, w);
      setCap(t.cap2, ['영역 ', el('b', {}, num(rm[t.id].PSNR, 2)), ' dB'], t.id === S.ref ? null : delta(rm[t.id].PSNR, rm[S.ref].PSNR, true, 2, ''));
    } else if (t.kind === 'err') {
      const m = errMap(sc, t.id); paintMap(t.zoom, m, W, vmax, 'heat', r0, c0, w, w);
      setCap(t.cap2, ['평균 ', el('b', {}, num(mean(sub(m, H, W, 0, ...reg)), 1)), ' DN'], own ? scaleEl('heat', `0-${dn(vmax)}`) : null);
    } else {
      const g = gainMap(sc, t.id); paintMap(t.zoom, g, W, gmax, 'div', r0, c0, w, w);
      setCap(t.cap2, ['평균 ', signed(mean(sub(g, H, W, 0, ...reg)), 1), ' DN'], own ? scaleEl('div', `±${dn(gmax)}`) : null);
    }
  }
  document.querySelectorAll('.box').forEach(b => { b.style.left = `${c0 / W * 100}%`; b.style.top = `${r0 / H * 100}%`; b.style.width = `${w / W * 100}%`; b.style.height = `${w / H * 100}%`; });
  renderNumbers(rm);
  syncURL();
}

const REGION_COLS = [['PSNR', 'PSNR dB', true, 2], ['SAM', 'SAM °', false, 3], ['ERGAS', 'ERGAS', false, 3], ['MAE', '평균 오차 DN', false, 2]];
const SCENE_COLS = [['PSNR', 'PSNR dB', true, 2], ['SAM', 'SAM °', false, 3], ['ERGAS', 'ERGAS', false, 3], ['SCC', 'SCC', true, 4], ['Q2n', 'Q2n', true, 4]];
const MEAN_COLS = SCENE_COLS.slice(0, 3);
function renderNumbers(rm) {
  const ds = dataset(S.ds), res = ds.results, ref = S.ref, multi = S.visible.length > 1;
  const cells = (cols, get, id) => cols.map(([k, , hb, dg], i) => el('td', { class: i === 0 ? 'g' : '' }, num(get(id)[k], dg), multi && id !== ref ? delta(get(id)[k], get(ref)[k], hb, dg) : null));
  const heads = cols => cols.map(([, t], i) => el('th', { class: i === 0 ? 'g' : '' }, t));
  const wins = id => res[id].scene_metrics.filter((r, i) => r.PSNR > res[ref].scene_metrics[i].PSNR).length;
  const table = el('table', { class: 'numbers' },
    el('thead', {},
      el('tr', { class: 'groups' }, el('th', {}), el('th', { class: 'g', colspan: REGION_COLS.length }, `확대 영역 ${S.win}×${S.win}`), el('th', { class: 'g', colspan: SCENE_COLS.length }, `장면 ${S.scene} 전체`),
        el('th', { class: 'g', colspan: MEAN_COLS.length + (multi ? 1 : 0) }, `데이터셋 평균 · ${ds.scenes}장면`)),
      el('tr', {}, el('th', {}, '모델'), heads(REGION_COLS), heads(SCENE_COLS), heads(MEAN_COLS), multi ? el('th', { title: '비교 기준보다 PSNR이 높은 장면 수' }, 'PSNR 우세 장면') : null)),
    el('tbody', {}, S.visible.map(id => el('tr', {}, el('td', {}, model(id).label, multi && id === ref ? [' ', el('span', { class: 'tag' }, '기준')] : null),
      cells(REGION_COLS, i => rm[i], id), cells(SCENE_COLS, i => res[i].scene_metrics[S.scene], id), cells(MEAN_COLS, i => res[i].mean, id),
      multi ? el('td', {}, id === ref ? '' : `${wins(id)}/${ds.scenes}`) : null))));
  $('#numbers').replaceChildren(table);
}

// ---------- interaction ----------
function clampBox() {
  const n = current ? current.H : 256;
  S.win = Math.min(S.win, n); S.x = Math.max(0, Math.min(n - S.win, Math.round(S.x))); S.y = Math.max(0, Math.min(n - S.win, Math.round(S.y)));
}
function setWindow(w) {
  const cx = S.x + S.win / 2, cy = S.y + S.win / 2; S.win = w; S.x = cx - w / 2; S.y = cy - w / 2; clampBox();
  document.querySelectorAll('.bar .seg').forEach(s => { if ([...s.children].some(b => b.textContent === String(w))) [...s.children].forEach(b => b.setAttribute('aria-pressed', b.textContent === String(w))); });
  renderZoom();
}
function autoWindow(sign) { if (!current || !S.cand) return; [S.y, S.x] = bestWindow(current, sign); renderZoom(); }

function wire() {
  for (const t of tiles) for (const [frame, zoom] of [[t.fullFrame, false], [t.zoomFrame, true]]) {
    const pos = e => { const r = frame.getBoundingClientRect(); return [(e.clientY - r.top - frame.clientTop) / frame.clientHeight, (e.clientX - r.left - frame.clientLeft) / frame.clientWidth]; };
    let off = null;
    frame.addEventListener('pointermove', e => {
      if (!current) return;
      const [v, u] = pos(e), n = current.H;
      if (off && frame.hasPointerCapture(e.pointerId)) { S.y = v * n - off[0]; S.x = u * n - off[1]; clampBox(); renderZoom(); }
      if (v < 0 || v >= 1 || u < 0 || u >= 1) return hover(null);
      hover(zoom ? S.y + Math.floor(v * S.win) : Math.floor(v * n), zoom ? S.x + Math.floor(u * S.win) : Math.floor(u * n));
    });
    frame.addEventListener('pointerleave', () => hover(null));
    if (zoom) continue;
    frame.addEventListener('pointerdown', e => {
      if (!current) return;
      const [v, u] = pos(e), n = current.H, py = v * n, px = u * n;
      off = py >= S.y && py < S.y + S.win && px >= S.x && px < S.x + S.win ? [py - S.y, px - S.x] : [S.win / 2, S.win / 2];
      S.y = py - off[0]; S.x = px - off[1]; clampBox(); renderZoom();
      frame.setPointerCapture(e.pointerId); e.preventDefault();
    });
    const end = () => { off = null; };
    frame.addEventListener('pointerup', end); frame.addEventListener('pointercancel', end);
  }
}

function hover(row, col) {
  const out = $('#readout'), inside = row != null && current && row >= S.y && row < S.y + S.win && col >= S.x && col < S.x + S.win;
  for (const t of tiles) {
    const v = $('.vl', t.zoomFrame), h = $('.hl', t.zoomFrame);
    v.style.display = h.style.display = inside ? 'block' : 'none';
    if (inside) { v.style.left = `${(col - S.x + 0.5) / S.win * 100}%`; h.style.top = `${(row - S.y + 0.5) / S.win * 100}%`; }
  }
  if (!out) return;
  if (row == null || !current || row < 0 || col < 0 || row >= current.H || col >= current.W) { out.textContent = ''; return; }
  const sc = current, ds = dataset(S.ds), N = sc.H * sc.W, o = row * sc.W + col, bands = ds.composites[S.comp];
  const gt = bands.map(b => sc.gt[b * N + o].toFixed(0)).join(' ');
  const errs = S.visible.map(id => `${model(id).short} 오차 ${errMap(sc, id)[o].toFixed(1)}`).join('   ');
  out.textContent = `행 ${row} 열 ${col}   정답 ${gt}   ${errs} DN`;
}

document.addEventListener('keydown', e => {
  if (!current || e.metaKey || e.ctrlKey || e.altKey || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
  const n = dataset(S.ds).scenes;
  if (e.key === '[') return setScene((S.scene + n - 1) % n);
  if (e.key === ']') return setScene((S.scene + 1) % n);
  const step = e.shiftKey ? 8 : 1, mv = { ArrowLeft: [0, -step], ArrowRight: [0, step], ArrowUp: [-step, 0], ArrowDown: [step, 0] }[e.key];
  if (!mv) return;
  e.preventDefault(); S.y += mv[0]; S.x += mv[1]; clampBox(); renderZoom();
});
document.addEventListener('click', e => { const pop = $('#pop'); if (pop && pop.open && !pop.contains(e.target)) pop.open = false; });

function toggleModel(id, on) {
  const order = modelsIn(dataset(S.ds)).map(m => m.id);
  S.visible = order.filter(m => (m === id ? on : S.visible.includes(m)));
  if (!S.visible.length) S.visible = [id];
  if (!S.visible.includes(S.ref)) S.ref = S.visible[0];
  if (!S.visible.includes(S.cand) || S.cand === S.ref) S.cand = S.visible.find(v => v !== S.ref) || null;
  build();
}

let loadToken = 0;
async function setScene(i, rebuild = false) {
  S.scene = i; const token = ++loadToken;
  const slow = setTimeout(() => { $('#loading').style.display = 'block'; }, 150);
  try {
    const sc = await loadScene(S.ds, i);
    if (token !== loadToken) return;
    current = sc; clampBox();
    if (rebuild || !$('#layers')) build(); else renderAll();
  } catch (err) {
    $('#app').replaceChildren(el('p', { class: 'empty' }, `장면 데이터를 읽지 못했습니다 (${err.message})`));
  } finally { clearTimeout(slow); if (token === loadToken) $('#loading').style.display = 'none'; }
}

function setDataset(id) {
  const ds = dataset(id); S.ds = id;
  const ids = modelsIn(ds).map(m => m.id);
  S.visible = S.visible.filter(v => ids.includes(v)); if (!S.visible.length) S.visible = ids.slice();
  if (!S.visible.includes(S.ref)) S.ref = S.visible[0];
  if (!S.visible.includes(S.cand) || S.cand === S.ref) S.cand = S.visible.find(v => v !== S.ref) || null;
  if (!ds.composites[S.comp]) S.comp = Object.keys(ds.composites)[0];
  if (S.errBand !== 'mean' && +S.errBand >= ds.bands) S.errBand = 'mean';
  current = null;
  setScene(S.scene < ds.scenes ? S.scene : 0, true);
}

let urlTimer = 0;
function syncURL() {
  clearTimeout(urlTimer);
  urlTimer = setTimeout(() => {
    const p = new URLSearchParams({ ds: S.ds, scene: S.scene, x: S.x, y: S.y, w: S.win, comp: S.comp, ref: S.ref, models: S.visible.join(',') });
    history.replaceState(null, '', '#' + p.toString());
  }, 200);
}
function readURL() {
  const p = new URLSearchParams(location.hash.slice(1));
  const n = (k, d) => (p.has(k) && !isNaN(+p.get(k)) ? +p.get(k) : d);
  return { ds: p.get('ds'), scene: n('scene', 0), x: n('x', S.x), y: n('y', S.y), w: n('w', S.win), comp: p.get('comp'), ref: p.get('ref'), models: p.get('models') };
}

// ---------- theme ----------
const THEMES = { auto: '화면 자동', light: '밝은 화면', dark: '어두운 화면' };
function applyTheme() {
  if (S.theme === 'auto') delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = S.theme;
  $('#theme').textContent = THEMES[S.theme];
}
$('#theme').addEventListener('click', () => {
  const keys = Object.keys(THEMES); S.theme = keys[(keys.indexOf(S.theme) + 1) % keys.length];
  try { localStorage.setItem('viewer-theme', S.theme); } catch (err) { /* private window */ }
  applyTheme();
});
try { const t = localStorage.getItem('viewer-theme'); if (THEMES[t]) S.theme = t; } catch (err) { /* private window */ }
applyTheme();

function applyURL() {                              // also runs when a link is pasted into an open tab
  const u = readURL(), ready = S.index.datasets.filter(d => d.scenes && Object.keys(d.results || {}).length);
  S.x = u.x; S.y = u.y; S.win = WINDOWS.includes(u.w) ? u.w : S.win; if (u.comp) S.comp = u.comp;
  if (u.models) { const ids = u.models.split(',').filter(id => model(id)); if (ids.length) S.visible = ids; }
  if (u.ref && model(u.ref)) S.ref = u.ref;
  S.scene = u.scene;
  setDataset(ready.some(d => d.id === u.ds) ? u.ds : S.ds || ready[0].id);
}
addEventListener('hashchange', () => { if (S.index) applyURL(); });

async function boot() {
  let index;
  try {
    const r = await fetch('data/index.json', { cache: 'no-cache' });
    if (!r.ok) throw new Error(r.status);
    index = await r.json();
  } catch (err) {
    $('#app').replaceChildren(el('div', { class: 'empty' },
      el('p', {}, '예측 데이터(data/index.json)가 없습니다. 아래 두 명령으로 데이터를 만들고 이 폴더를 엽니다.'),
      el('p', {}, el('code', {}, 'python SSA-MRN/viewer/export.py')), el('p', {}, el('code', {}, 'python -m http.server 8765 -d SSA-MRN/viewer'))));
    return;
  }
  S.index = index; factor = index.storage.factor;
  const ready = index.datasets.filter(d => d.scenes && Object.keys(d.results || {}).length);
  $('#dsTabs').replaceChildren(...ready.map(d => el('button', { type: 'button', 'data-id': d.id, 'aria-pressed': 'false', onclick: () => setDataset(d.id) }, d.label)));
  applyURL();
}
boot();
