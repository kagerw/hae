// ハエにマウンテンカー覚えさせてみた - the page.
//
// The environment and the reward run here, line for line as mountaincar/train.py
// runs them (gymnasium's MountainCar-v0 and MountainCarEnergyReward); the brain
// runs in worker.js. One step: the worker looks at the state and picks an action,
// the car moves, and the reward goes back to the worker as dopamine.

// ------------------------------------------------------------------ environment
// gymnasium.envs.classic_control.MountainCarEnv (v0), with its float32 state
const f32 = Math.fround;
class MountainCarEnv {
  constructor() {
    this.min_position = -1.2; this.max_position = 0.6; this.max_speed = 0.07;
    this.goal_position = 0.5; this.goal_velocity = 0;
    this.force = 0.001; this.gravity = 0.0025;
    this.state = null;
  }
  reset(rnd) {
    this.state = [-0.6 + rnd() * 0.2, 0];
    return this.state.map(f32);
  }
  step(action) {
    let [position, velocity] = this.state;
    velocity += (action - 1) * this.force + Math.cos(3 * position) * -this.gravity;
    velocity = Math.min(Math.max(velocity, -this.max_speed), this.max_speed);
    position += velocity;
    position = Math.min(Math.max(position, this.min_position), this.max_position);
    if (position === this.min_position && velocity < 0) velocity = 0;
    const terminated = position >= this.goal_position && velocity >= this.goal_velocity;
    this.state = [f32(position), f32(velocity)];
    return [this.state.slice(), -1, terminated, false, {}];
  }
}

// gymnasium's TimeLimit (MountainCar-v0: 200 steps)
class TimeLimit {
  constructor(env, max) { this.env = env; this.max = max; this.t = 0; }
  get unwrapped() { return this.env; }
  reset(rnd) { this.t = 0; return this.env.reset(rnd); }
  step(a) {
    const r = this.env.step(a);
    if (++this.t >= this.max) r[3] = true;
    return r;
  }
}

// MountainCarEnergyReward - the reward the task is learned from, as given
class MountainCarEnergyReward {
  constructor(env, ke_scale = 1.0, goal_bonus = 50.0) {
    this.env = env; this.prev_energy = null; this.g = 9.8;
    this.ke_scale = ke_scale; this.goal_bonus = goal_bonus;
  }
  reset(rnd) {
    const obs = this.env.reset(rnd);
    this.prev_energy = this.compute_energy(obs);
    return obs;
  }
  step(action) {
    const [obs, _orig, terminated, truncated, info] = this.env.step(action);
    const current_energy = this.compute_energy(obs);
    let shaped = 10000.0 * (current_energy - this.prev_energy);
    this.prev_energy = current_energy;
    if (terminated && obs[0] >= this.env.unwrapped.goal_position) {
      shaped += this.goal_bonus;
      info.reached_goal = true;
    }
    return [obs, shaped, terminated, truncated, info];
  }
  compute_energy([position, velocity]) {
    const kinetic = 0.5 * velocity ** 2;
    const potential = (0.0025 / 3.0) * Math.sin(3.0 * position);
    return kinetic + potential;
  }
}

function mulberry32(a) {
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// ------------------------------------------------------------------ the brain
const worker = new Worker(new URL('./worker.js?v=1', import.meta.url), { type: 'module' });
const waiting = [];
worker.onmessage = ({ data: m }) => {
  if (m.type === 'progress') return progress(m);
  if (m.type === 'error') { fail(m.message); return; }
  const w = waiting.shift();
  if (w) w(m);
};
worker.onerror = (e) => fail(e.message || 'worker error');
const ask = (m) => new Promise((res) => { waiting.push(res); worker.postMessage(m); });

// ------------------------------------------------------------------ page
const $ = (id) => document.getElementById(id);
const ACT = ['左に押す', '押さない', '右に押す'];
const COL = ['#59b7ff', '#5b6673', '#ffb454'];
const STREAK = 10, EPS0 = 0.1, EPS_DECAY = 0.97;
const PX = 14, PY = 12;             // the policy map's grid

const env = new MountainCarEnergyReward(new TimeLimit(new MountainCarEnv(), 200));
let rnd = mulberry32(20260923);
let running = false, busy = false;
let st;                               // statistics over the episodes
let ep;                               // the episode in progress
let policy = null, lastTrail = [], lastDrive = [1, 1, 1], lastGains = [1, 1, 1];

function freshStats() {
  return { episodes: [], cumSteps: 0, first: null, mastered: null, streak: 0, goals: 0 };
}

function setStatus(t, err = false) { const s = $('status'); s.textContent = t; s.className = err ? 'err' : ''; }
function fail(msg) { running = false; setStatus('エラー: ' + msg, true); console.error(msg); }
function progress(p) {
  if (p.phase === 'download' && p.total) setStatus(`脳の配線をダウンロード中… ${(p.loaded / 1e6).toFixed(1)} / ${(p.total / 1e6).toFixed(1)} MB`);
  else setStatus(p.phase === 'decompress' ? '配線を展開中…' : 'キノコ体を組み立て中…');
}

function newEpisode() {
  const n = st.episodes.length + 1;
  const obs = env.reset(rnd);
  ep = { n, obs, steps: 0, ret: 0, maxPos: obs[0], acts: [0, 0, 0], along: 0,
    eps: EPS0 * EPS_DECAY ** (n - 1), trail: [obs.slice()], lastAct: null, lastRew: null, done: false };
}

async function step() {
  const look = await ask({ type: 'look', obs: ep.obs, eps: ep.eps });
  const a = look.action;
  lastDrive = look.drive;
  ep.acts[a]++;
  if (a === (ep.obs[1] >= 0 ? 2 : 0)) ep.along++;
  const [obs, reward, terminated, truncated, info] = env.step(a);
  if ($('learnOn').checked) lastGains = (await ask({ type: 'learn', action: a, reward })).gains;
  ep.obs = obs; ep.steps++; ep.ret += reward; ep.maxPos = Math.max(ep.maxPos, obs[0]);
  ep.lastAct = a; ep.lastRew = reward; ep.trail.push(obs.slice());
  if (terminated || truncated) { ep.done = true; ep.goal = !!info.reached_goal; }
}

async function endEpisode() {
  const e = { n: ep.n, steps: ep.steps, goal: ep.goal, maxPos: ep.maxPos, ret: ep.ret,
    along: ep.along / ep.steps, eps: ep.eps, learned: $('learnOn').checked };
  st.episodes.push(e);
  st.cumSteps += e.steps;
  st.streak = e.goal ? st.streak + 1 : 0;
  if (e.goal) st.goals++;
  if (e.goal && !st.first) st.first = { n: e.n, cum: st.cumSteps, steps: e.steps };
  if (st.streak >= STREAK && !st.mastered) st.mastered = { n: e.n, cum: st.cumSteps };
  lastTrail = ep.trail;
  addLog(e);
  drawStats(); drawChart();
  policy = await ask({ type: 'policy', nx: PX, ny: PY });
  drawPolicy();
}

// The learning loop is paced by the brain (each step waits for the worker) and,
// unless fast, by a pause per step; drawing runs on its own frame loop, so a
// background tab (no frames) does not stop the learning.
const PAUSE = { slow: 150, normal: 35, fast: 0 };
let dirty = true;
async function loop() {
  if (busy) return;
  busy = true;
  try {
    while (running) {
      if (!ep || ep.done) newEpisode();
      await step();
      dirty = true;
      if (ep.done) {
        await endEpisode();
        if ($('speed').value !== 'fast') await new Promise((r) => setTimeout(r, 600));   // see the finish
      }
      const ms = PAUSE[$('speed').value];
      if (ms) await new Promise((r) => setTimeout(r, ms));
    }
  } finally { busy = false; }
}
(function frame() {
  if (dirty) { dirty = false; drawWorld(); drawReadout(); drawComp(); }
  requestAnimationFrame(frame);
})();

// ------------------------------------------------------------------ drawing
function fit(canvas) {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = +canvas.getAttribute('height');
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); canvas.style.height = h + 'px';
  }
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [g, w, h];
}

function drawWorld() {
  const [g, w, h] = fit($('world'));
  g.clearRect(0, 0, w, h);
  const x0 = -1.2, x1 = 0.6, pad = 18;
  const sx = (x) => pad + (x - x0) / (x1 - x0) * (w - 2 * pad);
  const sy = (y) => h - 36 - (y + 1) / 2 * (h - 80);
  const hill = (x) => Math.sin(3 * x);
  // ground
  g.beginPath();
  g.moveTo(sx(x0), h);
  for (let i = 0; i <= 200; i++) { const x = x0 + (x1 - x0) * i / 200; g.lineTo(sx(x), sy(hill(x))); }
  g.lineTo(sx(x1), h); g.closePath();
  g.fillStyle = '#18222c'; g.fill();
  g.beginPath();
  for (let i = 0; i <= 200; i++) { const x = x0 + (x1 - x0) * i / 200; const px = sx(x), py = sy(hill(x)); i ? g.lineTo(px, py) : g.moveTo(px, py); }
  g.strokeStyle = '#3d5063'; g.lineWidth = 2; g.stroke();
  // flag
  const fx = sx(0.5), fy = sy(hill(0.5));
  g.strokeStyle = '#c9d4de'; g.lineWidth = 2;
  g.beginPath(); g.moveTo(fx, fy); g.lineTo(fx, fy - 38); g.stroke();
  g.fillStyle = '#54d98c';
  g.beginPath(); g.moveTo(fx, fy - 38); g.lineTo(fx + 20, fy - 32); g.lineTo(fx, fy - 26); g.closePath(); g.fill();
  if (!ep) return;
  // this episode's highest point
  if (ep.maxPos > -1.2) {
    const mx = sx(ep.maxPos);
    g.strokeStyle = 'rgba(255,180,84,.35)'; g.setLineDash([4, 4]); g.lineWidth = 1;
    g.beginPath(); g.moveTo(mx, sy(hill(ep.maxPos)) - 4); g.lineTo(mx, 22); g.stroke(); g.setLineDash([]);
  }
  // the car, tilted with the slope
  const [p] = ep.obs;
  const cx = sx(p), cy = sy(hill(p));
  const slope = 3 * Math.cos(3 * p) * ((h - 80) / 2) / ((w - 2 * pad) / (x1 - x0));
  const ang = -Math.atan(slope);
  g.save(); g.translate(cx, cy); g.rotate(ang);
  g.fillStyle = '#e8eef4';
  g.fillRect(-20, -22, 40, 14);
  g.fillStyle = '#9fb3c6';
  g.fillRect(-10, -30, 18, 9);
  g.fillStyle = '#0b0e12';
  for (const wx of [-12, 12]) { g.beginPath(); g.arc(wx, -6, 6, 0, Math.PI * 2); g.fill(); g.strokeStyle = '#8c99a8'; g.lineWidth = 2; g.stroke(); }
  // the push
  if (ep.lastAct != null && ep.lastAct !== 1) {
    const d = ep.lastAct === 2 ? 1 : -1;
    g.strokeStyle = COL[ep.lastAct]; g.fillStyle = COL[ep.lastAct]; g.lineWidth = 3;
    g.beginPath(); g.moveTo(d * 24, -15); g.lineTo(d * 44, -15); g.stroke();
    g.beginPath(); g.moveTo(d * 50, -15); g.lineTo(d * 42, -21); g.lineTo(d * 42, -9); g.closePath(); g.fill();
  }
  g.restore();
  g.fillStyle = '#8c99a8'; g.font = '12px system-ui, sans-serif';
  g.fillText(`試行 ${ep.n}  ステップ ${ep.steps}`, 12, 18);
  if (ep.done) {
    g.font = '600 18px system-ui, sans-serif';
    g.fillStyle = ep.goal ? '#54d98c' : '#ff7b7b';
    g.fillText(ep.goal ? `ゴール！ ${ep.steps} ステップ` : '時間切れ', 12, 42);
  }
}

function drawReadout() {
  if (!ep) return;
  $('rStep').textContent = ep.steps;
  $('rPos').textContent = ep.obs[0].toFixed(3);
  $('rVel').textContent = (ep.obs[1] >= 0 ? '+' : '') + ep.obs[1].toFixed(4);
  $('rAct').textContent = ep.lastAct == null ? '—' : ACT[ep.lastAct];
  $('rAct').style.color = ep.lastAct == null ? '' : COL[ep.lastAct];
  $('rRew').textContent = ep.lastRew == null ? '—' : (ep.lastRew >= 0 ? '+' : '') + ep.lastRew.toFixed(3);
  $('rRet').textContent = ep.ret.toFixed(2);
}

function drawComp() {
  const el = $('comp');
  if (!el.children.length) {
    el.innerHTML = ACT.map((a, i) => `<span style="color:${COL[i]}">${a}</span><div class="track"><div class="fill" id="cf${i}" style="background:${COL[i]}"></div><div class="mid"></div></div><span id="cg${i}">1.000</span>`).join('');
  }
  const lo = Math.min(...lastDrive);
  lastDrive.forEach((d, i) => {
    const f = $('cf' + i);
    f.style.width = Math.max(0, Math.min(100, d / 2 * 100)) + '%';
    f.style.opacity = d === lo ? 1 : 0.45;
    $('cg' + i).textContent = lastGains[i].toFixed(3);
  });
}

function drawStats() {
  const n = st.episodes.length;
  $('sEp').textContent = n;
  $('sEpSub').textContent = `累計 ${st.cumSteps.toLocaleString()} ステップ`;
  $('sFirst').textContent = st.first ? `${st.first.n} 回目` : '—';
  $('sFirstSub').textContent = st.first ? `累計 ${st.first.cum.toLocaleString()} ステップ目（その回 ${st.first.steps} ステップ）` : 'まだ';
  $('cardFirst').classList.toggle('hit', !!st.first);
  $('sMaster').textContent = st.mastered ? `${st.mastered.n} 回目` : '—';
  $('sMasterSub').textContent = st.mastered ? `累計 ${st.mastered.cum.toLocaleString()} ステップ` : `あと ${STREAK - Math.min(STREAK, st.streak)} 回連続`;
  $('cardMaster').classList.toggle('hit', !!st.mastered);
  $('sStreak').textContent = st.streak;
  $('sRate').textContent = `ゴール ${st.goals} / ${n} 回`;
}

function drawChart() {
  const [g, w, h] = fit($('chart'));
  g.clearRect(0, 0, w, h);
  const E = st.episodes, padL = 30, padB = 20, padT = 8;
  const N = Math.max(20, E.length);
  const bw = (w - padL - 6) / N;
  const sy = (s) => h - padB - s / 200 * (h - padB - padT);
  g.strokeStyle = '#232c36'; g.fillStyle = '#8c99a8'; g.font = '11px system-ui, sans-serif'; g.lineWidth = 1;
  for (const s of [0, 100, 200]) { g.beginPath(); g.moveTo(padL, sy(s)); g.lineTo(w - 4, sy(s)); g.stroke(); g.fillText(s, 4, sy(s) + 4); }
  E.forEach((e, i) => {
    g.fillStyle = e.goal ? (e.learned ? '#54d98c' : '#b69cff') : '#3a4552';
    g.fillRect(padL + i * bw + bw * 0.12, sy(e.steps), Math.max(1, bw * 0.76), sy(0) - sy(e.steps));
  });
  const mark = (m, col, label) => {
    if (!m) return;
    const x = padL + (m.n - 0.5) * bw;
    g.strokeStyle = col; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(x, padT); g.lineTo(x, sy(0)); g.stroke(); g.setLineDash([]);
    g.fillStyle = col; g.fillText(label, Math.min(x + 3, w - 60), padT + 10 + (label === '習得' ? 14 : 0));
  };
  mark(st.first, '#ffb454', '初成功');
  mark(st.mastered, '#54d98c', '習得');
  g.fillStyle = '#8c99a8';
  g.fillText('試行 →', w - 44, h - 4);
}

function drawPolicy() {
  const [g, w, h] = fit($('policy'));
  g.clearRect(0, 0, w, h);
  if (policy) {
    const cw = w / policy.nx, ch = h / policy.ny;
    policy.cells.forEach((c, k) => {
      const i = k % policy.nx, j = Math.floor(k / policy.nx);
      g.fillStyle = c.a < 0 ? '#0f1419' : COL[c.a];
      g.globalAlpha = c.a < 0 ? 1 : 0.35 + 0.65 * Math.min(1, c.margin / 0.02);
      g.fillRect(i * cw + 0.5, h - (j + 1) * ch + 0.5, cw - 1, ch - 1);
    });
    g.globalAlpha = 1;
  }
  const sx = (p) => (p + 1.2) / 1.8 * w, sy = (v) => h - (v + 0.07) / 0.14 * h;
  g.strokeStyle = 'rgba(255,255,255,.25)'; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, sy(0)); g.lineTo(w, sy(0)); g.stroke();
  g.beginPath(); g.moveTo(sx(0.5), 0); g.lineTo(sx(0.5), h); g.stroke();
  if (lastTrail.length) {
    g.strokeStyle = '#ffffff'; g.lineWidth = 1.5;
    g.beginPath();
    lastTrail.forEach(([p, v], i) => (i ? g.lineTo(sx(p), sy(v)) : g.moveTo(sx(p), sy(v))));
    g.stroke();
  }
  g.fillStyle = 'rgba(232,238,244,.75)'; g.font = '11px system-ui, sans-serif';
  g.fillText('位置 →', w - 42, h - 5); g.fillText('速度 ↑', 4, 12); g.fillText('ゴール', sx(0.5) + 3, 12);
}

function addLog(e) {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${e.n}</td><td class="${e.goal ? 'goal' : ''}">${e.goal ? 'ゴール' : '時間切れ'}${e.learned ? '' : '（学習なし）'}</td>` +
    `<td>${e.steps}</td><td>${e.maxPos.toFixed(3)}</td><td>${e.ret.toFixed(2)}</td><td>${Math.round(e.along * 100)}%</td><td>${e.eps.toFixed(3)}</td>`;
  $('log').prepend(tr);
}

// ------------------------------------------------------------------ controls
$('bRun').onclick = () => {
  running = !running;
  $('bRun').textContent = running ? '⏸ 一時停止' : '▶ 再開';
  if (running) { setStatus('学習中 — キノコ体が状態を見て行動を選び、報酬がドーパミンとして返ります。'); loop(); }
  else setStatus('一時停止中');
};
$('bForget').onclick = async () => {
  running = false;
  while (busy) await new Promise((r) => setTimeout(r, 20));
  const r = await ask({ type: 'forget' });
  lastGains = r.gains; lastDrive = [1, 1, 1];
  st = freshStats(); ep = null; policy = null; lastTrail = [];
  rnd = mulberry32(20260923);
  $('log').innerHTML = '';
  $('bRun').textContent = '▶ 学習開始';
  setStatus('シナプスの重みを学習前に戻しました。');
  drawAll();
};
$('learnOn').onchange = () => {
  worker.postMessage({ type: 'eta', eta: $('learnOn').checked ? 6e-5 : 0 });
};
window.addEventListener('resize', () => drawAll());

function drawAll() { drawWorld(); drawReadout(); drawComp(); drawStats(); drawChart(); drawPolicy(); }

st = freshStats();
drawAll();
ask({ type: 'init' }).then((m) => {
  const i = m.info;
  setStatus(`準備完了: FlyWire v783 の ${i.neurons.toLocaleString()} ニューロンのうちキノコ体（ケニヨン細胞 ${i.kc.toLocaleString()}、投射ニューロン ${i.pn}、MBON ${i.compartments.reduce((s, c) => s + c.mbons, 0)}）。「学習開始」を押してください。`);
  $('bRun').disabled = false; $('bForget').disabled = false;
});
