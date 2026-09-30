// ハエと三目並べ - the page.
//
// The board is 9 cells, 0 = empty, FLY = the fly's ○, OPP = the opponent's ×.
// Every player is { name, async choose(board, me) -> { cell, drive? } }: the
// fly, you (a click on the board), a random player and a perfect (minimax) one.
// The fly's mushroom body runs in worker.js: it looks at the board and plays the
// cell whose compartment is driven least, and after every game it is shown its
// positions again with the result as dopamine (rules.mjs dopamineFor) - or,
// taught, it is corrected right after each wrong move (fly.mjs teach). What it
// has learned (the synapse gains) is kept in this browser's IndexedDB.
import { EMPTY, FLY, OPP, other, empties, winner, randomMove, perfectMove, dopamineFor, explorationAfter } from './rules.mjs?v=3';

const DELAY = { slow: 900, normal: 450, fast: 40 };

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------------ the brain
const worker = new Worker(new URL('./worker.js?v=3', import.meta.url), { type: 'module' });
const waiting = [];
worker.onmessage = ({ data: m }) => {
  if (m.type === 'progress') return progress(m);
  if (m.type === 'error') { fail(m.message); return; }
  const w = waiting.shift();
  if (w) w(m);
};
worker.onerror = (e) => fail(e.message || 'worker error');
const ask = (m) => new Promise((res) => { waiting.push(res); worker.postMessage(m); });

function setStatus(t, err = false) { const s = $('status'); s.textContent = t; s.className = err ? 'err' : ''; }
function fail(msg) { setAuto(false); setStatus('エラー: ' + msg, true); console.error(msg); }
function progress(p) {
  if (p.phase === 'download' && p.total) setStatus(`脳の配線をダウンロード中… ${(p.loaded / 1e6).toFixed(1)} / ${(p.total / 1e6).toFixed(1)} MB`);
  else setStatus(p.phase === 'decompress' ? '配線を展開中…' : 'キノコ体を組み立て中…');
}

// the learned gains, kept between visits (IndexedDB; a private window simply starts naive)
const DB = 'hae-sanmoku', KEY = 'gains-v1';
function idb(mode, fn) {
  return new Promise((res) => {
    try {
      const open = indexedDB.open(DB, 1);
      open.onupgradeneeded = () => open.result.createObjectStore('brain');
      open.onerror = () => res(null);
      open.onsuccess = () => {
        const tx = open.result.transaction('brain', mode), req = fn(tx.objectStore('brain'));
        tx.oncomplete = () => res(req.result ?? null);
        tx.onerror = tx.onabort = () => res(null);
      };
    } catch { res(null); }
  });
}
const loadGains = () => idb('readonly', (s) => s.get(KEY));
const saveGains = (v) => idb('readwrite', (s) => s.put(v, KEY));
const dropGains = () => idb('readwrite', (s) => s.delete(KEY));

const brainState = { learned: 0, gains: null };
const ready = (async () => {
  const saved = await loadGains();
  const m = await ask({ type: 'init', gains: saved?.gains });
  brainState.learned = m.loaded ? saved.learned || 0 : 0;
  brainState.gains = m.gains;
  showBrain();
  setStatus(`キノコ体（ケニヨン細胞 ${m.info.kc.toLocaleString()} 個）の準備ができました。` +
    (m.loaded ? `このブラウザで ${brainState.learned} 局学んだ脳を読み込みました。` : 'まだ何も学んでいません。'));
})();

function showBrain() {
  $('sLearned').textContent = brainState.learned;
  const g = brainState.gains;
  const eps = exploration();
  $('sLearnedSub').textContent = (eps ? `探索 ${(100 * eps).toFixed(1)}%  ` : '') +
    (g ? `区画の重み ${Math.min(...g).toFixed(3)}〜${Math.max(...g).toFixed(3)}` : '');
}

// the chance the fly plays a random move instead of its choice - only while it learns
function exploration() {
  return $('explore').checked && $('mode').value !== 'none' ? explorationAfter(brainState.learned) : 0;
}

// ------------------------------------------------------------------ players
const flyPlayer = {
  name: 'ハエ',
  async choose(board) {
    await ready;
    const m = await ask({ type: 'look', board, teach: $('mode').value === 'teach', eps: exploration() });
    if (m.gains) { brainState.gains = m.gains; showBrain(); }
    return { cell: m.cell, drive: m.drive, best: m.best, corrected: m.corrected, explored: m.explored };
  },
  async learn({ moves, result, mode }) {
    if (mode === 'none') return;
    if (mode === 'teach') return this.remember();
    // the positions the fly played in, and the cell it chose in each
    const board = Array(9).fill(EMPTY), mine = [];
    for (const [p, cell] of moves) {
      if (p === FLY) mine.push([board.slice(), cell]);
      board[cell] = p;
    }
    const m = await ask({ type: 'learn', moves: mine.map(([b, cell], k) => [b, cell, dopamineFor(result, k, mine.length)]) });
    brainState.gains = m.gains;
    await this.remember();
  },
  async remember() {
    brainState.learned++;
    showBrain();
    const g = await ask({ type: 'export' });
    await saveGains({ gains: g.gains, learned: brainState.learned });
  },
  async forget() {
    await ready;
    const m = await ask({ type: 'forget' });
    brainState.learned = 0;
    brainState.gains = m.gains;
    await dropGains();
    showBrain();
  },
};

const randomPlayer = { name: 'でたらめ', async choose(b) { return { cell: randomMove(b) }; } };
const perfectPlayer = { name: '負けない相手', async choose(b, me) { return { cell: perfectMove(b, me) }; } };

let clickWaiter = null;
const humanPlayer = {
  name: 'あなた',
  choose() { return new Promise((resolve) => { clickWaiter = (cell) => resolve({ cell }); }); },
};

const OPPONENTS = { human: humanPlayer, random: randomPlayer, perfect: perfectPlayer };

// ------------------------------------------------------------------ drawing
const SVG_O = '<svg viewBox="0 0 100 100" aria-hidden="true"><circle class="o" cx="50" cy="50" r="38"/></svg>';
const SVG_X = '<svg viewBox="0 0 100 100" aria-hidden="true"><path class="x" d="M15 15L85 85"/><path class="x" d="M85 15L15 85"/></svg>';

const boardEl = $('board');
const cells = [];
for (let i = 0; i < 9; i++) {
  const c = document.createElement('button');
  c.className = 'cell';
  c.setAttribute('role', 'gridcell');
  c.addEventListener('click', () => {
    if (clickWaiter && state.board[i] === EMPTY) { const w = clickWaiter; clickWaiter = null; w(i); }
  });
  boardEl.appendChild(c);
  cells.push(c);
}
const mindEl = $('mind');
const mindCells = Array.from({ length: 9 }, () => mindEl.appendChild(document.createElement('div')));

function cellLabel(i, v) {
  return `${Math.floor(i / 3) + 1}行${(i % 3) + 1}列 ${v === FLY ? '○（ハエ）' : v === OPP ? '×（' + state.opp.name + '）' : '空き'}`;
}

function drawBoard(last = -1, win = null) {
  const b = state.board;
  const humanTurn = !!clickWaiter;
  cells.forEach((c, i) => {
    const v = b[i];
    const want = v === FLY ? 'o' : v === OPP ? 'x' : '';
    if (c.dataset.v !== want) { c.innerHTML = v === FLY ? SVG_O : v === OPP ? SVG_X : ''; c.dataset.v = want; }
    c.classList.toggle('new', i === last);
    c.classList.toggle('last', i === last);
    c.classList.toggle('win', !!win?.line?.includes(i));
    c.disabled = !(humanTurn && v === EMPTY);
    c.setAttribute('aria-label', cellLabel(i, v));
  });
  boardEl.querySelector('.strike')?.remove();
  if (win?.line) {
    const at = (i) => [17 + (i % 3) * 33, 17 + Math.floor(i / 3) * 33];
    const [x1, y1] = at(win.line[0]), [x2, y2] = at(win.line[2]);
    boardEl.insertAdjacentHTML('beforeend',
      `<svg class="strike" viewBox="0 0 100 100" aria-hidden="true"><line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/></svg>`);
  }
}

function drawMind(drive, pick, best = null) {
  const b = state.board;
  const vals = drive ? empties(b).map((i) => drive[i]) : [];
  const lo = Math.min(...vals), hi = Math.max(...vals);
  mindCells.forEach((d, i) => {
    d.className = b[i] !== EMPTY && i !== pick ? 'taken' : i === pick ? 'pick' : '';
    if (best?.includes(i) && !best.includes(pick)) d.classList.add('best');
    if (drive && b[i] === EMPTY || drive && i === pick) {
      // the less driven, the brighter (that's the cell the fly wants)
      const t = hi > lo ? (hi - drive[i]) / (hi - lo) : 1;
      d.style.background = `rgba(255,180,84,${(0.08 + 0.55 * t).toFixed(3)})`;
      d.textContent = drive[i].toFixed(2);
    } else {
      d.style.background = '';
      d.textContent = b[i] === EMPTY || i === pick ? '—' : '';
    }
  });
}

function drawTurn(turn, thinking) {
  $('tFly').classList.toggle('on', turn === FLY);
  $('tYou').classList.toggle('on', turn === OPP);
  $('tYouName').textContent = state.opp.name;
  const msg = $('tMsg');
  msg.classList.toggle('dots', !!thinking);
  msg.textContent = turn === 0 ? '' : thinking ? (turn === FLY ? '考えています' : `${state.opp.name}の番`) :
    (turn === OPP && state.opp === humanPlayer ? 'マスを選んでください' : '');
  boardEl.classList.toggle('thinking', turn === FLY);
}

// ------------------------------------------------------------------ records
const STORE = 'hae-sanmoku-v1';
let history = [];
try { history = JSON.parse(localStorage.getItem(STORE) || '[]'); } catch { history = []; }
const save = () => { try { localStorage.setItem(STORE, JSON.stringify(history.slice(-500))); } catch {} };

function drawStats() {
  const n = history.length;
  const count = (gs, r) => gs.filter((g) => g.result === r).length;
  const pct = (k, of) => (of ? `${Math.round((100 * k) / of)}%` : '');
  const recent = history.slice(-20);
  $('sGames').textContent = n;
  $('sGamesSub').textContent = n ? `直近 ${recent.length} 戦: ハエ ${count(recent, 'fly')} 勝` : ' ';
  $('sFly').textContent = count(history, 'fly');
  $('sYou').textContent = count(history, 'opp');
  $('sDraw').textContent = count(history, 'draw');
  $('sFlySub').textContent = pct(count(history, 'fly'), n) || ' ';
  $('sYouSub').textContent = pct(count(history, 'opp'), n) || ' ';
  $('sDrawSub').textContent = pct(count(history, 'draw'), n) || ' ';

  const MODE = { self: '自分で', teach: '先生', none: 'なし' };
  const RES = { fly: ['ハエの勝ち', 'rf'], opp: ['相手の勝ち', 'ry'], draw: ['引き分け', 'rd'] };
  $('log').innerHTML = history.slice(-200).map((g, k, arr) => {
    const no = n - arr.length + k + 1;
    const kifu = g.moves.map(([p, i]) => (p === FLY ? '○' : '×') + (i + 1)).join(' ');
    return `<tr><td>${no}</td><td>${g.opp}</td><td>${MODE[g.mode] ?? ''}${g.mode === 'teach' ? ` (直し ${g.fixes})` : ''}${g.explored ? ` 探索 ${g.explored}` : ''}</td><td>${g.first === FLY ? 'ハエ' : '相手'}</td>` +
      `<td class="${RES[g.result][1]}">${RES[g.result][0]}</td><td>${g.moves.length}</td><td class="kifu">${kifu}</td></tr>`;
  }).reverse().join('');
}

// ------------------------------------------------------------------ the game
const state = { board: Array(9).fill(EMPTY), opp: humanPlayer, game: 0, firstFlip: false, auto: false };

async function playGame() {
  const id = ++state.game;
  const alive = () => id === state.game;
  clickWaiter = null;
  state.opp = OPPONENTS[$('opp').value];
  state.board = Array(9).fill(EMPTY);
  const f = $('first').value;
  const first = f === 'fly' ? FLY : f === 'you' ? OPP : (state.firstFlip = !state.firstFlip) ? OPP : FLY;
  const mode = $('mode').value;
  const moves = [];
  let fixes = 0, explored = 0;
  $('teachNote').textContent = '';
  $('result').textContent = ''; $('result').className = 'result';
  drawBoard(); drawMind(null, -1);

  let turn = first, w = null;
  while (!(w = winner(state.board))) {
    const player = turn === FLY ? flyPlayer : state.opp;
    const human = player === humanPlayer;
    drawTurn(turn, !human);
    const t0 = performance.now();
    const pending = player.choose(state.board.slice(), turn);
    if (human) drawBoard(moves.at(-1)?.[1] ?? -1);
    const move = await pending;
    if (!alive()) return;
    if (!human) { const wait = DELAY[$('speed').value] - (performance.now() - t0); if (wait > 0) await sleep(wait); }
    if (!alive()) return;
    if (state.board[move.cell] !== EMPTY) throw new Error(`${player.name} が埋まったマス ${move.cell + 1} に打とうとした`);
    if (turn === FLY) {
      drawMind(move.drive, move.cell, move.best);
      if (move.corrected) fixes++;
      if (move.explored) explored++;
      $('teachNote').innerHTML = (move.explored ? '探索: 脳の選んだマスではなく、でたらめに打ちました。' : '') + (!move.best ? '' : move.corrected ?
        `先生: <b>${move.cell + 1}</b> ではなく <b class="good">${move.best.map((c) => c + 1).join('・')}</b> がよい手（ドーパミンで直しました）` :
        `先生: <b>${move.cell + 1}</b> は<span class="good">よい手</span>`);
    }
    state.board[move.cell] = turn;
    moves.push([turn, move.cell]);
    drawBoard(move.cell);
    turn = other(turn);
  }

  const result = w.who === FLY ? 'fly' : w.who === OPP ? 'opp' : 'draw';
  drawTurn(0);
  drawBoard(moves.at(-1)[1], w);
  const r = $('result');
  r.className = 'result ' + (result === 'fly' ? 'fly' : result === 'opp' ? 'you' : 'draw');
  r.textContent = result === 'fly' ? 'ハエの勝ち！' : result === 'draw' ? '引き分け' :
    state.opp === humanPlayer ? 'あなたの勝ち！' : `${state.opp.name}の勝ち`;
  history.push({ opp: state.opp.name, first, result, moves, mode, fixes, explored, t: Date.now() });
  save(); drawStats();
  await flyPlayer.learn({ moves, result, mode });

  if (state.auto && alive()) {
    await sleep(DELAY[$('speed').value] * 2 + 200);
    if (state.auto && alive()) playGame();
  }
}

function start() {
  playGame().catch((e) => fail(e.message || e));
}

function setAuto(on) {
  state.auto = on;
  $('bAuto').textContent = on ? '■ 止める' : '▶ 続けて対戦';
}

$('bNew').addEventListener('click', start);
$('bAuto').addEventListener('click', () => { setAuto(!state.auto); if (state.auto) start(); });
$('opp').addEventListener('change', () => {
  const human = $('opp').value === 'human';
  $('bAuto').hidden = human;
  if (human) setAuto(false);
  start();
});
$('first').addEventListener('change', start);
$('explore').addEventListener('change', showBrain);
$('mode').addEventListener('change', showBrain);
$('bForget').addEventListener('click', async () => {
  if (!confirm('ハエが学んだことをすべて忘れさせますか？（対戦の記録は残ります）')) return;
  await flyPlayer.forget();
  setStatus('ハエは学んだことをすべて忘れました。');
  start();
});
$('bReset').addEventListener('click', () => {
  if (!history.length || !confirm('対戦の記録を消しますか？')) return;
  history = []; save(); drawStats();
});

drawStats();
start();
