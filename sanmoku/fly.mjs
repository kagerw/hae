// The fly's mushroom body set up to play tic-tac-toe - shared by the Node
// trainer (train.mjs) and the page's worker (worker.js), so both run exactly
// the same brain.
//
// The same machinery as mountaincar/fly.mjs:
//   * the board is drawn as a 12 x 12 picture, a 4 x 4 block per cell: the
//     fly's ○ lights the block's edge midpoints (a ring), the opponent's × its
//     corners and centre (a cross) - two marks that share no pixel - and every
//     pixel drives its own share of the olfactory projection neurons, so the
//     Kenyon cells see a sparse code of the whole position;
//   * the MBONs are dealt into one compartment per cell, and the fly plays the
//     empty cell whose compartment the Kenyon cells drive least;
//   * learning on its own: after the game every position the fly played in is
//     shown again and the result comes back as dopamine into the compartment
//     of the cell it chose there - a win weakens the synapses from the active
//     Kenyon cells (so that move wins next time in that position), a loss
//     strengthens them;
//   * learning from a teacher: right after each move, while the position is
//     still shown, a wrong move (not one of minimax's best) brings dopamine -
//     -1 into the chosen cell's compartment, +1 into the best cells'.

export const DEFAULTS = {
  hz: 220,           // Poisson rate for a lit pixel
  ms: 80,            // how long a position is shown before choosing
  feedbackMs: 60,    // dopamine while the position is shown again
  eta: 1e-3,         // a game brings only a few dopamine pulses (train.mjs: 3e-4..3e-3 all learn alike)
  gainMin: 0.05, gainMax: 2,
  seed: 1,
};
export const SIZE = 12, CELLS = 9, EMPTY = 0, FLY = 1, OPP = 2;

// a cell's 4 x 4 block: ○ = the edge midpoints, × = the corners and the centre
const RING = [[0, 1], [0, 2], [1, 0], [2, 0], [1, 3], [2, 3], [3, 1], [3, 2]];
const CROSS = [[0, 0], [0, 3], [3, 0], [3, 3], [1, 1], [1, 2], [2, 1], [2, 2]];

/** The board (9 cells of EMPTY / FLY / OPP) as the picture the fly sees. */
export function picture(board) {
  const img = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < CELLS; i++) {
    if (!board[i]) continue;
    const r0 = Math.floor(i / 3) * 4, c0 = (i % 3) * 4;
    for (const [r, c] of board[i] === FLY ? RING : CROSS) img[(r0 + r) * SIZE + c0 + c] = 1;
  }
  return img;
}

async function readBytes(url) {
  if (url.protocol === 'file:') {
    const { readFile } = await import('node:fs/promises');
    return new Uint8Array(await readFile(url));
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

/**
 * @param {object} o
 * @param {typeof import('../flybrain/flybrain.js').FlyBrain} o.FlyBrain
 * @param {URL} o.base        the flybrain/ directory
 * @param {object} [o.config] overrides of DEFAULTS
 * @param {string} [o.v]      cache-busting query for the library's data
 */
export async function makeBoardBrain({ FlyBrain, base, config = {}, v = '', onProgress }) {
  const C = { ...DEFAULTS, ...config };
  const mb = JSON.parse(new TextDecoder().decode(await readBytes(new URL('data/mb783.json' + v, base))));
  const brain = await FlyBrain.load({
    graph: new URL('data/flywire783.fbg.gz' + v, base),
    wasm: new URL('flybrain.wasm' + v, base),
    onProgress,
  });
  const G = mb.groups;
  const pick = (p) => Object.keys(G).filter((k) => k.startsWith(p)).sort();
  const all = (keys) => keys.flatMap((k) => G[k].idx);
  const KC = all(pick('kc:')), MBON = all(pick('mbon:')), DAN = all(pick('dan:')),
        MBIN = all(pick('mbin:')), ALPN = all(pick('alpn:'));

  // the mushroom body alone: the whole-brain model runs away on any olfactory input
  const inside = new Set([...KC, ...MBON, ...DAN, ...MBIN, ...ALPN]);
  const outside = [];
  for (let i = 0; i < brain.n; i++) if (!inside.has(i)) outside.push(i);
  brain.silence(outside, true, { byIndex: true });

  // MBONs dealt to the cells, best-connected first, to whichever has the fewest KC inputs so far
  const kcIn = new Map(), mbonSet = new Set(MBON);
  for (const k of KC) for (const i of brain.outgoing(k, { byIndex: true }).post)
    if (mbonSet.has(i)) kcIn.set(i, (kcIn.get(i) || 0) + 1);
  const typeOf = new Map();
  for (const t of pick('mbon:')) for (const i of G[t].idx) typeOf.set(i, t.slice(5));
  const groups = Array.from({ length: CELLS }, (_, cell) => ({ cell, cells: [], types: [], inputs: 0 }));
  for (const i of [...kcIn.keys()].sort((a, b) => kcIn.get(b) - kcIn.get(a) || a - b)) {
    let g = groups[0];
    for (const h of groups) if (h.inputs < g.inputs) g = h;
    g.cells.push(i); g.types.push(typeOf.get(i)); g.inputs += kcIn.get(i);
  }
  const plastic = (eta) => ({ eta, tauTrace: 40, tauDopa: 1e7, gainMin: C.gainMin, gainMax: C.gainMax });
  const synapses = brain.setPlasticity({ pre: KC, groups: groups.map((g) => ({ post: g.cells, modulators: [] })), ...plastic(0) });

  // pixels -> projection neurons, dealt round-robin after a fixed shuffle
  const npix = SIZE * SIZE;
  let rs = 12345;
  const lcg = () => ((rs = (rs * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const deck = [...ALPN];
  for (let i = deck.length - 1; i > 0; i--) { const j = Math.floor(lcg() * (i + 1)); [deck[i], deck[j]] = [deck[j], deck[i]]; }
  const chan = Array.from({ length: npix }, () => []);
  deck.forEach((p, i) => chan[i % npix].push(p));

  // the choice's own randomness (ties, exploration), separate from the brain's Poisson input
  let ss = C.seed * 2654435761 >>> 0;
  const rnd = () => { ss = (ss + 0x6d2b79f5) >>> 0; let t = ss; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };

  let brainSeed = C.seed * 100000;
  /** Show a position and read the compartments; also how many Kenyon cells fired. */
  function look(board) {
    const img = picture(board);
    brain.clearStimuli();
    for (let i = 0; i < npix; i++) if (img[i]) brain.stimulate(chan[i], C.hz * img[i], { byIndex: true });
    brain.setPlasticityParams(plastic(0));
    brain.reset(++brainSeed);
    brain.run(C.ms, { events: false });
    const c = brain.counts(), spikes = new Uint16Array(KC.length);
    let active = 0;
    for (let k = 0; k < KC.length; k++) { spikes[k] = Math.min(65535, c[KC[k]]); if (spikes[k]) active++; }
    return { drive: [...brain.driveByGroup(KC, spikes)], active };
  }

  /** The empty cell whose compartment is driven least; ties (a naive brain) broken at random. */
  function choose(board, drive, eps = 0) {
    const free = [];
    for (let i = 0; i < CELLS; i++) if (!board[i]) free.push(i);
    let best = [free[0]];
    for (const i of free.slice(1)) {
      if (drive[i] < drive[best[0]] - 1e-9) best = [i];
      else if (Math.abs(drive[i] - drive[best[0]]) <= 1e-9) best.push(i);
    }
    const greedy = best[Math.floor(rnd() * best.length)];
    const cell = rnd() < eps ? free[Math.floor(rnd() * free.length)] : greedy;
    return { cell, greedy };
  }

  /** Dopamine into cells' compartments, [[cell, level], ...], while the position is shown (call right after look). */
  function reinforce(pairs) {
    pairs = pairs.map(([cell, d]) => [cell, Math.max(-1, Math.min(1, d))]).filter(([, d]) => d !== 0);
    if (!pairs.length || C.eta === 0) return pairs;
    brain.setPlasticityParams(plastic(C.eta));
    for (const [cell, d] of pairs) brain.dopamine(cell, d);
    brain.run(C.feedbackMs, { events: false });
    for (const [cell] of pairs) brain.dopamine(cell, 0);
    brain.setPlasticityParams(plastic(0));
    return pairs;
  }
  /** Dopamine into one cell's compartment (call right after look); returns the level used. */
  const learn = (cell, dopa) => reinforce([[cell, dopa]])[0]?.[1] ?? 0;

  /**
   * The teacher (right after look, the position still shown): if `cell` is not
   * one of the `best` moves, the chosen cell's compartment gets -1 (its synapses
   * strengthen, so it is chosen less here) and every best cell's +1 (weaken,
   * chosen more; punish / praise scale the two). A right move gets nothing - as in the reading school, only a
   * wrong answer brings dopamine.
   */
  function teach(cell, best, { punish = 1, praise = 1 } = {}) {
    if (best.includes(cell)) return [];
    return reinforce([[cell, -punish], ...best.map((c) => [c, praise])]);
  }

  return {
    config: C, brain, KC, ALPN, groups, synapses, look, choose, learn, reinforce, teach,
    gains: () => groups.map((_, c) => brain.gain(c)),
    setEta: (eta) => { C.eta = eta; },
    exportGains: () => brain.exportGains(),
    importGains: (g) => brain.importGains(g),
    forget: () => brain.forget(),
    info: () => ({ neurons: brain.n, kc: KC.length, pn: ALPN.length, config: C,
      compartments: groups.map((g, c) => ({ cell: g.cell, mbons: g.cells.length,
        types: [...new Set(g.types)].join('+'), kcInputs: g.inputs, synapses: synapses[c] })) }),
  };
}
