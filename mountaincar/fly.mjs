// The fly's mushroom body set up to drive a mountain car - shared by the Node
// process (brain.mjs, driven by train.py) and the page's worker (worker.js), so
// both run exactly the same brain.
//
// The same machinery as juku/reader.mjs, rebuilt here so the reading flies are
// not touched:
//   * the car's state (position, velocity) is drawn as a SIZE x SIZE picture -
//     a soft blob at (position, velocity) - and every pixel drives its own share
//     of the olfactory projection neurons, so the Kenyon cells see a sparse code
//     of where in the state space the car is;
//   * the MBONs are dealt into one compartment per action (push left, don't
//     push, push right), and the action is the compartment the Kenyon cells
//     drive least (spikes x synapses x learned weight, relative to naive);
//   * the reward from the environment comes back as dopamine into the chosen
//     action's compartment while the state is still being shown - positive
//     reward weakens its synapses from the active Kenyon cells (so this action
//     wins next time in this state), negative reward strengthens them (so it
//     loses). The learning happens in the WASM core's plasticity rule.

export const DEFAULTS = {
  size: 12,          // the state picture is size x size
  sigma: 0.8,        // blob width, pixels
  hz: 220,           // Poisson rate for a full pixel
  ink: 0.05,
  ms: 80,            // how long each state is shown before choosing
  feedbackMs: 60,    // dopamine while the state is still shown
  eta: 6e-5,
  dopaScale: 1.0,    // reward -> dopamine level (clipped to +-1)
  gainMin: 0.05, gainMax: 2,
  seed: 1,
};
export const LO = [-1.2, -0.07], HI = [0.6, 0.07];
export const ACTIONS = ['左に押す', '押さない', '右に押す'];

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
export async function makeCarBrain({ FlyBrain, base, config = {}, v = '', onProgress }) {
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

  // MBONs dealt to the actions, best-connected first, to whichever has the fewest KC inputs so far
  const kcIn = new Map(), mbonSet = new Set(MBON);
  for (const k of KC) for (const i of brain.outgoing(k, { byIndex: true }).post)
    if (mbonSet.has(i)) kcIn.set(i, (kcIn.get(i) || 0) + 1);
  const typeOf = new Map();
  for (const t of pick('mbon:')) for (const i of G[t].idx) typeOf.set(i, t.slice(5));
  const groups = ACTIONS.map((name) => ({ action: name, cells: [], types: [], inputs: 0 }));
  for (const i of [...kcIn.keys()].sort((a, b) => kcIn.get(b) - kcIn.get(a) || a - b)) {
    let g = groups[0];
    for (const h of groups) if (h.inputs < g.inputs) g = h;
    g.cells.push(i); g.types.push(typeOf.get(i)); g.inputs += kcIn.get(i);
  }
  const plastic = (eta) => ({ eta, tauTrace: 40, tauDopa: 1e7, gainMin: C.gainMin, gainMax: C.gainMax });
  const synapses = brain.setPlasticity({ pre: KC, groups: groups.map((g) => ({ post: g.cells, modulators: [] })), ...plastic(0) });

  // pixels -> projection neurons, dealt round-robin after a fixed shuffle
  const npix = C.size * C.size;
  let rs = 12345;
  const lcg = () => ((rs = (rs * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const deck = [...ALPN];
  for (let i = deck.length - 1; i > 0; i--) { const j = Math.floor(lcg() * (i + 1)); [deck[i], deck[j]] = [deck[j], deck[i]]; }
  const chan = Array.from({ length: npix }, () => []);
  deck.forEach((p, i) => chan[i % npix].push(p));

  // the choice's own randomness (exploration), separate from the brain's Poisson input
  let ss = C.seed * 2654435761 >>> 0;
  const rnd = () => { ss = (ss + 0x6d2b79f5) >>> 0; let t = ss; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };

  /** (position, velocity) -> a size x size picture: a gaussian blob, x = position, y = velocity. */
  function picture([p, v]) {
    const img = new Float32Array(npix), s = C.size - 1;
    const x = Math.max(0, Math.min(1, (p - LO[0]) / (HI[0] - LO[0]))) * s;
    const y = Math.max(0, Math.min(1, (v - LO[1]) / (HI[1] - LO[1]))) * s;
    for (let r = 0; r < C.size; r++) for (let c = 0; c < C.size; c++)
      img[r * C.size + c] = Math.exp(-((c - x) ** 2 + (r - y) ** 2) / (2 * C.sigma ** 2));
    return img;
  }

  let brainSeed = C.seed * 100000;
  /** Show a state and read the compartments; also how many Kenyon cells fired. */
  function look(obs) {
    const img = picture(obs);
    brain.clearStimuli();
    for (let i = 0; i < npix; i++) if (img[i] > C.ink) brain.stimulate(chan[i], C.hz * img[i], { byIndex: true });
    brain.setPlasticityParams(plastic(0));
    brain.reset(++brainSeed);
    brain.run(C.ms, { events: false });
    const c = brain.counts(), spikes = new Uint16Array(KC.length);
    let active = 0;
    for (let k = 0; k < KC.length; k++) { spikes[k] = Math.min(65535, c[KC[k]]); if (spikes[k]) active++; }
    return { drive: [...brain.driveByGroup(KC, spikes)], active, img };
  }

  /** The compartment driven least; ties (a naive brain) broken at random; eps = chance of a random action. */
  function choose(drive, eps = 0) {
    let best = [0];
    for (let a = 1; a < drive.length; a++) {
      if (drive[a] < drive[best[0]] - 1e-9) best = [a];
      else if (Math.abs(drive[a] - drive[best[0]]) <= 1e-9) best.push(a);
    }
    const greedy = best[Math.floor(rnd() * best.length)];
    const action = rnd() < eps ? Math.floor(rnd() * drive.length) : greedy;
    return { action, greedy };
  }

  /** Dopamine into the chosen action's compartment while the state is still shown. */
  function learn(action, reward) {
    const d = Math.max(-1, Math.min(1, reward * C.dopaScale));
    if (d === 0 || C.eta === 0) return 0;
    brain.setPlasticityParams(plastic(C.eta));
    brain.dopamine(action, d);
    brain.run(C.feedbackMs, { events: false });
    brain.dopamine(action, 0);
    brain.setPlasticityParams(plastic(0));
    return d;
  }

  return {
    config: C, brain, KC, ALPN, groups, synapses, look, choose, learn, picture,
    gains: () => groups.map((_, c) => brain.gain(c)),
    setEta: (eta) => { C.eta = eta; },
    exportGains: () => brain.exportGains(),
    importGains: (g) => brain.importGains(g),
    forget: () => brain.forget(),
    info: () => ({ neurons: brain.n, kc: KC.length, pn: ALPN.length, config: C,
      compartments: groups.map((g, c) => ({ action: g.action, mbons: g.cells.length,
        types: [...new Set(g.types)].join('+'), kcInputs: g.inputs, synapses: synapses[c] })) }),
  };
}
