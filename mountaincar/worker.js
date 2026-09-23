// The page's fly: the same mushroom body as brain.mjs (fly.mjs), in a worker.
//
//   in : {type:'init'} | {type:'look', obs, eps} | {type:'learn', action, reward}
//        {type:'policy', nx, ny} | {type:'forget'} | {type:'eta', eta}
//   out: {type:'progress'|'ready'|'look'|'learn'|'policy'|'error', ...}
import { FlyBrain } from '../flybrain/flybrain.js?v=5';
import { makeCarBrain, LO, HI } from './fly.mjs?v=1';

const V = '?v=5';
const post = (m) => self.postMessage(m);
let B = null;

self.onmessage = async ({ data: m }) => {
  try {
    if (m.type === 'init') {
      B = await makeCarBrain({ FlyBrain, base: new URL('../flybrain/', import.meta.url), v: V,
        onProgress: (p) => post({ type: 'progress', phase: p.phase, loaded: p.loaded, total: p.total }) });
      post({ type: 'ready', info: B.info() });
    } else if (m.type === 'look') {
      const { drive, active } = B.look(m.obs);
      post({ type: 'look', ...B.choose(drive, m.eps || 0), drive, active });
    } else if (m.type === 'learn') {
      post({ type: 'learn', dopa: B.learn(m.action, m.reward), gains: B.gains() });
    } else if (m.type === 'policy') {
      // what the fly would do everywhere in the state space right now (looking only, no learning)
      const cells = [];
      for (let j = 0; j < m.ny; j++) for (let i = 0; i < m.nx; i++) {
        const p = LO[0] + (i + 0.5) / m.nx * (HI[0] - LO[0]);
        const v = LO[1] + (j + 0.5) / m.ny * (HI[1] - LO[1]);
        const { drive } = B.look([p, v]);
        let best = 0;
        for (let a = 1; a < 3; a++) if (drive[a] < drive[best]) best = a;
        const sorted = [...drive].sort((a, b) => a - b);
        cells.push({ a: Math.abs(sorted[1] - sorted[0]) < 1e-9 ? -1 : best, margin: sorted[1] - sorted[0] });
      }
      post({ type: 'policy', nx: m.nx, ny: m.ny, cells });
    } else if (m.type === 'forget') {
      B.forget();
      post({ type: 'forgot', gains: B.gains() });
    } else if (m.type === 'eta') {
      B.setEta(m.eta);
    }
  } catch (e) {
    post({ type: 'error', message: String(e && e.stack || e) });
  }
};
