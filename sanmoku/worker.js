// The page's fly: the same mushroom body as train.mjs (fly.mjs), in a worker.
//
//   in : {type:'init', gains?} | {type:'look', board, teach?, eps?} | {type:'learn', moves:[[board, cell, dopa]]}
//        {type:'forget'} | {type:'export'}
//   out: {type:'progress'|'ready'|'look'|'learn'|'forgot'|'export'|'error', ...}
import { FlyBrain } from '../flybrain/flybrain.js?v=5';
import { makeBoardBrain, FLY } from './fly.mjs?v=2';
import { bestMoves } from './rules.mjs?v=3';

const V = '?v=5';
const post = (m) => self.postMessage(m);
let B = null;

self.onmessage = async ({ data: m }) => {
  try {
    if (m.type === 'init') {
      B = await makeBoardBrain({ FlyBrain, base: new URL('../flybrain/', import.meta.url), v: V, config: m.config,
        onProgress: (p) => post({ type: 'progress', phase: p.phase, loaded: p.loaded, total: p.total }) });
      let loaded = false;
      if (m.gains) try { B.importGains(new Float32Array(m.gains)); loaded = true; } catch { /* another wiring: start naive */ }
      post({ type: 'ready', info: B.info(), gains: B.gains(), loaded });
    } else if (m.type === 'look') {
      const { drive, active } = B.look(m.board);
      const { cell, greedy } = B.choose(m.board, drive, m.eps || 0);
      // the teacher looks at the same position; a wrong move is corrected while it is still shown
      const best = m.teach ? bestMoves(m.board, FLY) : null;
      const dopa = best ? B.teach(cell, best) : [];
      post({ type: 'look', cell, explored: cell !== greedy, drive, active, best, corrected: dopa.length > 0, gains: dopa.length ? B.gains() : null });
    } else if (m.type === 'learn') {
      // show every position the fly played in again, with the result as dopamine into the cell it chose
      const dopa = m.moves.map(([board, cell, d]) => { B.look(board); return B.learn(cell, d); });
      post({ type: 'learn', dopa, gains: B.gains() });
    } else if (m.type === 'forget') {
      B.forget();
      post({ type: 'forgot', gains: B.gains() });
    } else if (m.type === 'export') {
      const g = B.exportGains();
      self.postMessage({ type: 'export', gains: g.buffer }, [g.buffer]);
    }
  } catch (e) {
    post({ type: 'error', message: String(e && e.stack || e) });
  }
};
