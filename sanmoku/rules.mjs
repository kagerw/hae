// Tic-tac-toe: the board, the opponents and what a game's result is worth as
// dopamine - shared by the page (app.js) and the trainer (train.mjs).

export const EMPTY = 0, FLY = 1, OPP = 2;
export const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
export const other = (p) => (p === FLY ? OPP : FLY);
export const empties = (b) => b.flatMap((v, i) => (v === EMPTY ? [i] : []));

/** { who, line } once the game is over (who = 0 for a draw), else null. */
export function winner(b) {
  for (const line of LINES) {
    const [a, c, d] = line;
    if (b[a] && b[a] === b[c] && b[a] === b[d]) return { who: b[a], line };
  }
  return empties(b).length ? null : { who: 0, line: null };
}

export function randomMove(b, rnd = Math.random) {
  const e = empties(b);
  return e[Math.floor(rnd() * e.length)];
}

// minimax with depth, so it wins fast and loses late
function minimax(b, me, turn, depth) {
  const w = winner(b);
  if (w) return w.who === 0 ? 0 : (w.who === me ? 10 - depth : depth - 10);
  let best = turn === me ? -Infinity : Infinity;
  for (const i of empties(b)) {
    b[i] = turn;
    const s = minimax(b, me, other(turn), depth + 1);
    b[i] = EMPTY;
    best = turn === me ? Math.max(best, s) : Math.min(best, s);
  }
  return best;
}

/** Every best move for `me` (minimax: win soonest, lose latest). */
export function bestMoves(b, me) {
  b = b.slice();
  let best = -Infinity, cells = [];
  for (const i of empties(b)) {
    b[i] = me;
    const s = minimax(b, me, other(me), 1);
    b[i] = EMPTY;
    if (s > best) { best = s; cells = [i]; } else if (s === best) cells.push(i);
  }
  return cells;
}

/** A best move for `me`; ties broken at random, so it never loses but varies its games. */
export function perfectMove(b, me, rnd = Math.random) {
  const cells = bestMoves(b, me);
  return cells[Math.floor(rnd() * cells.length)];
}

// Exploration while learning: the chance of a random move, EPS0 at first and
// x EPS_DECAY for every game learned (as mountaincar: 10%, then less).
export const EPS0 = 0.1, EPS_DECAY = 0.995;
export const explorationAfter = (games) => EPS0 * EPS_DECAY ** games;

// The result as dopamine, into the compartment of the cell the fly played, for
// each of its moves: the last move gets it all, earlier ones less (decay per move).
export const REWARD = { win: 1, draw: 0.3, loss: -1, decay: 0.7 };

/** Dopamine for the fly's k-th move (0-based) of n in a game that ended in `result`. */
export function dopamineFor(result, k, n, R = REWARD) {
  const base = result === 'fly' ? R.win : result === 'opp' ? R.loss : R.draw;
  return base * R.decay ** (n - 1 - k);
}
