// Train the fly (fly.mjs) at tic-tac-toe in Node and measure it - the same
// brain, rules and dopamine as the page.
//
//   node sanmoku/train.mjs games=2000 opp=random every=200 test=100 out=sanmoku/brain.bin.gz
//   args: games= opp=random|perfect|mix mode=self|teach explore=1|0 punish= praise= every= test= eta= win= draw= loss= decay= seed= out=
import { writeFile } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';
import { FlyBrain } from '../flybrain/flybrain.js';
import { makeBoardBrain } from './fly.mjs';
import { EMPTY, FLY, OPP, winner, randomMove, perfectMove, bestMoves, REWARD, dopamineFor, explorationAfter } from './rules.mjs';

const arg = Object.fromEntries(process.argv.slice(2).map((a) => a.split('=')));
const GAMES = +(arg.games ?? 2000), EVERY = +(arg.every ?? 200), TEST = +(arg.test ?? 100);
const OPPONENT = arg.opp ?? 'random', MODE = arg.mode ?? 'self', EXPLORE = arg.explore !== '0';
const reward = { ...REWARD };
for (const k of ['win', 'draw', 'loss', 'decay']) if (arg[k] != null) reward[k] = +arg[k];
const TEACH = { punish: +(arg.punish ?? 1), praise: +(arg.praise ?? 1) };
const config = { seed: +(arg.seed ?? 1) };
if (arg.eta != null) config.eta = +arg.eta;

const B = await makeBoardBrain({ FlyBrain, base: new URL('../flybrain/', import.meta.url), config });
console.log(JSON.stringify({ ...B.info(), compartments: undefined, reward, games: GAMES, opp: OPPONENT, mode: MODE, explore: EXPLORE, teach: TEACH }));

let r = (config.seed * 7919) >>> 0;
const rnd = () => { r = (r + 0x6d2b79f5) >>> 0; let t = r; t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };

let corrections = 0;   // teacher corrections so far (training games only)

/** One game; the fly learns from it unless `learn` is false. Returns 'fly' | 'opp' | 'draw'. */
function play(opp, flyFirst, learn, eps = 0) {
  const board = Array(9).fill(EMPTY), mine = [];
  let fixes = 0;
  let turn = flyFirst ? FLY : OPP, w;
  while (!(w = winner(board))) {
    let cell;
    if (turn === FLY) {
      cell = B.choose(board, B.look(board).drive, eps).cell;
      if (learn && MODE === 'teach' && B.teach(cell, bestMoves(board, FLY), TEACH).length) fixes++;
      mine.push([board.slice(), cell]);
    } else cell = opp === 'perfect' ? perfectMove(board, OPP, rnd) : randomMove(board, rnd);
    board[cell] = turn;
    turn = turn === FLY ? OPP : FLY;
  }
  const result = w.who === FLY ? 'fly' : w.who === OPP ? 'opp' : 'draw';
  if (learn) corrections += fixes;
  if (learn && MODE === 'self') mine.forEach(([b, cell], k) => { B.look(b); B.learn(cell, dopamineFor(result, k, mine.length, reward)); });
  return result;
}

function test(opp) {
  const n = { fly: 0, draw: 0, opp: 0 };
  for (let g = 0; g < TEST; g++) n[play(opp, g % 2 === 0, false)]++;
  return `${opp} ${n.fly}/${n.draw}/${n.opp}`;
}

const t0 = Date.now();
console.log(`games 0  (${test('random')}  ${test('perfect')})  win/draw/loss of ${TEST}`);
for (let g = 1; g <= GAMES; g++) {
  const opp = OPPONENT === 'mix' ? (rnd() < 0.5 ? 'random' : 'perfect') : OPPONENT;
  play(opp, g % 2 === 1, true, EXPLORE ? explorationAfter(g - 1) : 0);
  if (g % EVERY === 0)
    console.log(`games ${g}  (${test('random')}  ${test('perfect')})  fixes ${corrections}  gains ${B.gains().map((x) => x.toFixed(2)).join(' ')}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
if (arg.out) {
  await writeFile(arg.out, gzipSync(Buffer.from(B.exportGains().buffer)));
  console.log('wrote', arg.out);
}
