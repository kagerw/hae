// The fly's mushroom body (fly.mjs) as a line-oriented process that drives a
// mountain car simulated elsewhere (mountaincar/train.py, gymnasium's MountainCar-v0).
//
//   stdin : {"cmd":"look","obs":[p,v],"eps":0.1}  -> {"action":a,"drive":[...],"greedy":a}
//           {"cmd":"learn","action":a,"reward":r} -> {"ok":true,"dopa":d}
//           {"cmd":"gains"}                        -> {"gain":[...]}
//           {"cmd":"save","path":"..."} / {"cmd":"load","path":"..."}
//   stdout: one JSON object per line; the first is {"ready":true,...}
//   args  : size= sigma= hz= ms= fb= eta= dscale= seed=
import { createInterface } from 'node:readline';
import { readFile, writeFile } from 'node:fs/promises';
import { gzipSync, gunzipSync } from 'node:zlib';
import { FlyBrain } from '../flybrain/flybrain.js';
import { makeCarBrain } from './fly.mjs';

const arg = Object.fromEntries(process.argv.slice(2).map((a) => a.split('=')));
const NAMES = { size: 'size', sigma: 'sigma', hz: 'hz', ms: 'ms', fb: 'feedbackMs', eta: 'eta', dscale: 'dopaScale', seed: 'seed' };
const config = {};
for (const [k, name] of Object.entries(NAMES)) if (arg[k] != null) config[name] = +arg[k];

const B = await makeCarBrain({ FlyBrain, base: new URL('../flybrain/', import.meta.url), config });
const say = (o) => process.stdout.write(JSON.stringify(o) + '\n');
say({ ready: true, ...B.info() });

for await (const line of createInterface({ input: process.stdin })) {
  if (!line.trim()) continue;
  const m = JSON.parse(line);
  if (m.cmd === 'look') {
    const { drive } = B.look(m.obs);
    say({ ...B.choose(drive, m.eps || 0), drive: drive.map((d) => +d.toFixed(5)) });
  } else if (m.cmd === 'learn') {
    say({ ok: true, dopa: B.learn(m.action, m.reward) });
  } else if (m.cmd === 'gains') {
    say({ gain: B.gains().map((g) => +g.toFixed(5)) });
  } else if (m.cmd === 'save') {
    await writeFile(m.path, gzipSync(Buffer.from(B.exportGains().buffer)));
    say({ ok: true });
  } else if (m.cmd === 'load') {
    B.importGains(new Float32Array(new Uint8Array(gunzipSync(await readFile(m.path))).buffer));
    say({ ok: true });
  } else if (m.cmd === 'quit') {
    break;
  } else say({ error: 'unknown cmd ' + m.cmd });
}
