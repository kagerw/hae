"""ハエのコネクトーム（FlyWire v783 のキノコ体）に Mountain Car を学習させる。

    py -3.12 mountaincar/train.py                 # 学習（既定 300 エピソードまで）
    py -3.12 mountaincar/train.py --episodes 500 --seed 2

環境は gymnasium の MountainCar-v0（1 エピソード最大 200 ステップ）。
報酬は MountainCarEnergyReward（力学的エネルギーの増分 + ゴールボーナス）だけを使い、
元の -1/step の報酬は使わない。

脳は mountaincar/brain.mjs（Node 子プロセス）。1 ステップごとに
  1. 状態 (位置, 速度) を絵にして投射ニューロンに入れ、ケニヨン細胞 → MBON を走らせ、
  2. 3 つの行動区画のうち MBON が最も駆動されない区画を行動として選び、
  3. 環境を 1 ステップ進めて得た報酬を、その行動区画へのドーパミンとして返す
     （状態を見せたまま。正の報酬 → その行動が選ばれやすく、負 → 選ばれにくく）。

「何回の試行錯誤で成功したか」は、
  - 初めてゴールした エピソード番号 と それまでの累計ステップ（= 行動の試行回数）
  - 連続 --streak 回ゴールできるようになった（=習得した）エピソード番号
として表示し、mountaincar/runs/<日時>-seed<n>-eta<η>/ に episodes.csv・summary.json・学習した
シナプス重み（gains.bin.gz）を残す（matplotlib があれば learning_curve.png も）。
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import gymnasium as gym
import numpy as np


class MountainCarEnergyReward(gym.Wrapper):
    """位置・運動エネルギーの増分を報酬にするラッパー。"""

    def __init__(self, env, ke_scale=1.0, goal_bonus=50.0):
        super().__init__(env)
        self.prev_energy = None
        self.g = 9.8
        self.ke_scale = ke_scale
        self.goal_bonus = goal_bonus

    # --- API 差異の吸収（元コードの方針を踏襲） ---------------
    @staticmethod
    def _unpack_reset(result):
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    @staticmethod
    def _unpack_step(result):
        if len(result) == 4:
            obs, rew, done, info = result
            return obs, rew, done, False, info
        return result

    def _ensure_obs_shape(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        if obs.ndim > 1:
            obs = obs[0]
        if obs.shape[0] != 2:
            raise ValueError(f"Expected observation shape (2,), got {obs.shape}")
        return obs

    def reset(self, **kwargs):
        obs, info = self._unpack_reset(self.env.reset(**kwargs))
        obs = self._ensure_obs_shape(obs)
        self.prev_energy = self.compute_energy(obs)
        return obs, info

    def step(self, action):
        obs, _orig, terminated, truncated, info = self._unpack_step(
            self.env.step(action)
        )
        obs = self._ensure_obs_shape(obs)

        current_energy = self.compute_energy(obs)
        shaped = 10000.0  * (current_energy - self.prev_energy)
        self.prev_energy = current_energy

        # ---- 元コードのバグ修正 --------------------------------
        # 旧: step_per_episode - info.get("time_step", 0)
        #     → step_per_episode が未定義で NameError。しかも結果を捨てていた。
        if terminated and obs[0] >= self.env.unwrapped.goal_position:
            shaped += self.goal_bonus
            info["reached_goal"] = True

        return obs, shaped, terminated, truncated, info

    def compute_energy(self, obs):
        position, velocity = obs
        # gym の MountainCar の力学から導かれる保存量
        #   dv/dt = 0.001*(a-1) - 0.0025*cos(3p)
        #   => 0.5*v^2 + (0.0025/3)*sin(3p) = const（押さなければ保存）
        kinetic   = 0.5 * velocity ** 2
        potential = (0.0025 / 3.0) * np.sin(3.0 * position)
        return kinetic + potential


HERE = os.path.dirname(os.path.abspath(__file__))


class FlyBrain:
    """mountaincar/brain.mjs との JSON 行プロトコル。"""

    def __init__(self, **opts):
        args = ["node", "--experimental-detect-module", os.path.join(HERE, "brain.mjs")]
        args += [f"{k}={v}" for k, v in opts.items()]
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, encoding="utf-8", bufsize=1)
        self.info = self._read()
        if not self.info.get("ready"):
            raise RuntimeError(f"brain did not start: {self.info}")

    def _read(self):
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("brain process exited")
        return json.loads(line)

    def ask(self, **msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        return self._read()

    def close(self):
        try:
            self.p.stdin.write('{"cmd":"quit"}\n')
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


DOPA_MODES = {
    "reward": "報酬をそのままドーパミンに（本来の条件）",
    "none": "ドーパミンなし（可塑性はオンのまま、ドーパミンを一切流さない）",
    "shuffle": "ドーパミンはあるが報酬と無関係（毎ステップ報酬の符号をランダムに反転）",
    "goal": "ゴールしたときだけドーパミン（エネルギーの増分は使わない）",
}


def dopamine_signal(mode, reward, info, rng):
    """その 1 ステップで脳に返すドーパミン源（brain.mjs が ±1 で頭打ちにする）。None = 流さない。"""
    if mode == "reward":
        return reward
    if mode == "none":
        return None
    if mode == "shuffle":
        return reward * (1.0 if rng.random() < 0.5 else -1.0)
    if mode == "goal":
        return 1.0 if info.get("reached_goal") else None
    raise ValueError(mode)


def run_episode(env, brain, seed, eps, learn=True, dopa="reward", rng=None):
    obs, _ = env.reset(seed=seed)
    total, steps, reached, max_pos = 0.0, 0, False, float(obs[0])
    acts, along = [0, 0, 0], 0
    while True:
        r = brain.ask(cmd="look", obs=[float(obs[0]), float(obs[1])], eps=eps)
        a = r["action"]
        acts[a] += 1
        along += a == (2 if obs[1] >= 0 else 0)    # 今の速度の向きに押したか（エネルギーが増える押し方）
        obs, reward, terminated, truncated, info = env.step(a)
        if learn:
            d = dopamine_signal(dopa, float(reward), info, rng)
            if d is not None:
                brain.ask(cmd="learn", action=a, reward=float(d))
        total += reward
        steps += 1
        max_pos = max(max_pos, float(obs[0]))
        if info.get("reached_goal"):
            reached = True
        if terminated or truncated:
            break
    return dict(steps=steps, reached_goal=reached, shaped_return=total,
                max_position=max_pos, actions=acts, along=along / steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300, help="学習エピソードの上限")
    ap.add_argument("--streak", type=int, default=10, help="この回数連続でゴールしたら習得とみなす")
    ap.add_argument("--eval", type=int, default=20, help="習得後、学習を止めて試すエピソード数")
    ap.add_argument("--eps", type=float, default=0.1, help="探索率の初期値（エピソードごとに減衰）")
    ap.add_argument("--eps-decay", type=float, default=0.97)
    ap.add_argument("--eps-min", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dopa", choices=list(DOPA_MODES), default="reward",
                    help="ドーパミンの与え方: " + " / ".join(f"{k}={v}" for k, v in DOPA_MODES.items()))
    ap.add_argument("--out", default=None, help="記録を置くディレクトリ（既定 mountaincar/runs）")
    ap.add_argument("--eta", type=float, default=6e-5, help="KC→MBON 可塑性の学習率")
    ap.add_argument("--ms", type=int, default=80, help="1 状態を見せる時間 (ms, 脳内時間)")
    ap.add_argument("--fb", type=int, default=60, help="ドーパミンを流す時間 (ms, 脳内時間)")
    ap.add_argument("--dscale", type=float, default=1.0, help="報酬 → ドーパミン量の倍率（±1 で頭打ち）")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-seed{args.seed}-eta{args.eta:g}"
    if args.dopa != "reward":
        stamp += f"-dopa{args.dopa}"
    out = args.out or os.path.join(HERE, "runs")
    run_dir = os.path.join(out, stamp)
    n = 1
    while os.path.exists(run_dir):
        n += 1
        run_dir = os.path.join(out, f"{stamp}-{n}")
    os.makedirs(run_dir)

    env = MountainCarEnergyReward(gym.make("MountainCar-v0"))
    brain = FlyBrain(seed=args.seed, eta=args.eta, ms=args.ms, fb=args.fb, dscale=args.dscale)
    print(f"ドーパミン: {args.dopa} — {DOPA_MODES[args.dopa]}")
    print(f"脳: FlyWire v783 {brain.info['neurons']:,} ニューロン中、キノコ体のみ "
          f"(KC {brain.info['kc']:,} / 投射ニューロン {brain.info['pn']})")
    for c in brain.info["compartments"]:
        print(f"  区画「{c['action']}」: MBON {c['mbons']} 個, 可塑シナプス {c['synapses']:,}")
    print()

    rows, first_success, mastered = [], None, None
    total_steps, streak = 0, 0
    eps = args.eps
    dopa_rng = np.random.default_rng(args.seed + 7919)
    t0 = time.time()
    with open(os.path.join(run_dir, "episodes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "steps", "reached_goal", "shaped_return", "max_position",
                    "eps", "cum_steps", "along_velocity", "left", "none", "right", "gain_left", "gain_none", "gain_right"])
        for ep in range(1, args.episodes + 1):
            r = run_episode(env, brain, seed=args.seed * 100000 + ep, eps=eps, dopa=args.dopa, rng=dopa_rng)
            total_steps += r["steps"]
            gains = brain.ask(cmd="gains")["gain"]
            streak = streak + 1 if r["reached_goal"] else 0
            if r["reached_goal"] and first_success is None:
                first_success = dict(episode=ep, cum_steps=total_steps, steps=r["steps"])
            if streak >= args.streak and mastered is None:
                mastered = dict(episode=ep, cum_steps=total_steps)
            row = dict(episode=ep, eps=round(eps, 4), cum_steps=total_steps, gains=gains, **r)
            rows.append(row)
            w.writerow([ep, r["steps"], int(r["reached_goal"]), round(r["shaped_return"], 3),
                        round(r["max_position"], 4), round(eps, 4), total_steps, round(r["along"], 4), *r["actions"], *gains])
            f.flush()
            mark = "★ゴール" if r["reached_goal"] else "      "
            print(f"ep {ep:4d} {mark} steps={r['steps']:3d} 最高位置={r['max_position']:+.3f} "
                  f"報酬={r['shaped_return']:+8.2f} 速度の向きに押した={100 * r['along']:3.0f}% 行動[左/無/右]={r['actions']} eps={eps:.3f} "
                  f"連続={streak} ({time.time() - t0:.0f}s)", flush=True)
            if first_success and first_success["episode"] == ep:
                print(f"  >>> 初成功: {ep} 回目の試行（累計 {total_steps:,} ステップ）", flush=True)
            if mastered and mastered["episode"] == ep:
                print(f"  >>> 習得: {ep} 回目で {args.streak} 回連続ゴール", flush=True)
                break
            eps = max(args.eps_min, eps * args.eps_decay)

    brain.ask(cmd="save", path=os.path.join(run_dir, "gains.bin.gz"))

    # 習得後: 学習も探索も止めて、覚えたことだけで走らせる
    evaluation = None
    if args.eval > 0:
        ev = [run_episode(env, brain, seed=10**7 + i, eps=0.0, learn=False) for i in range(args.eval)]
        ok = [e for e in ev if e["reached_goal"]]
        evaluation = dict(episodes=len(ev), successes=len(ok),
                          mean_steps_success=float(np.mean([e["steps"] for e in ok])) if ok else None)
    brain.close()
    env.close()

    summary = dict(
        episodes_run=len(rows), total_steps=total_steps,
        first_success=first_success, mastered=mastered, streak_required=args.streak,
        successes=sum(r["reached_goal"] for r in rows),
        evaluation_without_learning=evaluation,
        wall_seconds=round(time.time() - t0, 1), args=vars(args),
        brain=dict(neurons=brain.info["neurons"], kc=brain.info["kc"], pn=brain.info["pn"],
                   compartments=brain.info["compartments"]),
    )
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n================ 結果 ================")
    if first_success:
        print(f"初成功        : {first_success['episode']} 回目の試行 "
              f"（それまでの行動 {first_success['cum_steps']:,} ステップ, そのエピソードは {first_success['steps']} ステップでゴール）")
    else:
        print(f"初成功        : {len(rows)} 回の試行では一度もゴールせず")
    if mastered:
        print(f"習得          : {mastered['episode']} 回目の試行で {args.streak} 回連続ゴール（累計 {mastered['cum_steps']:,} ステップ）")
    else:
        print(f"習得          : {args.streak} 回連続ゴールには至らず")
    print(f"学習中のゴール: {summary['successes']} / {len(rows)} エピソード")
    if evaluation:
        ms = evaluation["mean_steps_success"]
        print(f"学習停止後テスト: {evaluation['successes']} / {evaluation['episodes']} 回ゴール"
              + (f"（平均 {ms:.1f} ステップ）" if ms else ""))
    print(f"記録          : {run_dir}")

    try:
        plot(rows, first_success, mastered, os.path.join(run_dir, "learning_curve.png"))
    except ImportError:
        pass


def plot(rows, first_success, mastered, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
    ep = [r["episode"] for r in rows]
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(ep, [r["steps"] for r in rows], color="#4a6fa5", lw=1.5)
    ax[0].scatter([r["episode"] for r in rows if r["reached_goal"]],
                  [r["steps"] for r in rows if r["reached_goal"]], color="#d1495b", s=14, zorder=3, label="ゴール")
    ax[0].set_ylabel("ステップ数 (200 = 時間切れ)")
    ax[0].legend(loc="upper right")
    ax[1].plot(ep, [r["max_position"] for r in rows], color="#4a6fa5", lw=1.5)
    ax[1].axhline(0.5, color="#999", ls="--", lw=1)
    ax[1].set_ylabel("最高到達位置")
    ax[1].set_xlabel("エピソード（試行回数）")
    for a in ax:
        if first_success:
            a.axvline(first_success["episode"], color="#d1495b", ls=":", lw=1)
        if mastered:
            a.axvline(mastered["episode"], color="#2e7d32", ls=":", lw=1)
    title = "ハエのキノコ体が Mountain Car を学習"
    if first_success:
        title += f" — 初成功 {first_success['episode']} 回目"
    if mastered:
        title += f" / 習得 {mastered['episode']} 回目"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
