"""ドーパミンの条件を変えて train.py を何本も走らせ、結果を並べる。

    py -3.12 mountaincar/compare.py                              # 4 条件 × seed 1-5, 各 100 エピソードまで
    py -3.12 mountaincar/compare.py --dopa reward none --seeds 1 2 3 --episodes 50
    py -3.12 mountaincar/compare.py --out mountaincar/runs/compare-<日時>   # 止まったところから

条件（train.py --dopa）:
  reward  — 報酬をそのままドーパミンに（本来の条件）
  none    — ドーパミンなし（可塑性はオンのまま）
  shuffle — ドーパミンはあるが報酬と無関係（符号をランダムに反転、大きさは同じ）
  goal    — ゴールしたときだけドーパミン

記録は mountaincar/runs/compare-<日時>/ に、各 run のディレクトリと results.csv・summary.md・compare.json
（matplotlib があれば compare.png）。compare.json を mountaincar/ にコピーするとブラウザ版に結果が出る。
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {"reward": "報酬→ドーパミン", "none": "ドーパミンなし",
          "shuffle": "無関係なドーパミン", "goal": "ゴール時のみ"}


def find_run(out, dopa, seed):
    """out の中にある、この条件・seed の run ディレクトリ（なければ None）。"""
    tail = f"-seed{seed}-eta6e-05" + ("" if dopa == "reward" else f"-dopa{dopa}")
    for d in sorted(os.listdir(out)):
        if d.endswith(tail) and os.path.isdir(os.path.join(out, d)):
            return os.path.join(out, d)
    return None


def run_one(out, dopa, seed, a, tries=3):
    run_dir = find_run(out, dopa, seed)
    if run_dir and not os.path.exists(os.path.join(run_dir, "summary.json")):
        shutil.rmtree(run_dir)            # 途中で止まった run はやり直す
        run_dir = None
    cmd = [sys.executable, os.path.join(HERE, "train.py"), "--dopa", dopa, "--seed", str(seed),
           "--episodes", str(a.episodes), "--streak", str(a.streak), "--eval", str(a.eval), "--out", out]
    log = os.path.join(out, f"{dopa}-seed{seed}.log")
    for attempt in range(1, tries + 1):
        if run_dir:
            break
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        run_dir = find_run(out, dopa, seed)
        if p.returncode == 0 and run_dir:
            break
        # 脳を同時に読み込みすぎるとメモリが足りず落ちることがある: 少し待ってやり直す
        print(f"  {LABELS[dopa]} seed {seed}: 失敗（{attempt}/{tries} 回目, ログ {log}）", flush=True)
        if run_dir:
            shutil.rmtree(run_dir)
            run_dir = None
        time.sleep(20 * attempt)
    if not run_dir:
        return None
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    with open(os.path.join(run_dir, "episodes.csv"), encoding="utf-8") as f:
        eps = list(csv.DictReader(f))
    ev = s["evaluation_without_learning"] or {}
    r = dict(dopa=dopa, seed=seed, episodes_run=s["episodes_run"], successes=s["successes"],
             first_success=(s["first_success"] or {}).get("episode"),
             first_success_steps=(s["first_success"] or {}).get("cum_steps"),
             mastered=(s["mastered"] or {}).get("episode"),
             eval_successes=ev.get("successes"), eval_episodes=ev.get("episodes"),
             along=float(np.mean([float(e["along_velocity"]) for e in eps])),
             max_position=max(float(e["max_position"]) for e in eps),
             steps=[int(e["steps"]) for e in eps])
    print(f"  {LABELS[dopa]:<10} seed {seed}: 初成功 {r['first_success'] or '—'} / 習得 {r['mastered'] or '—'} / "
          f"ゴール {r['successes']}/{r['episodes_run']} / 停止後 {r['eval_successes']}/{r['eval_episodes']}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dopa", nargs="+", default=list(LABELS), choices=list(LABELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--streak", type=int, default=10)
    ap.add_argument("--eval", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=3, help="同時に走らせる run の数（1 本ごとに脳を 1 つ読み込む）")
    ap.add_argument("--out", default=None, help="この compare-* ディレクトリの続きから（済んだ run は使い回す）")
    a = ap.parse_args()

    out = a.out or os.path.join(HERE, "runs", "compare-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    print(f"{len(a.dopa)} 条件 × {len(a.seeds)} seed、各 {a.episodes} エピソードまで → {out}", flush=True)
    with ThreadPoolExecutor(a.jobs) as pool:
        rows = list(pool.map(lambda job: run_one(out, *job, a), [(d, s) for d in a.dopa for s in a.seeds]))
    failed = len([r for r in rows if r is None])
    rows = [r for r in rows if r is not None]
    if failed:
        print(f"\n{failed} 本の run が失敗した（--out {out} で続きから）", flush=True)
    if not rows:
        return

    with open(os.path.join(out, "results.csv"), "w", newline="", encoding="utf-8") as f:
        keys = [k for k in rows[0] if k != "steps"]
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    md = [f"| 条件 | 初成功（試行） | 習得（{a.streak} 連続） | 学習中のゴール | 学習停止後 | 速度の向きに押した | 最高位置 |",
          "|---|---|---|---|---|---|---|"]
    for d in a.dopa:
        rs = [r for r in rows if r["dopa"] == d]
        if not rs:
            continue
        fs = [r["first_success"] for r in rs if r["first_success"]]
        ms = [r["mastered"] for r in rs if r["mastered"]]
        ev = sum(r["eval_successes"] or 0 for r in rs), sum(r["eval_episodes"] or 0 for r in rs)
        md.append(f"| {LABELS[d]} | {len(fs)}/{len(rs)} seed" + (f"（中央値 {np.median(fs):g} 回目）" if fs else "") +
                  f" | {len(ms)}/{len(rs)} seed" + (f"（中央値 {np.median(ms):g} 回目）" if ms else "") +
                  f" | {sum(r['successes'] for r in rs)}/{sum(r['episodes_run'] for r in rs)}"
                  f" | {ev[0]}/{ev[1]} | {100 * np.mean([r['along'] for r in rs]):.0f}%"
                  f" | {np.mean([r['max_position'] for r in rs]):+.2f} |")
    text = "\n".join(md)
    write_json(rows, a, os.path.join(out, "compare.json"))
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text + f"\n\n記録: {out}")

    try:
        plot(rows, a, os.path.join(out, "compare.png"))
    except ImportError:
        pass


def padded_steps(rows, a, d):
    """条件 d の run ごとのステップ数（習得して打ち切った run は、その後も最後の --streak 回の平均で走るとみなして伸ばす）。"""
    return np.array([r["steps"] + [np.mean(r["steps"][-a.streak:])] * (a.episodes - len(r["steps"]))
                     for r in rows if r["dopa"] == d])


def write_json(rows, a, path):
    """ブラウザ版（index.html）に載せる結果。mountaincar/compare.json にコピーするとページに出る。"""
    conds = []
    for d in a.dopa:
        rs = [r for r in rows if r["dopa"] == d]
        if not rs:
            continue
        m = padded_steps(rows, a, d)
        conds.append(dict(
            dopa=d, label=LABELS[d], seeds=[r["seed"] for r in rs],
            mean=[round(float(x), 1) for x in m.mean(0)], min=m.min(0).astype(int).tolist(),
            max=m.max(0).astype(int).tolist(),
            first_success=[r["first_success"] for r in rs], mastered=[r["mastered"] for r in rs],
            successes=sum(r["successes"] for r in rs), episodes=sum(r["episodes_run"] for r in rs),
            eval_successes=sum(r["eval_successes"] or 0 for r in rs), eval_episodes=sum(r["eval_episodes"] or 0 for r in rs),
            along=round(float(np.mean([r["along"] for r in rs])), 4),
            max_position=round(float(np.mean([r["max_position"] for r in rs])), 4)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(date=datetime.now().strftime("%Y-%m-%d"), episodes=a.episodes, streak=a.streak,
                       eval=a.eval, conditions=conds), f, ensure_ascii=False, separators=(",", ":"))


def plot(rows, a, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
    colors = {"reward": "#d1495b", "none": "#4a6fa5", "shuffle": "#edae49", "goal": "#66a182"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for d in a.dopa:
        if not any(r["dopa"] == d for r in rows):
            continue
        m = padded_steps(rows, a, d)
        x = np.arange(1, a.episodes + 1)
        ax.plot(x, m.mean(0), color=colors[d], lw=2, label=LABELS[d])
        ax.fill_between(x, m.min(0), m.max(0), color=colors[d], alpha=0.15, lw=0)
    ax.set_ylim(0, 205)
    ax.set_xlabel("エピソード（試行回数）")
    ax.set_ylabel("ステップ数（200 = 時間切れ）")
    ax.legend(loc="lower left")
    ax.set_title(f"ドーパミンの条件別の学習（seed {len(a.seeds)} 本の平均、帯は最小〜最大）")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
