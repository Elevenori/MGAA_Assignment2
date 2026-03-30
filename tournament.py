"""
本地锦标赛脚本
==============
让4个 agent 同时对打，同时追踪所有蛇的 ELO。

使用方法：
    终端1: python mcts_rave.py                      (port 8000)
    终端2: PORT=8001 python mcts_opponent_model.py  (port 8001)
    终端3: PORT=8002 python mcts_agent.py           (port 8002)
    终端4: PORT=8003 python heuristic_agent.py      (port 8003)
    终端5: python tournament.py
"""

import json
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────
#  锦标赛配置
# ─────────────────────────────────────────
MATCHES     = 100       # 总对局数（建议至少100局保证统计显著性）
LOG_PATH    = Path("tournament_result.json")
RESULTS_CSV = "tournament_results.csv"

SNAKES = [
    {"name": "RAVE",           "url": "http://127.0.0.1:8000"},
    {"name": "OpponentModel",  "url": "http://127.0.0.1:8001"},
    {"name": "VanillaMCTS",    "url": "http://127.0.0.1:8002"},
    {"name": "Heuristic",      "url": "http://127.0.0.1:8003"},
]

CMD_BASE = [
    "battlesnake", "play",
    "-W", "11", "-H", "11",
    "-g", "standard",
    "-m", "hz_hazard_pits",
    "--foodSpawnChance", "25",
    "--minimumFood", "2",
    "--timeout", "1000",
    "--output", str(LOG_PATH),
]


# ─────────────────────────────────────────
#  ELO 系统（追踪所有蛇）
# ─────────────────────────────────────────
class EloSystem:
    def __init__(self, names, k=32):
        self.ratings = {name: 1000.0 for name in names}
        self.K = k

    def update(self, winner_name: str, losers: list):
        """赢家从每个输家身上吸取分数"""
        if winner_name not in self.ratings:
            return
        for loser in losers:
            if loser not in self.ratings or loser == winner_name:
                continue
            exp_win = 1.0 / (1.0 + 10 ** (
                (self.ratings[loser] - self.ratings[winner_name]) / 400.0))
            self.ratings[winner_name] += self.K * (1 - exp_win)
            self.ratings[loser]       += self.K * (0 - (1 - exp_win))

    def standings(self) -> list:
        """返回按 ELO 排序的列表"""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────
#  读取游戏结果
# ─────────────────────────────────────────
def load_last_state(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            if line.strip():
                obj = json.loads(line)
                if isinstance(obj, dict) and "board" in obj:
                    return obj
    except Exception:
        return None
    return None


# ─────────────────────────────────────────
#  主锦标赛逻辑
# ─────────────────────────────────────────
def run_tournament():
    names     = [s["name"] for s in SNAKES]
    elo       = EloSystem(names)
    wins      = {name: 0 for name in names}
    wins["Draw"] = 0
    survival  = {name: [] for name in names}  # 每局存活回合

    # ELO 历史（用于画趋势图）
    elo_history = {name: [] for name in names}

    print(f"\n🐍 开始锦标赛：{MATCHES} 场对局")
    print(f"参赛选手：{' | '.join(names)}\n")

    for i in range(MATCHES):
        print(f"  对局 {i+1}/{MATCHES}", end="\r", flush=True)

        if LOG_PATH.exists():
            LOG_PATH.unlink()

        cmd = CMD_BASE.copy()
        for s in SNAKES:
            cmd.extend(["--name", s["name"], "--url", s["url"]])

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        state = load_last_state(LOG_PATH)
        if not state:
            continue

        turn      = state.get("turn", 0)
        survivors = state["board"]["snakes"]
        all_names = {s["name"] for s in SNAKES}

        # 记录存活回合（包括死亡的蛇，用 turn 估算）
        survivor_names = {s["name"] for s in survivors}
        for name in names:
            if name in survivor_names:
                survival[name].append(turn)
            else:
                survival[name].append(turn)

        # 判断胜负
        if len(survivors) == 1:
            winner = survivors[0]["name"]
            losers = [n for n in names if n != winner]
            wins[winner] += 1
            elo.update(winner, losers)
        else:
            wins["Draw"] += 1

        # 记录每局后的 ELO
        for name in names:
            elo_history[name].append(elo.ratings[name])

    print(f"\n✅ {MATCHES} 场对局完成！\n")
    return elo, wins, survival, elo_history


# ─────────────────────────────────────────
#  结果展示
# ─────────────────────────────────────────
def print_results(elo, wins, survival, matches):
    names = [s["name"] for s in SNAKES]

    print("=" * 60)
    print("           锦标赛最终结果")
    print("=" * 60)
    print(f"{'排名':<4} {'Agent':<20} {'ELO':>8} {'胜率':>8} {'平均存活':>10}")
    print("-" * 60)

    standings = elo.standings()
    for rank, (name, rating) in enumerate(standings, 1):
        win_rate  = (wins[name] / matches) * 100
        avg_surv  = sum(survival[name]) / len(survival[name]) if survival[name] else 0
        print(f"{rank:<4} {name:<20} {rating:>8.1f} {win_rate:>7.1f}% {avg_surv:>10.1f}")

    print("-" * 60)
    print(f"     {'Draw/Timeout':<20} {wins['Draw']:>8} 局")
    print("=" * 60)


def save_results(elo, wins, survival, matches):
    names = [s["name"] for s in SNAKES]
    rows  = []
    for name, rating in elo.standings():
        rows.append({
            "Agent":      name,
            "ELO":        round(rating, 2),
            "Wins":       wins[name],
            "Win_Rate":   round(wins[name] / matches * 100, 1),
            "Avg_Turns":  round(sum(survival[name]) / len(survival[name]), 1)
                          if survival[name] else 0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n📄 结果已保存到 {RESULTS_CSV}")
    return df


def plot_results(df, elo_history, matches):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 图1：最终 ELO 柱状图
    axes[0].bar(df["Agent"], df["ELO"], color=["#FF6B6B","#A855F7","#00CC00","#50B946"])
    axes[0].axhline(y=1000, color="gray", linestyle="--", alpha=0.5, label="Baseline (1000)")
    axes[0].set_title("Final ELO Ratings")
    axes[0].set_ylabel("ELO")
    axes[0].set_xlabel("Agent")
    axes[0].legend()
    for i, (_, row) in enumerate(df.iterrows()):
        axes[0].text(i, row["ELO"] + 5, f"{row['ELO']:.0f}", ha="center", fontsize=10)

    # 图2：胜率柱状图
    axes[1].bar(df["Agent"], df["Win_Rate"], color=["#FF6B6B","#A855F7","#00CC00","#50B946"])
    axes[1].axhline(y=25, color="gray", linestyle="--", alpha=0.5, label="Random baseline (25%)")
    axes[1].set_title("Win Rate (%)")
    axes[1].set_ylabel("Win Rate (%)")
    axes[1].set_xlabel("Agent")
    axes[1].legend()
    for i, (_, row) in enumerate(df.iterrows()):
        axes[1].text(i, row["Win_Rate"] + 0.3, f"{row['Win_Rate']:.1f}%", ha="center", fontsize=10)

    # 图3：ELO 趋势图
    colors = ["#FF6B6B", "#A855F7", "#00CC00", "#50B946"]
    for (name, history), color in zip(elo_history.items(), colors):
        axes[2].plot(range(1, len(history)+1), history,
                     label=name, color=color, linewidth=1.5)
    axes[2].axhline(y=1000, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_title("ELO Progression Over Matches")
    axes[2].set_xlabel("Match Number")
    axes[2].set_ylabel("ELO")
    axes[2].legend()

    plt.suptitle(f"BattleSnake Tournament Results ({matches} matches)", fontsize=14)
    plt.tight_layout()
    plt.savefig("tournament_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 图表已保存到 tournament_results.png")


# ─────────────────────────────────────────
#  入口
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("⚠️  请确保以下4条蛇都已启动：")
    for s in SNAKES:
        print(f"    {s['name']:<20} → {s['url']}")
    print()
    input("👉 确认全部启动后，按 【回车键】 开始锦标赛...")

    elo, wins, survival, elo_history = run_tournament()

    print_results(elo, wins, survival, MATCHES)
    df = save_results(elo, wins, survival, MATCHES)
    plot_results(df, elo_history, MATCHES)
