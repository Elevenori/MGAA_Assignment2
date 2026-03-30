"""
Local Tournament Script
=======================
Runs all 4 agents simultaneously and tracks ELO for each.

Usage:
    Terminal 1: python mcts_rave.py                      (port 8000)
    Terminal 2: PORT=8001 python mcts_opponent_model.py  (port 8001)
    Terminal 3: PORT=8002 python mcts_agent.py           (port 8002)
    Terminal 4: PORT=8003 python heuristic_agent.py      (port 8003)
    Terminal 5: python tournament.py
"""

import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────
#  Tournament Configuration
# ─────────────────────────────────────────
MATCHES     = 100
LOG_PATH    = Path("tournament_result.json")
RESULTS_CSV = "tournament_results.csv"

SNAKES = [
    {"name": "RAVE",          "url": "http://127.0.0.1:8000"},
    {"name": "OpponentModel", "url": "http://127.0.0.1:8001"},
    {"name": "VanillaMCTS",   "url": "http://127.0.0.1:8002"},
    {"name": "Heuristic",     "url": "http://127.0.0.1:8003"},
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
#  ELO System (tracks all snakes)
# ─────────────────────────────────────────
class EloSystem:
    def __init__(self, names, k=32):
        self.ratings = {name: 1000.0 for name in names}
        self.K = k

    def update(self, winner_name: str, losers: list):
        """Winner gains points from each loser."""
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
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────
#  Parse Game Result
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
#  Main Tournament Logic
# ─────────────────────────────────────────
def run_tournament():
    names = [s["name"] for s in SNAKES]
    elo   = EloSystem(names)
    wins  = {name: 0 for name in names}
    wins["Draw"] = 0

    # ELO history for trend plot
    elo_history = {name: [] for name in names}

    print(f"\nStarting tournament: {MATCHES} matches")
    print(f"Agents: {' | '.join(names)}\n")

    for i in range(MATCHES):
        print(f"  Match {i+1}/{MATCHES}", end="\r", flush=True)

        if LOG_PATH.exists():
            LOG_PATH.unlink()

        cmd = CMD_BASE.copy()
        for s in SNAKES:
            cmd.extend(["--name", s["name"], "--url", s["url"]])

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        state = load_last_state(LOG_PATH)
        if not state:
            continue

        survivors = state["board"]["snakes"]

        # Determine winner
        if len(survivors) == 1:
            winner = survivors[0]["name"]
            losers = [n for n in names if n != winner]
            wins[winner] += 1
            elo.update(winner, losers)
        else:
            wins["Draw"] += 1

        # Record ELO after each match
        for name in names:
            elo_history[name].append(elo.ratings[name])

    print(f"\nAll {MATCHES} matches completed!\n")
    return elo, wins, elo_history


# ─────────────────────────────────────────
#  Display Results
# ─────────────────────────────────────────
def print_results(elo, wins, matches):
    print("=" * 52)
    print("         Tournament Final Results")
    print("=" * 52)
    print(f"{'Rank':<6} {'Agent':<20} {'ELO':>8} {'Win Rate':>10}")
    print("-" * 52)

    for rank, (name, rating) in enumerate(elo.standings(), 1):
        win_rate = (wins[name] / matches) * 100
        print(f"{rank:<6} {name:<20} {rating:>8.1f} {win_rate:>9.1f}%")

    print("-" * 52)
    print(f"{'':6} {'Draw/Timeout':<20} {wins['Draw']:>8} matches")
    print("=" * 52)


def save_results(elo, wins, matches):
    rows = []
    for name, rating in elo.standings():
        rows.append({
            "Agent":    name,
            "ELO":      round(rating, 2),
            "Wins":     wins[name],
            "Win_Rate": round(wins[name] / matches * 100, 1),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved to {RESULTS_CSV}")
    return df


def plot_results(df, elo_history, matches):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = ["#FF6B6B", "#A855F7", "#00CC00", "#50B946"]

    # Plot 1: Final ELO bar chart
    axes[0].bar(df["Agent"], df["ELO"], color=colors)
    axes[0].axhline(y=1000, color="gray", linestyle="--",
                    alpha=0.5, label="Baseline (1000)")
    axes[0].set_title("Final ELO Ratings")
    axes[0].set_ylabel("ELO")
    axes[0].set_xlabel("Agent")
    axes[0].legend()
    for i, (_, row) in enumerate(df.iterrows()):
        axes[0].text(i, row["ELO"] + 5, f"{row['ELO']:.0f}",
                     ha="center", fontsize=10)

    # Plot 2: Win rate bar chart
    axes[1].bar(df["Agent"], df["Win_Rate"], color=colors)
    axes[1].axhline(y=25, color="gray", linestyle="--",
                    alpha=0.5, label="Random baseline (25%)")
    axes[1].set_title("Win Rate (%)")
    axes[1].set_ylabel("Win Rate (%)")
    axes[1].set_xlabel("Agent")
    axes[1].legend()
    for i, (_, row) in enumerate(df.iterrows()):
        axes[1].text(i, row["Win_Rate"] + 0.3, f"{row['Win_Rate']:.1f}%",
                     ha="center", fontsize=10)

    # Plot 3: ELO progression
    for (name, history), color in zip(elo_history.items(), colors):
        axes[2].plot(range(1, len(history)+1), history,
                     label=name, color=color, linewidth=1.5)
    axes[2].axhline(y=1000, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_title("ELO Progression Over Matches")
    axes[2].set_xlabel("Match Number")
    axes[2].set_ylabel("ELO")
    axes[2].legend()

    plt.suptitle(f"BattleSnake Tournament Results ({matches} matches)",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig("tournament_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved to tournament_results.png")


# ─────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Please ensure all 4 snakes are running:")
    for s in SNAKES:
        print(f"    {s['name']:<20} -> {s['url']}")
    print()
    input("Press [ENTER] when all snakes are ready...")

    elo, wins, elo_history = run_tournament()

    print_results(elo, wins, MATCHES)
    df = save_results(elo, wins, MATCHES)
    plot_results(df, elo_history, MATCHES)
