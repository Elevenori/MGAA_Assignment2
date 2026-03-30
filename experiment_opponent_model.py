"""
Opponent Modeling MCTS - C Value Hyperparameter Tuning
=======================================================
Tests C values around the known optimal (0.5 from vanilla MCTS tuning).
Since opponent modeling produces higher quality simulations, the optimal
C may be smaller than or equal to 0.5.

Usage:
    Terminal 1: python mcts_opponent_model.py   (port 8000) <- restart for each C
    Terminal 2: PORT=8001 python mcts_agent.py
    Terminal 3: PORT=8002 python mcts_agent.py
    Terminal 4: PORT=8003 python mcts_agent.py
    Terminal 5: python experiment_opponent_model.py
"""

import pandas as pd
from evaluation import run_tournament
from plot import plot_C_results

c_values = [0.3, 0.5, 0.7]
results  = []

print("Please ensure the following 3 baseline snakes are running:")
print("    PORT=8001 python heuristic_agent.py")
print("    PORT=8002 python heuristic_agent.py")
print("    PORT=8003 python heuristic_agent.py")
print("Opponent Modeling agent (mcts_opponent_model.py) runs on port 8000\n")

for c in c_values:
    input(f"\n>>> Open mcts_opponent_model.py and set:\n"
          f"        EXPERIMENT_C = {c}\n"
          f"    Save the file, then restart: python mcts_opponent_model.py\n"
          f"    Press [ENTER] when ready to run 50 matches...")

    elo, win_rate, avg_turns, avg_iters = run_tournament(
        test_agent_name=f"OpponentModel_C{c}", matches=50)

    results.append({
        "C_Value":   c,
        "ELO":       elo,
        "Win_Rate":  win_rate,
        "Avg_Turns": avg_turns,
        "Avg_Iters": avg_iters,
    })
    print(f"  C={c} | ELO={elo:.0f} | Win Rate={win_rate:.1f}%")

# ── Results ────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)

print("\n===== Opponent Modeling C-Value Tuning Results =====")
print(df.to_string(index=False))

df.to_csv("opponent_model_c_tuning_results.csv", index=False)
print("\nResults saved to opponent_model_c_tuning_results.csv")

best_c = df.loc[df["ELO"].idxmax(), "C_Value"]
print(f"\nBest C value: {best_c} (ELO = {df['ELO'].max():.0f})")
print(f">>> Set EXPERIMENT_C = {best_c} in mcts_opponent_model.py as the final agent")

plot_C_results(df, title="Opponent Modeling MCTS: C-Value Hyperparameter Tuning")
