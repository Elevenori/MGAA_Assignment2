import pandas as pd
from evaluation import run_tournament


c_values_to_test = [0.5]
results = []

print("Please make sure that the three Baseline processes are running in the background!")

for c in c_values_to_test:
    input(f"\n Open mcts_agent.py, change EXPERIMENT_C to {c}, and save the file.\n"
          f" Then restart `python mcts_agent.py`\n"
          f" When you're ready, press the [Enter] key to start the 50 races.")

    elo, win_rate, avg_turns, avg_iters = run_tournament(
        test_agent_name=f"MCTS_Heuristic_{c}", matches=50)

    results.append({
        "C_Value": c,
        "ELO": elo,
        "Win_Rate": win_rate,
        "Avg_Turns": avg_turns,
        "Avg_Iters": avg_iters
    })

df = pd.DataFrame(results)

print(df.to_string(index=False))

# save to CSV
df.to_csv("mcts_c_tuning_results3.csv", index=False)
print("\nsave to mcts_c_tuning_results3.csv")
