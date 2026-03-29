import pandas as pd
from evaluation import run_tournament
from plot import plot_C_results

c_values_to_test = [0.2, 2.0]
results = []

print("⚠️ 请确保 3 条 Baseline 蛇已经启动，并在后台运行！")

for c in c_values_to_test:
    # 这一步是暂停程序，等你手动去改代码并重启测试蛇
    input(f"\n👉 请打开 mcts_agent.py，将 EXPERIMENT_C 改为 {c}，保存。\n"
          f"👉 然后在终端重启 `python mcts_agent.py`。\n"
          f"👉 准备好后，按 【回车键】 开始跑 50 场比赛...")

    # 自动跑 50 场
    elo, win_rate, avg_turns, avg_iters = run_tournament(
        test_agent_name=f"MCTS_Heuristic_{c}", matches=50)

    # 记录进字典
    results.append({
        "C_Value": c,
        "ELO": elo,
        "Win_Rate": win_rate,
        "Avg_Turns": avg_turns,
        "Avg_Iters": avg_iters
    })

# ==========================================
# 实验结束，生成 DataFrame 并画图
# ==========================================
df = pd.DataFrame(results)

print(df.to_string(index=False))  # 打印出漂亮的表格

# 保存为 CSV，以防数据丢失
df.to_csv("mcts_c_tuning_results3.csv", index=False)
print("\nsave to mcts_c_tuning_results3.csv")

# 直接把 DataFrame 里的列抽出来喂给画图函数
plot_C_results(df, title="Heuristic MCTS C-Value Hyperparameter Tuning")
