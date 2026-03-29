"""
poolRave 超参数调参实验
对应论文 Table 1 的实验设计：
    第一轮：固定 pool_size=10，调 p
    第二轮：固定最优 p，调 pool_size
"""
import pandas as pd
from evaluation import run_tournament
from plot import plot_C_results

print("⚠️  请确保以下 3 条 Baseline 蛇已在后台运行：")
print("    PORT=8001 python heuristic_agent.py")
print("    PORT=8002 python heuristic_agent.py")
print("    PORT=8003 python heuristic_agent.py")
print("⚠️  RAVE agent (mcts_rave.py) 运行在端口 8000\n")

# ══════════════════════════════════════════
# 第一轮：固定 pool_size=10，调 p
# 论文测试值: 1/4, 1/2, 3/4
# ══════════════════════════════════════════
p_values = [0.25, 0.5, 0.75]
results_p = []

print("=" * 50)
print("第一轮：固定 EXPERIMENT_POOL=10，调 EXPERIMENT_P")
print("=" * 50)

for p in p_values:
    input(f"\n👉 打开 mcts_rave.py，设置：\n"
          f"     EXPERIMENT_P    = {p}\n"
          f"     EXPERIMENT_POOL = 10\n"
          f"👉 保存后重启：python mcts_rave.py\n"
          f"👉 准备好后按 【回车键】 开始跑 50 场...")

    elo, win_rate, avg_turns, avg_iters = run_tournament(
        test_agent_name=f"poolRave_p{p}_pool10", matches=50)

    results_p.append({
        "C_Value":   p,
        "ELO":       elo,
        "Win_Rate":  win_rate,
        "Avg_Turns": avg_turns,
        "Avg_Iters": avg_iters
    })
    print(f"  p={p} | ELO={elo:.0f} | 胜率={win_rate:.1f}%")

df_p = pd.DataFrame(results_p)
print("\n===== p 调参结果 =====")
print(df_p.to_string(index=False))
df_p.to_csv("rave_p_tuning_results.csv", index=False)
plot_C_results(df_p, title="poolRave: p Hyperparameter Tuning (pool_size=10)")

best_p = df_p.loc[df_p["ELO"].idxmax(), "C_Value"]
print(f"\n✅ 第一轮最优 p = {best_p}")

# ══════════════════════════════════════════
# 第二轮：固定最优 p，调 pool_size
# 论文测试值: 5, 10, 20
# ══════════════════════════════════════════
pool_values = [5, 10, 20]
results_pool = []

print("\n" + "=" * 50)
print(f"第二轮：固定 EXPERIMENT_P={best_p}，调 EXPERIMENT_POOL")
print("=" * 50)

for pool in pool_values:
    input(f"\n👉 打开 mcts_rave.py，设置：\n"
          f"     EXPERIMENT_P    = {best_p}\n"
          f"     EXPERIMENT_POOL = {pool}\n"
          f"👉 保存后重启：python mcts_rave.py\n"
          f"👉 准备好后按 【回车键】 开始跑 50 场...")

    elo, win_rate, avg_turns, avg_iters = run_tournament(
        test_agent_name=f"poolRave_p{best_p}_pool{pool}", matches=50)

    results_pool.append({
        "C_Value":   pool,
        "ELO":       elo,
        "Win_Rate":  win_rate,
        "Avg_Turns": avg_turns,
        "Avg_Iters": avg_iters
    })
    print(f"  pool_size={pool} | ELO={elo:.0f} | 胜率={win_rate:.1f}%")

df_pool = pd.DataFrame(results_pool)
print("\n===== pool_size 调参结果 =====")
print(df_pool.to_string(index=False))
df_pool.to_csv("rave_pool_tuning_results.csv", index=False)
plot_C_results(df_pool, title=f"poolRave: pool_size Tuning (p={best_p})")

best_pool = df_pool.loc[df_pool["ELO"].idxmax(), "C_Value"]
print(f"\n✅ 最终最优参数：p={best_p}, pool_size={int(best_pool)}")
print(f"👉 把这两个值填回 mcts_rave.py 的调参区，作为最终 agent 使用")
