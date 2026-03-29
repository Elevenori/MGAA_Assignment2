import subprocess
import json


class EloSystem:
    def __init__(self, names):
        self.ratings = {name: 1000 for name in names}
        self.K = 32

    def update(self, winner_name):
        if winner_name not in self.ratings:
            return

        # 赢家从所有输家身上吸取分数
        for snake in self.ratings:
            if snake == winner_name:
                continue
            expected_win = 1.0 / \
                (1.0 + 10 **
                 ((self.ratings[snake] - self.ratings[winner_name]) / 400.0))
            self.ratings[winner_name] += self.K * (1 - expected_win)
            self.ratings[snake] += self.K * (0 - (1 - expected_win))


def run_tournament(test_agent_name="MCTS_Test", matches=50):
    """
    运行锦标赛并返回测试蛇的统计数据
    """
    snakes_config = [
        {"name": test_agent_name, "url": "http://127.0.0.1:8000"},
        {"name": "Base8001", "url": "http://127.0.0.1:8001"},
        {"name": "Base8002", "url": "http://127.0.0.1:8002"},
        {"name": "Base8003", "url": "http://127.0.0.1:8003"}
    ]

    elo_system = EloSystem([s["name"] for s in snakes_config])
    mcts_survival_turns = []
    wins = {s["name"]: 0 for s in snakes_config}
    wins["Draw/Timeout"] = 0

    # 实验开始前，通过写入空内容清空记录文件
    try:
        open("mcts_iters.txt", "w").close()
    except Exception:
        pass

    print(f"\n Start [{test_agent_name}] totally {matches} matches ...")

    for i in range(matches):
        print(f" Running Match {i+1}/{matches} ", end="\r", flush=True)

        cmd = [
            "battlesnake", "play",
            "-W", "11", "-H", "11",
            "-g", "standard",
            "-m", "hz_hazard_pits",
            "--foodSpawnChance", "25",
            "--minimumFood", "2",
            "--timeout", "1000",
            "--output", "game_result.json"
        ]

        for s in snakes_config:
            cmd.extend(["--name", s["name"], "--url", s["url"]])

        subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

        last_state = None
        try:
            with open("game_result.json", "r") as f:
                lines = f.readlines()
                if not lines:
                    continue

                for line in reversed(lines):
                    if line.strip():
                        temp_data = json.loads(line)
                        if "board" in temp_data:
                            last_state = temp_data
                            break
        except Exception as e:
            print(f" Error: {e}")
            continue

        if not last_state:
            continue

        turn = last_state.get("turn", 0)
        mcts_survival_turns.append(turn)
        survivors = last_state["board"]["snakes"]

        if len(survivors) == 1:
            winner = survivors[0]["name"]
            wins[winner] += 1
            elo_system.update(winner)
        else:
            wins["Draw/Timeout"] += 1

    print(f"\n All {matches} matches completed!")

    # 读取所有回合的迭代次数并算平均值
    avg_iters = 0
    try:
        with open("mcts_iters.txt", "r") as f:
            lines = f.readlines()
            iters = [int(l.strip()) for l in lines if l.strip().isdigit()]
            if iters:
                avg_iters = sum(iters) / len(iters)
    except Exception:
        print("Cannot read mcts_iters.txt")

    avg_turns = sum(mcts_survival_turns) / \
        len(mcts_survival_turns) if mcts_survival_turns else 0
    final_elo = elo_system.ratings[test_agent_name]
    win_rate = (wins[test_agent_name] / matches) * 100

    print(f" ELO: {final_elo:.0f} ; win rate: {win_rate:.1f}% ; avg survival: {avg_turns:.1f} 轮 | avg iters: {avg_iters:.1f} 次/步")

    return final_elo, win_rate, avg_turns, avg_iters
