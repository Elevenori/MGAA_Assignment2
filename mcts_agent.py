import math
import random
import time
import copy
from heuristic_agent import move_score
# ─────────────────────────────────────────
#  实验调参区
# ─────────────────────────────────────────
EXPERIMENT_C = 1.41         # 测试目标：调整 UCB 探索常数 (比如 0.5, 1.41, 2.0)
USE_HEURISTIC = True      # 测试目标：True 为启发式 Rollout，False 为纯随机 Rollout


def heuristic_adapter(game_state_obj: 'GameState') -> float:
    """
    将 GameState 转换为字典格式，并调用队友的 move_score
    由于队友的函数必须传入一个 'move'，我们将评估所有合法 move 的最高分
    作为当前局面的启发式得分。
    """
    my_snake = game_state_obj.snakes.get(game_state_obj.my_id)
    if not my_snake or not my_snake["alive"]:
        return -1.0  # 死了直接给最低分

    # 1. 组装队友需要的原始字典格式 game_state
    raw_game_state = {
        'turn': game_state_obj.turn,
        'board': {
            'width': game_state_obj.board_width,
            'height': game_state_obj.board_height,
            'food': [{'x': f[0], 'y': f[1]} for f in game_state_obj.food],
            'hazards': [{'x': h[0], 'y': h[1]} for h in game_state_obj.hazards],
            'snakes': []
        },
        'you': {
            'id': my_snake['id'],
            'health': my_snake['health'],
            'body': [{'x': b[0], 'y': b[1]} for b in my_snake['body']]
        }
    }

    for _, s_data in game_state_obj.snakes.items():
        if s_data['alive']:
            raw_game_state['board']['snakes'].append({
                'id': s_data['id'],
                'health': s_data['health'],
                'body': [{'x': b[0], 'y': b[1]} for b in s_data['body']]
            })

    # 2. 组装队友需要的额外参数
    occupied = {(s['x'], s['y']) for snake in raw_game_state['board']
                ['snakes'] for s in snake['body']}
    hazard_set = {(h['x'], h['y']) for h in raw_game_state['board']['hazards']}
    opponent_heads = [{'head': s['body'][0], 'length': len(s['body'])}
                      for s in raw_game_state['board']['snakes'] if s['id'] != game_state_obj.my_id]

    # 3. 获取当前状态下我方所有合法的动作
    safe_moves = game_state_obj.get_safe_moves(game_state_obj.my_id)

    if not safe_moves:
        return -1.0  # 必死之局

    # 4. 调用队友的 move_score 计算每个动作的分数，取最高分代表当前局面的潜力
    best_raw_score = float('-inf')
    for move in safe_moves:
        try:
            # 直接调用队友的函数
            score = move_score(move, raw_game_state, occupied,
                               hazard_set, opponent_heads)
            if score > best_raw_score:
                best_raw_score = score
        except Exception as e:
            # 万一队友的代码抛出异常，忽略这个动作
            continue
    # 在heuristic_agent.py 的move_score 函数中，进入低血量陷阱或与大蛇头碰头的惩罚值都是 -100,所以设置-50可以确保只要发生了这些“灾难性”事件，分数就会被立刻截断到极低值
    if best_raw_score == float('-inf') or best_raw_score < -50:
        # 如果没有安全动作，或者队友代码判定为极其危险（如撞大蛇或踩陷阱）
        return 0.0001

    if best_raw_score == float('-inf'):
        return 0.0001

    # 5. 分数归一化 (Sigmoid)
    # 队友的分数可能很大(比如吃到惩罚-100)，MCTS 需要 0 到 1 之间的分数
    # 这里的除数 100 可以根据队友分数的实际幅度调整
    normalized_score = 1.0 / (1.0 + math.exp(-best_raw_score / 30.0))

    # 调试打印：每 50 轮打印一次，避免刷屏太快
    # if game_state_obj.turn % 10 == 0:
    #     print(
    #         f"[Evaluation] Turn: {game_state_obj.turn} | Raw: {best_raw_score:.2f} | Normalized: {normalized_score:.4f}")
    return normalized_score

# ─────────────────────────────────────────
# 1. 游戏状态模拟
# ─────────────────────────────────────────


class GameState:
    def __init__(self, game_state: dict, my_id: str):
        self.board_width = game_state["board"]["width"]
        self.board_height = game_state["board"]["height"]
        self.food = [tuple(f.values()) for f in game_state["board"]["food"]]
        self.hazards = [tuple(h.values())
                        for h in game_state["board"]["hazards"]]
        self.my_id = my_id
        self.turn = game_state["turn"]

        # 每条蛇: {id, body(list of (x,y)), health, alive}
        self.snakes = {}
        for s in game_state["board"]["snakes"]:
            self.snakes[s["id"]] = {
                "id":     s["id"],
                "body":   [(b["x"], b["y"]) for b in s["body"]],
                "health": s["health"],
                "alive":  True,
            }

    def get_safe_moves(self, snake_id):
        """返回某条蛇的合法方向"""
        snake = self.snakes.get(snake_id)
        if not snake or not snake["alive"]:
            return ["up"]  # 死蛇随便返回

        head = snake["body"][0]
        directions = {
            "up":    (head[0],     head[1] + 1),
            "down":  (head[0],     head[1] - 1),
            "left":  (head[0] - 1, head[1]),
            "right": (head[0] + 1, head[1]),
        }
        safe = []
        for move, (nx, ny) in directions.items():
            # 不出界
            if not (0 <= nx < self.board_width and 0 <= ny < self.board_height):
                continue
            # 不撞自己/别人的身体（尾巴可以走，因为会移动）
            occupied = False
            for s in self.snakes.values():
                if s["alive"]:
                    if (nx, ny) in s["body"][:-1]:
                        occupied = True
                        break
            if not occupied:
                safe.append(move)
        return safe if safe else ["up"]

    def step(self, moves: dict):
        """
        执行一步：moves = {snake_id: "up"/"down"/"left"/"right"}
        返回新的 GameState（deep copy）
        """
        new_state = copy.deepcopy(self)
        new_state.turn += 1

        MOVE_DELTA = {"up": (0, 1), "down": (
            0, -1), "left": (-1, 0), "right": (1, 0)}
        new_heads = {}

        # 1. 移动所有蛇
        for sid, snake in new_state.snakes.items():
            if not snake["alive"]:
                continue
            move = moves.get(sid, random.choice(
                ["up", "down", "left", "right"]))
            dx, dy = MOVE_DELTA[move]
            old_head = snake["body"][0]
            new_head = (old_head[0]+dx, old_head[1]+dy)
            snake["body"].insert(0, new_head)
            snake["health"] -= 1

            # --- 修正后的陷阱扣血逻辑 ---
            if new_state.turn >= 26:
                if new_head in new_state.hazards:  # 必须用 new_head！
                    # 计算当前处于哪个周期内 (每 150 回合为一个大循环)
                    cycle_turn = (new_state.turn - 26) % 150

                    if cycle_turn < 75:
                        # 前 75 步为堆叠期：第 1, 2, 3 层 (每 25 步加一层)
                        stack_level = (cycle_turn // 25) + 1
                    else:
                        # 后 75 步为满载期：保持第 4 层
                        stack_level = 4

                    # 扣除额外伤害 (第一层 14 点，随层数倍增)
                    snake["health"] -= (14 * stack_level)
            # -----------------------------

            new_heads[sid] = new_head

        # 2. 吃食物 / 扣血 / 判断出界
        for sid, snake in new_state.snakes.items():
            if not snake["alive"]:
                continue
            head = snake["body"][0]

            # 出界
            if not (0 <= head[0] < new_state.board_width and
                    0 <= head[1] < new_state.board_height):
                snake["alive"] = False
                continue

            # 吃食物
            if head in new_state.food:
                snake["health"] = 100
                new_state.food.remove(head)
            else:
                snake["body"].pop()  # 没吃食物，尾巴缩短

            # 饿死
            if snake["health"] <= 0:
                snake["alive"] = False

        # 3. 碰撞检测
        for sid, snake in new_state.snakes.items():
            if not snake["alive"]:
                continue
            head = snake["body"][0]

            # 撞身体
            for sid2, snake2 in new_state.snakes.items():
                if not snake2["alive"]:
                    continue
                body_to_check = snake2["body"][1:] if sid == sid2 else snake2["body"]
                if head in body_to_check:
                    snake["alive"] = False
                    break

            # 头碰头
            if snake["alive"]:
                for sid2, head2 in new_heads.items():
                    if sid2 != sid and head == head2:
                        # 短的死，等长都死
                        len1 = len(new_state.snakes[sid]["body"])
                        len2 = len(new_state.snakes[sid2]["body"])
                        if len1 <= len2:
                            snake["alive"] = False

        return new_state

    def is_terminal(self):
        my_snake = self.snakes.get(self.my_id)
        if not my_snake or not my_snake["alive"]:
            return True
        alive_count = sum(1 for s in self.snakes.values() if s["alive"])
        return alive_count <= 1 or self.turn >= 300

    def evaluate(self, heuristic_fn=None):
        """终态/中间态评分，调用队友的启发式函数"""
        my_snake = self.snakes.get(self.my_id)
        if not my_snake or not my_snake["alive"]:
            return -1.0  # 我死了，最差
        # 如果我是唯一的幸存者，赢得比赛
        alive = [s for s in self.snakes.values() if s["alive"]]
        if len(alive) == 1 and alive[0]["id"] == self.my_id:
            return 1.0   # 我赢了

        # ⚠️ 这里调用我们写的适配器
        if heuristic_fn:
            return heuristic_fn(self)

        # 默认简单启发：存活即正分
        return 0.1


# ─────────────────────────────────────────
# 2. MCTS 节点
# ─────────────────────────────────────────
class MCTSNode:
    def __init__(self, state: GameState, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move       # 我方到达此节点的动作
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried = state.get_safe_moves(state.my_id)  # 未展开的动作

    def is_fully_expanded(self):
        return len(self.untried) == 0

    def ucb1(self, c=1.41):
        if self.visits == 0:
            return float("inf")
        return self.value / self.visits + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, c=1.41):
        return max(self.children, key=lambda n: n.ucb1(c))


# ─────────────────────────────────────────
# 3. MCTS 主算法
# ─────────────────────────────────────────
class MCTS:
    def __init__(self, time_limit=0.8, c=1.41, heuristic_fn=None):
        self.time_limit = time_limit   # 留200ms余量
        self.c = c
        self.heuristic_fn = heuristic_fn

    def search(self, root_state: GameState) -> str:
        root = MCTSNode(root_state)
        end_time = time.time() + self.time_limit

        count = 0
        while time.time() < end_time:  # 有可能时间内一次没有跑
            node = self._select(root)
            if not node.state.is_terminal():
                node = self._expand(node)
            result = self._simulate(node.state)
            self._backpropagate(node, result)
            count += 1

        # print(f"MCTS iterations completed: {count}")
        # 👇 新增这块：把迭代次数追加写入本地文件
        with open("mcts_iters.txt", "a") as f:
            f.write(f"{count}\n")

        # 选访问次数最多的子节点
        if not root.children:  # 极端情况保底
            return random.choice(root_state.get_safe_moves(root_state.my_id))
        # 选访问次数最多的子节点（最稳健）
        best = max(root.children, key=lambda n: n.visits)
        return best.move

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = node.best_child(self.c)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        move = node.untried.pop()

        # 其他蛇随机走
        moves = {sid: random.choice(node.state.get_safe_moves(sid))
                 for sid in node.state.snakes
                 if node.state.snakes[sid]["alive"] and sid != node.state.my_id}
        moves[node.state.my_id] = move

        new_state = node.state.step(moves)
        child = MCTSNode(new_state, parent=node, move=move)
        node.children.append(child)
        return child

    def _simulate(self, state: GameState, max_depth=20) -> float:
        """Rollout：随机/启发式模拟到终态"""
        sim_state = copy.deepcopy(state)
        for _ in range(max_depth):
            if sim_state.is_terminal():
                break
            moves = {}
            for sid, snake in sim_state.snakes.items():
                if snake["alive"]:
                    moves[sid] = random.choice(sim_state.get_safe_moves(sid))
            sim_state = sim_state.step(moves)
        # 这里会触发 GameState.evaluate，进而调用 heuristic_adapter
        return sim_state.evaluate(self.heuristic_fn)

    def _backpropagate(self, node: MCTSNode, result: float):
        while node:
            node.visits += 1
            node.value += result
            node = node.parent


# ─────────────────────────────────────────
# 4. 接入 BattleSnake API
# ─────────────────────────────────────────
# 根据开关，决定是传入评估函数，还是传入 None (纯随机)
active_heuristic = heuristic_adapter if USE_HEURISTIC else None
mcts = MCTS(time_limit=0.8, c=EXPERIMENT_C, heuristic_fn=active_heuristic)


def info():
    return {
        "apiversion": "1",
        "author": "",
        "color": "#00CC00",
        "head": "default",
        "tail": "default",
    }


def start(game_state: dict):
    print("MCTS GAME START")


def end(game_state: dict):
    print("MCTS GAME OVER")


def move(game_state: dict) -> dict:
    my_id = game_state["you"]["id"]
    state = GameState(game_state, my_id)

    safe = state.get_safe_moves(my_id)
    if not safe:
        return {"move": "up"}
    if len(safe) == 1:
        return {"move": safe[0]}

    # 调用 MCTS 搜索最佳动作
    try:
        best_move = mcts.search(state)
        # print(f"TURN {game_state['turn']} | MCTS move: {best_move}")
        return {"move": best_move}
    except Exception as e:
        # print(f"MCTS Error: {e}")
        return {"move": random.choice(safe)}  # 出错保底


if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})
