import math
import random
import time
import copy

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
        alive = [s for s in self.snakes.values() if s["alive"]]
        if len(alive) == 1 and alive[0]["id"] == self.my_id:
            return 1.0   # 我赢了
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

        print(f"MCTS iterations completed: {count}")
        # 选访问次数最多的子节点
        if not root.children:  # 极端情况保底
            return random.choice(root_state.get_safe_moves(self.my_id))
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
        return sim_state.evaluate(self.heuristic_fn)

    def _backpropagate(self, node: MCTSNode, result: float):
        while node:
            node.visits += 1
            node.value += result
            node = node.parent


# ─────────────────────────────────────────
# 4. 接入 BattleSnake API
# ─────────────────────────────────────────
mcts = MCTS(time_limit=0.8)


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
        print(f"TURN {game_state['turn']} | MCTS move: {best_move}")
        return {"move": best_move}
    except Exception as e:
        print(f"MCTS Error: {e}")
        return {"move": random.choice(safe)}  # 出错保底


if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})
