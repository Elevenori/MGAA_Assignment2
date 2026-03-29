"""
MCTS with RAVE + poolRave improvement
======================================
忠实实现论文：
    Rimmel, Teytaud & Teytaud (2010)
    "Biasing Monte-Carlo Simulations through RAVE Values"

Algorithm 1 (Right) — RMCTS(s):

选择阶段：
    s' ← argmax_{j∈C_s'} [ x̄_j + α·x̄^RAVE_{s',j} + √(2ln(n_s')/n_j) ]

模拟阶段 (poolRave)：
    s'' ← 树中最后一个访问次数 ≥ MIN_SIMS 的节点
    while s' 不是终态:
        if Random < p:
            s' ← 从 s'' 中 RAVE 值最好的 pool_size 个动作里随机选一个
        else:
            s' ← mc(s')   # 随机走

反向传播：
    同时更新普通统计 (n_s, x̄_s) 和 RAVE 统计 (n^RAVE, x̄^RAVE)

可调超参数（论文实验范围）：
    p         : 使用 RAVE 动作的概率    论文测试: 1/4, 1/2, 3/4, 1
    pool_size : 从前 k 个 RAVE 动作选  论文测试: 5, 10, 20, 30, 60
"""

import math
import random
import time
import copy
from heuristic_agent import move_score

# ─────────────────────────────────────────
#  实验调参区（论文参数）
# ─────────────────────────────────────────
EXPERIMENT_C    = 0.5   # UCB 探索系数（沿用 vanilla 最优值）
EXPERIMENT_P    = 0.25   # poolRave 概率：以此概率使用 RAVE 动作（论文最优 1/2）
EXPERIMENT_POOL = 10    # pool size：从前 k 个 RAVE 动作中随机选（论文最优 10）
MIN_SIMS        = 50    # 论文固定值：节点至少有多少次模拟才能提供 RAVE 参考
USE_HEURISTIC   = True  # True = 启发式评估终态，False = 简单评估


# ─────────────────────────────────────────
#  启发式适配器（我方终态评估）
# ─────────────────────────────────────────
def heuristic_adapter(game_state_obj: 'GameState') -> float:
    """调用队友 move_score，Sigmoid 归一化到 (0,1)"""
    my_snake = game_state_obj.snakes.get(game_state_obj.my_id)
    if not my_snake or not my_snake["alive"]:
        return -1.0

    safe_moves = game_state_obj.get_safe_moves(game_state_obj.my_id)
    if not safe_moves:
        return -1.0

    raw_game_state = {
        'turn': game_state_obj.turn,
        'board': {
            'width':   game_state_obj.board_width,
            'height':  game_state_obj.board_height,
            'food':    [{'x': f[0], 'y': f[1]} for f in game_state_obj.food],
            'hazards': [{'x': h[0], 'y': h[1]} for h in game_state_obj.hazards],
            'snakes':  []
        },
        'you': {
            'id':     my_snake['id'],
            'health': my_snake['health'],
            'body':   [{'x': b[0], 'y': b[1]} for b in my_snake['body']]
        }
    }
    for s in game_state_obj.snakes.values():
        if s['alive']:
            raw_game_state['board']['snakes'].append({
                'id':     s['id'],
                'health': s['health'],
                'body':   [{'x': b[0], 'y': b[1]} for b in s['body']]
            })

    occupied = {(b['x'], b['y'])
                for sn in raw_game_state['board']['snakes']
                for b in sn['body']}
    hazard_set = {(h['x'], h['y'])
                  for h in raw_game_state['board']['hazards']}
    opponent_heads = [
        {'head': s['body'][0], 'length': len(s['body'])}
        for s in raw_game_state['board']['snakes']
        if s['id'] != game_state_obj.my_id
    ]

    best_raw = float('-inf')
    for m in safe_moves:
        try:
            sc = move_score(m, raw_game_state, occupied, hazard_set, opponent_heads)
            if sc > best_raw:
                best_raw = sc
        except Exception:
            continue

    if best_raw == float('-inf') or best_raw < -50:
        return 0.0001
    return 1.0 / (1.0 + math.exp(-best_raw / 30.0))


# ─────────────────────────────────────────
# 1. 游戏状态模拟
# ─────────────────────────────────────────
class GameState:
    def __init__(self, game_state: dict, my_id: str):
        self.board_width  = game_state["board"]["width"]
        self.board_height = game_state["board"]["height"]
        self.food         = [tuple(f.values()) for f in game_state["board"]["food"]]
        self.hazards      = [tuple(h.values()) for h in game_state["board"]["hazards"]]
        self.my_id        = my_id
        self.turn         = game_state["turn"]

        self.snakes = {}
        for s in game_state["board"]["snakes"]:
            self.snakes[s["id"]] = {
                "id":     s["id"],
                "body":   [(b["x"], b["y"]) for b in s["body"]],
                "health": s["health"],
                "alive":  True,
            }

    def get_safe_moves(self, snake_id):
        snake = self.snakes.get(snake_id)
        if not snake or not snake["alive"]:
            return ["up"]
        head = snake["body"][0]
        directions = {
            "up":    (head[0],     head[1] + 1),
            "down":  (head[0],     head[1] - 1),
            "left":  (head[0] - 1, head[1]),
            "right": (head[0] + 1, head[1]),
        }
        safe = []
        for move, (nx, ny) in directions.items():
            if not (0 <= nx < self.board_width and 0 <= ny < self.board_height):
                continue
            occupied = False
            for s in self.snakes.values():
                if s["alive"] and (nx, ny) in s["body"][:-1]:
                    occupied = True
                    break
            if not occupied:
                safe.append(move)
        return safe if safe else ["up"]

    def step(self, moves: dict):
        new_state = copy.deepcopy(self)
        new_state.turn += 1
        MOVE_DELTA = {"up": (0,1), "down": (0,-1), "left": (-1,0), "right": (1,0)}
        new_heads = {}

        for sid, snake in new_state.snakes.items():
            if not snake["alive"]:
                continue
            move = moves.get(sid, random.choice(["up","down","left","right"]))
            dx, dy = MOVE_DELTA[move]
            old_head = snake["body"][0]
            new_head = (old_head[0]+dx, old_head[1]+dy)
            snake["body"].insert(0, new_head)
            snake["health"] -= 1
            if new_state.turn >= 26 and new_head in new_state.hazards:
                cycle_turn  = (new_state.turn - 26) % 150
                stack_level = (cycle_turn // 25) + 1 if cycle_turn < 75 else 4
                snake["health"] -= 14 * stack_level
            new_heads[sid] = new_head

        for sid, snake in new_state.snakes.items():
            if not snake["alive"]:
                continue
            head = snake["body"][0]
            if not (0 <= head[0] < new_state.board_width and
                    0 <= head[1] < new_state.board_height):
                snake["alive"] = False
                continue
            if head in new_state.food:
                snake["health"] = 100
                new_state.food.remove(head)
            else:
                snake["body"].pop()
            if snake["health"] <= 0:
                snake["alive"] = False

        for sid, snake in new_state.snakes.items():
            if not snake["alive"]:
                continue
            head = snake["body"][0]
            for sid2, snake2 in new_state.snakes.items():
                if not snake2["alive"]:
                    continue
                body_to_check = snake2["body"][1:] if sid == sid2 else snake2["body"]
                if head in body_to_check:
                    snake["alive"] = False
                    break
            if snake["alive"]:
                for sid2, head2 in new_heads.items():
                    if sid2 != sid and head == head2:
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
        my_snake = self.snakes.get(self.my_id)
        if not my_snake or not my_snake["alive"]:
            return -1.0
        alive = [s for s in self.snakes.values() if s["alive"]]
        if len(alive) == 1 and alive[0]["id"] == self.my_id:
            return 1.0
        if heuristic_fn:
            return heuristic_fn(self)
        return 0.1


# ─────────────────────────────────────────
# 2. MCTS 节点（含 RAVE 统计）
# ─────────────────────────────────────────
class MCTSNode:
    def __init__(self, state: GameState, parent=None, move=None):
        self.state    = state
        self.parent   = parent
        self.move     = move
        self.children = []
        self.visits   = 0
        self.value    = 0.0
        self.untried  = state.get_safe_moves(state.my_id)

        # 论文符号：n^RAVE_{s,a} 和 x̄^RAVE_{s,a}
        self.rave_total  = {}   # 累计得分
        self.rave_visits = {}   # 访问次数

    def is_fully_expanded(self):
        return len(self.untried) == 0

    def rave_avg(self, move: str) -> float:
        """x̄^RAVE_{s, move}"""
        n = self.rave_visits.get(move, 0)
        if n == 0:
            return 0.0
        return self.rave_total.get(move, 0.0) / n

    def alpha(self) -> float:
        """
        α 随访问次数趋向 0（论文描述，具体公式参考 Gelly & Silver 2007）
        k_eq=300：访问约 100 次时 α≈0.71，访问 1000 次时 α≈0.30
        """
        k_eq = 300
        return math.sqrt(k_eq / (3.0 * max(self.visits, 1) + k_eq))

    def ucb_rave(self, c: float) -> float:
        """
        论文选择公式（Algorithm 1 Right, Descent）：
        x̄_j + α·x̄^RAVE_{s,j} + √(2ln(n_s) / n_j)
        """
        if self.visits == 0:
            return float("inf")

        exploit = self.value / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)

        rave = 0.0
        if self.move is not None and self.parent is not None:
            rave = self.parent.alpha() * self.parent.rave_avg(self.move)

        return exploit + rave + explore

    def best_child_rave(self, c: float) -> 'MCTSNode':
        return max(self.children, key=lambda n: n.ucb_rave(c))

    def top_k_rave_moves(self, k: int, safe_moves: list) -> list:
        """
        论文 poolRave：返回合法动作中 RAVE 值最好的前 k 个。
        论文原文："one of the k moves with best RAVE value in s''"
        """
        scored = sorted(safe_moves, key=lambda m: self.rave_avg(m), reverse=True)
        return scored[:k] if len(scored) >= k else scored


# ─────────────────────────────────────────
# 3. MCTS-poolRave 主算法
# ─────────────────────────────────────────
class MCTS_PoolRave:
    """
    忠实实现论文 Algorithm 1 (Right) —— RMCTS with poolRave

    超参数（论文实验范围）：
        p         : 1/4, 1/2, 3/4, 1     论文最优 p=1/2
        pool_size : 5, 10, 20, 30, 60    论文最优 pool_size=10
    """

    def __init__(self, time_limit=0.8, c=0.5,
                 p=0.5, pool_size=10, heuristic_fn=None):
        self.time_limit   = time_limit
        self.c            = c
        self.p            = p
        self.pool_size    = pool_size
        self.heuristic_fn = heuristic_fn

    def search(self, root_state: GameState) -> str:
        root     = MCTSNode(root_state)
        end_time = time.time() + self.time_limit

        count = 0
        while time.time() < end_time:
            node, path = self._select(root)

            if not node.state.is_terminal():
                node = self._expand(node)
                path.append(node)

            # 论文：找路径上最后一个访问次数 ≥ MIN_SIMS 的节点
            rave_ref = self._find_rave_reference(path)

            result, sim_moves = self._simulate(node.state, rave_ref)
            self._backpropagate(node, result, sim_moves)
            count += 1

        with open("mcts_iters.txt", "a") as f:
            f.write(f"{count}\n")

        if not root.children:
            return random.choice(root_state.get_safe_moves(root_state.my_id))

        best = max(root.children, key=lambda n: n.visits)
        return best.move

    # ── ① 选择 ────────────────────────────────────────────────────────────────
    def _select(self, node: MCTSNode):
        path = [node]
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node, path
            node = node.best_child_rave(self.c)
            path.append(node)
        return node, path

    # ── ② 扩展 ────────────────────────────────────────────────────────────────
    def _expand(self, node: MCTSNode) -> MCTSNode:
        move = node.untried.pop()
        moves = {
            sid: random.choice(node.state.get_safe_moves(sid))
            for sid, snake in node.state.snakes.items()
            if snake["alive"] and sid != node.state.my_id
        }
        moves[node.state.my_id] = move
        new_state = node.state.step(moves)
        child     = MCTSNode(new_state, parent=node, move=move)
        node.children.append(child)
        return child

    # ── 找 RAVE 参考节点 ───────────────────────────────────────────────────────
    def _find_rave_reference(self, path: list) -> MCTSNode:
        """
        论文原文：
        s'' ← last visited node in the tree with at least 50 simulations
        """
        for node in reversed(path):
            if node.visits >= MIN_SIMS:
                return node
        return path[0]

    # ── ③ 模拟（poolRave 核心）──────────────────────────────────────────────
    def _simulate(self, state: GameState,
                  rave_ref: MCTSNode, max_depth=20):
        """
        论文 poolRave 模拟（Algorithm 1 Right, Evaluation）：

        while s' 不是终态:
            if Random < p:
                s' ← 从 rave_ref 的 top-k RAVE 动作里随机选一个
            else:
                s' ← mc(s')
        """
        sim_state     = copy.deepcopy(state)
        my_moves_used = []

        for _ in range(max_depth):
            if sim_state.is_terminal():
                break

            moves = {}
            for sid, snake in sim_state.snakes.items():
                if not snake["alive"]:
                    continue

                safe = sim_state.get_safe_moves(sid)

                if sid == sim_state.my_id:
                    if random.random() < self.p:
                        # poolRave：从 top-k RAVE 动作中随机均匀选一个
                        top_k  = rave_ref.top_k_rave_moves(self.pool_size, safe)
                        chosen = random.choice(top_k)
                    else:
                        # 正常随机 mc(s')
                        chosen = random.choice(safe)

                    my_moves_used.append(chosen)
                    moves[sid] = chosen
                else:
                    moves[sid] = random.choice(safe)

            sim_state = sim_state.step(moves)

        result = sim_state.evaluate(self.heuristic_fn)
        return result, my_moves_used

    # ── ④ 反向传播 ─────────────────────────────────────────────────────────────
    def _backpropagate(self, node: MCTSNode,
                       result: float, sim_moves: list):
        """
        论文 Growth 阶段（Algorithm 1 Right）：
        同时更新普通统计和 RAVE 统计
        """
        current = node
        while current is not None:
            current.visits += 1
            current.value  += result
            for m in sim_moves:
                current.rave_total[m]  = current.rave_total.get(m, 0.0)  + result
                current.rave_visits[m] = current.rave_visits.get(m, 0)   + 1
            current = current.parent


# ─────────────────────────────────────────
# 4. 接入 BattleSnake API
# ─────────────────────────────────────────
active_heuristic = heuristic_adapter if USE_HEURISTIC else None
mcts_poolrave = MCTS_PoolRave(
    time_limit   = 0.8,
    c            = EXPERIMENT_C,
    p            = EXPERIMENT_P,
    pool_size    = EXPERIMENT_POOL,
    heuristic_fn = active_heuristic
)


def info():
    return {
        "apiversion": "1",
        "author":     "",
        "color":      "#FF6B6B",
        "head":       "default",
        "tail":       "default",
    }


def start(game_state: dict):
    print("MCTS-poolRave GAME START")


def end(game_state: dict):
    print("MCTS-poolRave GAME OVER")


def move(game_state: dict) -> dict:
    my_id = game_state["you"]["id"]
    state = GameState(game_state, my_id)

    safe = state.get_safe_moves(my_id)
    if not safe:
        return {"move": "up"}
    if len(safe) == 1:
        return {"move": safe[0]}

    try:
        best_move = mcts_poolrave.search(state)
        return {"move": best_move}
    except Exception as e:
        print(f"MCTS-poolRave Error: {e}")
        return {"move": random.choice(safe)}


if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})
