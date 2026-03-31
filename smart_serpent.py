import math
import random
import time
import copy
from heuristic_agent import move_score


# Experiment Parameters
EXPERIMENT_C  = 0.3    # UCB exploration constant
USE_HEURISTIC = True   # True = heuristic state evaluation, False = simple evaluation

def distance(point1, point2):
    return abs(point1['x'] - point2['x']) + abs(point1['y'] - point2['y']) #manhattan distance

def next_pos(head, move):
    if move == 'up':
        return {'x': head['x'], 'y': head['y'] + 1}
    elif move == 'down':
        return {'x': head['x'], 'y': head['y'] - 1}
    elif move == 'right':
        return {'x': head['x'] + 1, 'y': head['y']}
    elif move == 'left':
        return {'x': head['x'] - 1, 'y': head['y']}

def potential_spaces(start, board_width, board_height, occupied, max_blocks=30):
    visited = set()
    curr = [(start['x'], start['y'])]
    count = 0

    while curr and count < max_blocks:
        x, y = curr.pop()
        if (x, y) in visited:
            continue
        visited.add((x,y))
        if (x, y) in occupied:
            continue
        if not (0 <= x < board_width and 0 <= y < board_height):
            continue
        count += 1

        neighbors = [
            (x, y + 1),
            (x, y - 1),
            (x - 1, y),
            (x + 1, y),
        ]
        for x, y in neighbors:
            if (x, y) not in visited:
                curr.append((x, y))

    return count

def move_score(move, game_state, occupied, hazard_set, opponent_heads): # Evaluate the score of a potential move based on various factors
    my_head = game_state['you']['body'][0]
    my_health = game_state['you']['health']
    next_position = next_pos(my_head, move)
    score = 0

    # Score based on food
    food = game_state['board']['food']
    if food:
        closest_food_distance = min(distance(next_position, f) for f in food )
        if my_health < 30:  # Prioritize food if health is low
            score += 15 / (closest_food_distance + 1)  # Closer food gets higher score
        elif my_health < 70:  # Moderate priority for food
            score += 5 / (closest_food_distance + 1)  
        else:
            score += 1 / (closest_food_distance + 1)  # Low priority for food when health is high

    # Score based on free spaces around the next position
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    score += 2 * potential_spaces(next_position, board_width, board_height, occupied)

    # Score based on center proximity
    center_x, center_y = (board_width - 1) / 2, (board_height-1) / 2
    score -= 0.5 * distance(next_position, {'x': center_x, 'y': center_y})

    # Score based on Hazard zones
    if (next_position['x'], next_position['y']) in hazard_set:
        if my_health < 25:
            score -= 100  # Heavily penalize moving into hazards when health is low
        elif my_health < 60:
            score -= 40  # Moderate penalty for hazards when health is moderate
        else:
            score -= 15

    # Score based on proximity to opponents' heads
    my_length = len(game_state['you']['body'])
    for opponent_head in opponent_heads:
        head = opponent_head['head']
        length = opponent_head['length']
        danger_zone = [
                {'x': head['x'], 'y': head['y'] + 1},
                {'x': head['x'], 'y': head['y'] - 1},
                {'x': head['x'] + 1, 'y': head['y']},
                {'x': head['x'] - 1, 'y': head['y']}
            ]  # Opponent can move into the square next to their head
        
        for danger in danger_zone:
            if danger['x'] == next_position['x'] and danger['y'] == next_position['y']:
                if length >= my_length:
                    score -= 100  # Heavily penalize moves that could lead to head-to-head with longer or equal snakes
                else:
                    score += 20  # Encourage moves that could lead to head-to-head with shorter snakes
                break  # No need to check other danger zones for this opponent

    return score

#  Helper: build move_score input format
def _build_raw_state(state: 'GameState', pov_id: str) -> tuple:

    snake = state.snakes[pov_id]

    raw_game_state = {
        'turn': state.turn,
        'board': {
            'width':   state.board_width,
            'height':  state.board_height,
            'food':    [{'x': f[0], 'y': f[1]} for f in state.food],
            'hazards': [{'x': h[0], 'y': h[1]} for h in state.hazards],
            'snakes':  []
        },
        'you': {
            'id':     pov_id,
            'health': snake['health'],
            'body':   [{'x': b[0], 'y': b[1]} for b in snake['body']]
        }
    }

    for s in state.snakes.values():
        if s['alive']:
            raw_game_state['board']['snakes'].append({
                'id':     s['id'],
                'health': s['health'],
                'body':   [{'x': b[0], 'y': b[1]} for b in s['body']]
            })

    occupied = {
        (b['x'], b['y'])
        for s in raw_game_state['board']['snakes']
        for b in s['body']
    }
    hazard_set = {
        (h['x'], h['y'])
        for h in raw_game_state['board']['hazards']
    }
    opponent_heads = [
        {'head': s['body'][0], 'length': len(s['body'])}
        for s in raw_game_state['board']['snakes']
        if s['id'] != pov_id
    ]

    return raw_game_state, occupied, hazard_set, opponent_heads


def heuristic_adapter(game_state_obj: 'GameState') -> float:

    my_snake = game_state_obj.snakes.get(game_state_obj.my_id)
    if not my_snake or not my_snake["alive"]:
        return -1.0

    safe_moves = game_state_obj.get_safe_moves(game_state_obj.my_id)
    if not safe_moves:
        return -1.0

    raw, occupied, hazard_set, opp_heads = _build_raw_state(
        game_state_obj, game_state_obj.my_id)

    best_raw = float('-inf')
    for m in safe_moves:
        try:
            sc = move_score(m, raw, occupied, hazard_set, opp_heads)
            if sc > best_raw:
                best_raw = sc
        except Exception:
            continue

    if best_raw == float('-inf') or best_raw < -50:
        return 0.0001

    return 1.0 / (1.0 + math.exp(-best_raw / 30.0))


def opponent_best_move(state: 'GameState', snake_id: str) -> str:

    safe_moves = state.get_safe_moves(snake_id)

    if not safe_moves:
        return "up"
    if len(safe_moves) == 1:
        return safe_moves[0]

    snake = state.snakes.get(snake_id)
    if not snake or not snake["alive"]:
        return random.choice(safe_moves)

    try:
        raw, occupied, hazard_set, opp_heads = _build_raw_state(
            state, snake_id)

        best_move  = safe_moves[0]
        best_score = float('-inf')

        for m in safe_moves:
            sc = move_score(m, raw, occupied, hazard_set, opp_heads)
            if sc > best_score:
                best_score = sc
                best_move  = m

        return best_move

    except Exception:
        return random.choice(safe_moves)


# 1. Game State Simulation
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

        MOVE_DELTA = {"up": (0,1), "down": (0,-1),
                      "left": (-1,0), "right": (1,0)}
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


# 2. MCTS Node
class MCTSNode:
    def __init__(self, state: GameState, parent=None, move=None):
        self.state    = state
        self.parent   = parent
        self.move     = move
        self.children = []
        self.visits   = 0
        self.value    = 0.0
        self.untried  = state.get_safe_moves(state.my_id)

    def is_fully_expanded(self):
        return len(self.untried) == 0

    def ucb1(self, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        return (self.value / self.visits
                + c * math.sqrt(math.log(self.parent.visits) / self.visits))

    def best_child(self, c: float) -> 'MCTSNode':
        return max(self.children, key=lambda n: n.ucb1(c))


# 3. MCTS with Opponent Modeling
class MCTS_OpponentModel:

    def __init__(self, time_limit=0.8, c=0.5, heuristic_fn=None):
        self.time_limit   = time_limit
        self.c            = c
        self.heuristic_fn = heuristic_fn  # 我方 rollout 评估函数

    def search(self, root_state: GameState) -> str:
        root     = MCTSNode(root_state)
        end_time = time.time() + self.time_limit

        count = 0
        while time.time() < end_time:
            node   = self._select(root)
            if not node.state.is_terminal():
                node = self._expand(node)
            result = self._simulate(node.state)
            self._backpropagate(node, result)
            count += 1

        with open("mcts_iters.txt", "a") as f:
            f.write(f"{count}\n")

        if not root.children:
            return random.choice(root_state.get_safe_moves(root_state.my_id))

        best = max(root.children, key=lambda n: n.visits)
        return best.move

    
    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = node.best_child(self.c)
        return node

    # expansion, opponent modeling applied here
    def _expand(self, node: MCTSNode) -> MCTSNode:
        move = node.untried.pop()

        moves = {}
        for sid, snake in node.state.snakes.items():
            if not snake["alive"]:
                continue
            if sid == node.state.my_id:
                # Our snake: use the action being expanded
                moves[sid] = move
            else:
                # Opponent snakes: predict best move using heuristic
                moves[sid] = opponent_best_move(node.state, sid)

        new_state = node.state.step(moves)
        child     = MCTSNode(new_state, parent=node, move=move)
        node.children.append(child)
        return child

    # Simulation
    def _simulate(self, state: GameState, max_depth=20) -> float:

        sim_state = copy.deepcopy(state)

        for _ in range(max_depth):
            if sim_state.is_terminal():
                break

            moves = {}
            for sid, snake in sim_state.snakes.items():
                if not snake["alive"]:
                    continue
                if sid == sim_state.my_id:
                    moves[sid] = random.choice(sim_state.get_safe_moves(sid))
                else:
                    moves[sid] = opponent_best_move(sim_state, sid)

            sim_state = sim_state.step(moves)

        return sim_state.evaluate(self.heuristic_fn)

    def _backpropagate(self, node: MCTSNode, result: float):
        current = node
        while current is not None:
            current.visits += 1
            current.value  += result
            current = current.parent


# BattleSnake API
active_heuristic = heuristic_adapter if USE_HEURISTIC else None
mcts_om = MCTS_OpponentModel(
    time_limit   = 0.8,
    c            = EXPERIMENT_C,
    heuristic_fn = active_heuristic
)


def info():
    return {
        "apiversion": "1",
        "author":     "",
        "color":      "#A855F7",  #purple
        "head":       "default",
        "tail":       "default",
    }


def start(game_state: dict):
    print("MCTS-OpponentModel GAME START")


def end(game_state: dict):
    print("MCTS-OpponentModel GAME OVER")


def move(game_state: dict) -> dict:
    my_id = game_state["you"]["id"]
    state = GameState(game_state, my_id)

    safe = state.get_safe_moves(my_id)
    if not safe:
        return {"move": "up"}
    if len(safe) == 1:
        return {"move": safe[0]}

    try:
        best_move = mcts_om.search(state)
        return {"move": best_move}
    except Exception as e:
        print(f"MCTS-OM Error: {e}")
        return {"move": random.choice(safe)}


if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})
