import math
import random
import time
import copy
from heuristic_agent import move_score
# ─────────────────────────────────────────
#  hyperparameters
# ─────────────────────────────────────────
EXPERIMENT_C = 0.4       # goal: text UCB C (such as 0.5, 1.41, 2.0)
USE_HEURISTIC = True    # goal：True means heuristic Rollout，False means random Rollout


def heuristic_adapter(game_state_obj: 'GameState') -> float:
    """
    Convert the GameState to a dictionary format and call the teammate's `move_score` function.
    Since the teammate's function must be passed a ‘move’, we will use the highest score among all valid moves
    as the heuristic score for the current state.
    """
    my_snake = game_state_obj.snakes.get(game_state_obj.my_id)
    if not my_snake or not my_snake["alive"]:
        return -1.0  # if die, the lowest score

    # 1. transform the GameState to a dictionary format for the heuristic agent game_state
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

    # 2.adopt heuristis agent paremeters
    occupied = {(s['x'], s['y']) for snake in raw_game_state['board']
                ['snakes'] for s in snake['body']}
    hazard_set = {(h['x'], h['y']) for h in raw_game_state['board']['hazards']}
    opponent_heads = [{'head': s['body'][0], 'length': len(s['body'])}
                      for s in raw_game_state['board']['snakes'] if s['id'] != game_state_obj.my_id]

    # 3.  Get all valid actions available to our agent in the current state
    safe_moves = game_state_obj.get_safe_moves(game_state_obj.my_id)

    if not safe_moves:
        return -1.0  # fail

    # 4. Calculate the score for each move using the `move_score` function for heuristic agent, and use the highest score as an indication of the potential of the current position.
    best_raw_score = float('-inf')
    for move in safe_moves:
        try:
            score = move_score(move, raw_game_state, occupied,
                               hazard_set, opponent_heads)
            if score > best_raw_score:
                best_raw_score = score
        except Exception as e:
            # if error , ignore this action
            continue
    # In the `move_score` function of `heuristic_agent.py`, the penalty for falling into a low-health trap or colliding with the giant snake's head is -100, so setting the value to -50 ensures that the score is immediately clipped to an extremely low value whenever these “catastrophic” events occur.
    if best_raw_score == float('-inf') or best_raw_score < -50:
        return 0.0001  # dangerous

    if best_raw_score == float('-inf'):
        return 0.0001  # dangerous

    # 5. Sigmoid
    normalized_score = 1.0 / (1.0 + math.exp(-best_raw_score / 30.0))

    # if game_state_obj.turn % 10 == 0:
    #     print(
    #         f"[Evaluation] Turn: {game_state_obj.turn} | Raw: {best_raw_score:.2f} | Normalized: {normalized_score:.4f}")
    return normalized_score

# ─────────────────────────────────────────
# 1. Game state
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

        # each snake: {id, body(list of (x,y)), health, alive}
        self.snakes = {}
        for s in game_state["board"]["snakes"]:
            self.snakes[s["id"]] = {
                "id":     s["id"],
                "body":   [(b["x"], b["y"]) for b in s["body"]],
                "health": s["health"],
                "alive":  True,
            }

    def get_safe_moves(self, snake_id):
        """Return a valid direction for a given snake"""
        snake = self.snakes.get(snake_id)
        if not snake or not snake["alive"]:
            return ["up"]  # if die, return up

        head = snake["body"][0]
        directions = {
            "up":    (head[0],     head[1] + 1),
            "down":  (head[0],     head[1] - 1),
            "left":  (head[0] - 1, head[1]),
            "right": (head[0] + 1, head[1]),
        }
        safe = []
        for move, (nx, ny) in directions.items():
            # Stay in bounds
            if not (0 <= nx < self.board_width and 0 <= ny < self.board_height):
                continue
            # Do not collide with your own body or others' bodies (the tail is okay, since it moves)
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
        Step 1: moves = {snake_id: “up”/“down”/‘left’/“right”}
        Returns a new GameState (deep copy)
        """
        new_state = copy.deepcopy(self)
        new_state.turn += 1

        MOVE_DELTA = {"up": (0, 1), "down": (
            0, -1), "left": (-1, 0), "right": (1, 0)}
        new_heads = {}

        # 1. All snakes move at the same time
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

            if new_state.turn >= 26:
                if new_head in new_state.hazards:  # new_head
                    # Determine which cycle the game is currently in (each 150 rounds constitutes a major cycle)
                    cycle_turn = (new_state.turn - 26) % 150

                    if cycle_turn < 75:
                        # The first 75 steps constitute the stacking phase: layers 1, 2, and 3 (adding one layer every 25 steps)
                        stack_level = (cycle_turn // 25) + 1
                    else:
                        # The next 75 steps constitute the full-load phase: maintain the 4th layer
                        stack_level = 4

                    # Deducts additional damage (14 points at Tier 1, doubling with each tier)
                    snake["health"] -= (14 * stack_level)
            # -----------------------------

            new_heads[sid] = new_head

        # 2. Eat food / Lose health / Determine if out of bounds
        for sid, snake in new_state.snakes.items():
            if not snake["alive"]:
                continue
            head = snake["body"][0]

            # out of  boundaries
            if not (0 <= head[0] < new_state.board_width and
                    0 <= head[1] < new_state.board_height):
                snake["alive"] = False
                continue

            # eat food
            if head in new_state.food:
                snake["health"] = 100
                new_state.food.remove(head)
            else:
                snake["body"].pop()  # Didn't eat anything, tail shortened

            # starve to death
            if snake["health"] <= 0:
                snake["alive"] = False

        # 3. Collision Detection
        for sid, snake in new_state.snakes.items():
            if not snake["alive"]:
                continue
            head = snake["body"][0]

            # Bump into someone
            for sid2, snake2 in new_state.snakes.items():
                if not snake2["alive"]:
                    continue
                body_to_check = snake2["body"][1:] if sid == sid2 else snake2["body"]
                if head in body_to_check:
                    snake["alive"] = False
                    break

            # head to head
            if snake["alive"]:
                for sid2, head2 in new_heads.items():
                    if sid2 != sid and head == head2:
                        # The short one dies, and the ones of equal length all die
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
        """Final state/intermediate state scoring, call the heuristic function"""
        my_snake = self.snakes.get(self.my_id)
        if not my_snake or not my_snake["alive"]:
            return -1.0  # if we die, this is worst possible score
        # If I'm the only survivor, I win the game
        alive = [s for s in self.snakes.values() if s["alive"]]
        if len(alive) == 1 and alive[0]["id"] == self.my_id:
            return 1.0   # 我赢了

        # Call the adapter
        if heuristic_fn:
            return heuristic_fn(self)

        # Default random model: Survival is considered a positive outcome
        return 0.1


# ─────────────────────────────────────────
# 2. MCTS
# ─────────────────────────────────────────
class MCTSNode:
    def __init__(self, state: GameState, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move       # Our action upon reaching this state
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried = state.get_safe_moves(
            state.my_id)  # future actions to try

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
# 3. MCTS main
# ─────────────────────────────────────────
class MCTS:
    def __init__(self, time_limit=0.8, c=1.41, heuristic_fn=None):
        self.time_limit = time_limit   # 200ms left
        self.c = c
        self.heuristic_fn = heuristic_fn

    def search(self, root_state: GameState) -> str:
        root = MCTSNode(root_state)
        end_time = time.time() + self.time_limit

        count = 0
        while time.time() < end_time:
            node = self._select(root)
            if not node.state.is_terminal():
                node = self._expand(node)
            result = self._simulate(node.state)
            self._backpropagate(node, result)
            count += 1

        # print(f"MCTS iterations completed: {count}")
        with open("mcts_iters.txt", "a") as f:
            f.write(f"{count}\n")

        if not root.children:  # extreme situations
            return random.choice(root_state.get_safe_moves(root_state.my_id))
        # Select the child node with the highest number of visits
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

        # The other snakes move randomly
        moves = {sid: random.choice(node.state.get_safe_moves(sid))
                 for sid in node.state.snakes
                 if node.state.snakes[sid]["alive"] and sid != node.state.my_id}
        moves[node.state.my_id] = move

        new_state = node.state.step(moves)
        child = MCTSNode(new_state, parent=node, move=move)
        node.children.append(child)
        return child

    def _simulate(self, state: GameState, max_depth=20) -> float:
        sim_state = copy.deepcopy(state)
        for _ in range(max_depth):
            if sim_state.is_terminal():
                break
            moves = {}
            for sid, snake in sim_state.snakes.items():
                if snake["alive"]:
                    moves[sid] = random.choice(sim_state.get_safe_moves(sid))
            sim_state = sim_state.step(moves)
        # This will trigger `GameState.evaluate`, which in turn calls `heuristic_adapter`.
        return sim_state.evaluate(self.heuristic_fn)

    def _backpropagate(self, node: MCTSNode, result: float):
        while node:
            node.visits += 1
            node.value += result
            node = node.parent


# ─────────────────────────────────────────
# 4. BattleSnake
# ─────────────────────────────────────────
# heuristic or random
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

    # Call MCTS to find the best action
    try:
        best_move = mcts.search(state)
        # print(f"TURN {game_state['turn']} | MCTS move: {best_move}")
        return {"move": best_move}
    except Exception as e:
        # print(f"MCTS Error: {e}")
        return {"move": random.choice(safe)}


if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})
