# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "",  # TODO: Your Battlesnake Username
        "color": "#50B946",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:

    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    # We've included code to prevent your Battlesnake from moving backwards
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"

    if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
        is_move_safe["left"] = False

    elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
        is_move_safe["right"] = False

    elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
        is_move_safe["down"] = False

    elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
        is_move_safe["up"] = False

    # TODO: Step 1 - Prevent your Battlesnake from moving out of bounds
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    if my_head['x'] == 0:  # Don't move left if we're already on the left edge
        is_move_safe['left'] = False
    elif my_head['x'] == board_width - 1:  # Don't move right if we're already on the right edge
        is_move_safe['right'] = False
    if my_head['y'] == 0:  # Don't move down if we're already on the bottom edge
        is_move_safe['down'] = False
    elif my_head['y'] == board_height - 1:  # Don't move up if we're already on the top edge
        is_move_safe['up'] = False

    # TODO: Step 2 - Prevent your Battlesnake from colliding with itself
    my_body = game_state['you']['body']
    for segment in my_body[1:]:  # Skip the head
        if segment['x'] == my_head['x'] and segment['y'] == my_head['y'] + 1:
            is_move_safe['up'] = False
        elif segment['x'] == my_head['x'] and segment['y'] == my_head['y'] - 1:
            is_move_safe['down'] = False
        elif segment['x'] == my_head['x'] + 1 and segment['y'] == my_head['y']:
            is_move_safe['right'] = False
        elif segment['x'] == my_head['x'] - 1 and segment['y'] == my_head['y']:
            is_move_safe['left'] = False

    # TODO: Step 3 - Prevent your Battlesnake from colliding with other Battlesnakes
    opponents = game_state['board']['snakes']
    opponent_heads = []

    for opponent in opponents:
        if opponent['id'] == game_state['you']['id']:
            continue  # Skip our own snake
        opponent_heads.append({'head': opponent['body'][0], 'length': len(opponent['body'])})
        for segment in opponent['body']:
            if segment['x'] == my_head['x'] and segment['y'] == my_head['y'] + 1:
                is_move_safe['up'] = False
            elif segment['x'] == my_head['x'] and segment['y'] == my_head['y'] - 1:
                is_move_safe['down'] = False
            elif segment['x'] == my_head['x'] + 1 and segment['y'] == my_head['y']:
                is_move_safe['right'] = False
            elif segment['x'] == my_head['x'] - 1 and segment['y'] == my_head['y']:
                is_move_safe['left'] = False

    ''' 
        opponent_head = opponent['body'][0]
        opponent_length = len(opponent['body'])
        my_length = len(my_body)
        if opponent_length >= my_length:
            danger_zone = [
                {'x': opponent_head['x'], 'y': opponent_head['y'] + 1},
                {'x': opponent_head['x'], 'y': opponent_head['y'] - 1},
                {'x': opponent_head['x'] + 1, 'y': opponent_head['y']},
                {'x': opponent_head['x'] - 1, 'y': opponent_head['y']}
            ]  # Opponent can move into the square next to their head
            for danger in danger_zone:
                if danger['x'] == my_head['x'] and danger['y'] == my_head['y'] + 1:
                    is_move_safe['up'] = False
                elif danger['x'] == my_head['x'] and danger['y'] == my_head['y'] - 1:
                    is_move_safe['down'] = False
                elif danger['x'] == my_head['x'] + 1 and danger['y'] == my_head['y']:
                    is_move_safe['right'] = False
                elif danger['x'] == my_head['x'] - 1 and danger['y'] == my_head['y']:
                    is_move_safe['left'] = False
    '''

    # Are there any safe moves left?
    safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            safe_moves.append(move)

    if len(safe_moves) == 0:
        print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}
    
    # Hazard zones
    hazard_set = set()
    for hazard in game_state['board']['hazards']:
        hazard_set.add((hazard['x'], hazard['y']))

    # TODO: Step 4 - Move towards food instead of random, to regain health and survive longer
    '''food = game_state['board']['food']
    if food:
        closest_food = min(food, key=lambda f: distance(my_head, f))
        if closest_food['x'] < my_head['x'] and is_move_safe['left']:
            next_move = 'left'
        elif closest_food['x'] > my_head['x'] and is_move_safe['right']:
            next_move = 'right'
        elif closest_food['y'] < my_head['y'] and is_move_safe['down']:
            next_move = 'down'
        elif closest_food['y'] > my_head['y'] and is_move_safe['up']:
            next_move = 'up'
    else:
        next_move = random.choice(safe_moves)'''

    occupied = {(segment['x'], segment['y']) for snake in game_state['board']['snakes'] for segment in snake['body']}
    best_move = []
    best_score = float('-inf')
    for move in safe_moves:
        score = move_score(move, game_state, occupied, hazard_set, opponent_heads)
        if score > best_score:
            best_score = score
            best_move = [move]
        elif score == best_score:
            best_move.append(move)

    next_move = random.choice(best_move)

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}

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


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
