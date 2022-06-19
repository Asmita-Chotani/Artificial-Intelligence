import time
from copy import deepcopy


BOARDSIZE = 5


def all_neighbours(row, column):
    neighbours = []
    if row > 0:
        neighbours.append((row - 1, column))
    if row < BOARDSIZE - 1:
        neighbours.append((row + 1, column))
    if column > 0:
        neighbours.append((row, column - 1))
    if column < BOARDSIZE - 1:
        neighbours.append((row, column + 1))

    return neighbours


def all_partner_neighbours(board_details, row, column, player_colour):
    partner_positions = []
    adjacent_points = all_neighbours(row=row, column=column)
    for point in adjacent_points:
        if board_details[point[0]][point[1]] == player_colour:
            partner_positions.append(point)
    return partner_positions


# function that returns partner cluster of a point
# returns a list of partner cluster given a certain point on the board
def find_cluster_of_partners(board_details, row, column, player_colour):
    stack_of_points = [(row, column)]
    isVisited = {}
    isVisited[(row, column)] = True
    partners = []
    while stack_of_points:
        curr_point = stack_of_points.pop(0)
        partners.append(curr_point)
        neighbours_list = all_partner_neighbours(board_details= board_details, row= curr_point[0], column= curr_point[1], player_colour= player_colour)
        for neighbour in neighbours_list:
            if neighbour not in stack_of_points and neighbour not in partners:
                stack_of_points.append(neighbour)
                isVisited[neighbour] = True
    return partners


def count_liberties(x, y, current_board, player_colour):
    liberties = set()
    count = 0
    cluster = find_cluster_of_partners(board_details=current_board, row=x, column=y, player_colour=player_colour)
    isVisited = {}

    for point in cluster:
        neighbors = all_neighbours(row=point[0], column=point[1])

        for neighbor in neighbors:
            if neighbor not in isVisited:
                isVisited[neighbor] = 1

            if current_board[neighbor[0]][neighbor[1]] == 0:
                liberties.add(neighbor)
                count += 1

    return liberties, count


def find_dead_stones(board, player_colour):
    dead_stones = set()
    for i in range(BOARDSIZE):
        for j in range(BOARDSIZE):
            if board[i][j] == player_colour:

                # get the group, since we need it to add them as dead stones
                group = find_cluster_of_partners(row=i, column=j, board_details=board, player_colour=player_colour)

                # obtain liberties
                liberties, number_of_liberties = count_liberties(x=i, y=j, current_board=board,
                                                                 player_colour=player_colour)

                # if list of liberties is empty, then add all stones in that group as dead stones
                if number_of_liberties == 0:
                    for stone in group:
                        dead_stones.add(stone)

    return dead_stones


def remove_stones(board, locations):
    for stone in locations:
        board[stone[0]][stone[1]] = 0
    return board


def update_board(board, colour):
    # find dead stones
    dead_stones = find_dead_stones(board=board, player_colour=colour)
    # remove the dead stones
    temporary_board_updated = remove_stones(board=board, locations=dead_stones)

    return temporary_board_updated, len(dead_stones)


def execute_move(next_x, next_y, board, player_colour):
    temporary_board = deepcopy(board)

    # set the next move position as player's colour
    temporary_board[next_x][next_y] = player_colour

    opponent_colour = 3 - player_colour

    # find and remove opponent's dead stones
    updated_board1, opponent_dead_stones = update_board(board=temporary_board, colour=opponent_colour)

    # find and remove player's dead stones
    updated_board, player_dead_stones = update_board(board=updated_board1, colour=player_colour)

    return updated_board, player_dead_stones, opponent_dead_stones


def get_possible_moves(player_colour, previous_board, current_board):
    all_feasible_moves = set()

    # get all possible moves
    for i in range(BOARDSIZE):
        for j in range(BOARDSIZE):
            if current_board[i][j] != 0:
                current_position_liberties, number_of_liberties = count_liberties(x=i, y=j,
                                                                                  current_board=current_board,
                                                                                  player_colour=player_colour)
                all_feasible_moves = all_feasible_moves.union(current_position_liberties)
    valid_moves_scores = []

    for move in all_feasible_moves:
        output_board, players_stones_removed, opponents_stones_removed = execute_move(next_x=move[0],
                                                                                      next_y=move[1],
                                                                                      board=current_board,
                                                                                      player_colour=player_colour)

        partial_score = opponents_stones_removed - players_stones_removed

        # suicide and ko
        if output_board != current_board and output_board != previous_board:
            valid_moves_scores.append((move, partial_score))

    # sort valid moves
    valid_moves_scores = sorted(valid_moves_scores, key=lambda k: k[1], reverse=True)
    feasible_moves = []
    for score in valid_moves_scores:
        action = score[0]
        feasible_moves.append(action)

    if len(feasible_moves) != 0:
        return feasible_moves
    else:
        return None


def get_heuristic_score(board, player_colour):
    black_stones = 0
    white_stones = 0
    black_stones_in_danger = 0
    white_stones_in_danger = 0
    black_edge_count = 0
    white_edge_count = 0

    for i in range(BOARDSIZE):
        for j in range(BOARDSIZE):

            # black stone
            if board[i][j] == 1:
                black_stones = black_stones + 1
                if i == 0 or i == 4 or j == 0 or j == 4:
                    black_edge_count = 0
                
            #  white stone
            elif board[i][j] == 2:
                white_stones = white_stones + 1
                if i == 0 or i == 4 or j == 0 or j == 4:
                    white_edge_count = 0

            # number of liberties is 1 or less means stone is in danger of being captured
            sets_of_black_liberties, black_liberties = count_liberties(x=i, y=j, current_board=board, player_colour=1) 
            if black_liberties <= 1:
                black_stones_in_danger = black_stones_in_danger + 1

            sets_of_white_liberties, white_liberties = count_liberties(x=i, y=j, current_board=board, player_colour=2)
            if white_liberties <= 1:
                white_stones_in_danger = white_stones_in_danger + 1
           
    white_stones = white_stones + 6

    # player is white
    if player_colour == 2:
        heuristic_value = (10 * white_stones) - (10 * black_stones) + (2 * black_stones_in_danger) - (
               1.5 * white_stones_in_danger) - white_edge_count 

    # player is black
    elif player_colour == 1:
        heuristic_value = (10 * black_stones) - (10 * white_stones) + (2 * white_stones_in_danger) - (
               1.5 * black_stones_in_danger) - black_edge_count 

    return heuristic_value


def piece_count(board, colour):
    opponent_colour = 3 - colour
    opponent_stones = 0
    player_stones = 0
    for i in range(BOARDSIZE):
        for j in range(BOARDSIZE):
            if board[i][j] == opponent_colour:
                opponent_stones += 1
            if board[i][j] == colour:
                player_stones += 1
    return opponent_stones, player_stones


def max_function(curr_board, prev_board, my_colour, depth, alpha, beta):
    opponent_colour = 3 - my_colour
    if depth == 0:
        value = get_heuristic_score(curr_board, my_colour)
        return value, []

    opponent_stones, curr_player_stones = piece_count(board=curr_board, colour=my_colour)

    if opponent_stones == 0 and curr_player_stones == 0:
        return 100, [(2, 2)]
    if opponent_stones == 1 and curr_player_stones == 0:
        if curr_board[2][2] == opponent_colour:
            return 100, [(2, 1)]
        else:
            return 100, [(2, 2)]

    maximum = float('-inf')
    max_action = []

    for action in get_possible_moves(my_colour, prev_board, curr_board):
        temp = deepcopy(curr_board)
        next_board, curr_player_dead_stones, opponent_dead_stones = execute_move(next_x=action[0], next_y=action[1], board=temp, player_colour=my_colour)
        score, move = min_function(curr_board=next_board, prev_board=curr_board, my_colour=opponent_colour, depth=depth - 1, alpha=alpha, beta= beta)
        score += (opponent_dead_stones * 5) - (curr_player_dead_stones * 8.5)

        if score > maximum:
            maximum = score
            max_action = [action] + move
        # pruning
        if maximum >= beta: 
            return maximum, max_action
        if maximum > alpha:
            alpha = maximum

    return maximum, max_action


def min_function(curr_board, prev_board, my_colour, depth, alpha, beta):
    opponent_colour = 3 - my_colour
    if depth == 0:
        value = get_heuristic_score(curr_board, my_colour)
        return value, []
    minimum = float('inf')

    opponent_stones, curr_player_stones = piece_count(board=curr_board, colour=my_colour)

    if opponent_stones == 0 and curr_player_stones == 0:
        return 100, [(2, 2)]
    if opponent_stones == 1 and curr_player_stones == 0:
        if curr_board[2][2] == opponent_colour:
            return 100, [(2, 1)]
        else:
            return 100, [(2, 2)]

    min_action = []

    for action in get_possible_moves(my_colour, prev_board, curr_board):
        temp = deepcopy(curr_board)
        next_board, curr_player_dead_stones, opponent_dead_stones = execute_move(next_x=action[0], next_y=action[1], board=temp, player_colour=my_colour)
        score, move = max_function(curr_board=next_board, prev_board=curr_board, my_colour=opponent_colour, depth=depth - 1, alpha=alpha, beta=beta)

        score += (opponent_dead_stones * 5) - (curr_player_dead_stones * 8.5)

        if score < minimum:
            minimum = score
            min_action = [action] + move
        # pruning
        if minimum <= alpha: 
            return minimum, min_action
        if minimum < beta:
            alpha = minimum

    return minimum, min_action


def minmax(curr_board, prev_board, my_colour, depth):
    alpha = float("-inf")
    beta = float("inf")
    score, actions = max_function(curr_board, prev_board, my_colour, depth, alpha, beta)
    if len(actions) > 0:
        return actions[0]
    else:
        return "PASS"



start = time.time()

input_processed = []
previous_board = []
current_board = []

with open('input.txt', 'r') as fin:
    input = fin.readlines()
    for line in input:
        input_processed.append(line.strip())  # a list
    # assign colour of player
    player_colour = int(input_processed[0])
    # assign previous board, i.e board after last move
    for line in input_processed[1:BOARDSIZE + 1]:
        x = [int(val) for val in line]
        previous_board.append(x)
    # assign current board, i.e board after opponent's last move
    for line in input_processed[BOARDSIZE + 1:2 * BOARDSIZE + 1]:
        x = [int(val) for val in line]
        current_board.append(x)

depth = 2
action = minmax(curr_board=current_board, prev_board=previous_board, my_colour=player_colour, depth=depth)
f = open("output.txt", "w")
if action != "PASS":
    f.write(str(action[0]) + ',' + str(action[1]))
else:
    f.write("PASS")
f.close()
end = time.time()
print('total time of evaluation: ', end - start)
