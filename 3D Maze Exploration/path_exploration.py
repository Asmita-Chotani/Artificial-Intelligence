import math
import heapq

#         #       0       1        2       3        4       5       6        7        8        9        10       11      12       13        14      15       16       17       18
actionSteps = [[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],[1,0,1],[1,0,-1],[-1,0,1],[-1,0,-1],[0,1,1],[0,1,-1],[0,-1,1],[0,-1,-1]]


def cost_calculation(algo, current, point2):
    x = point2[0] - current[0]
    y = point2[1] - current[1]
    z = point2[2] - current[2]
    distance = abs(x) + abs(y) + abs(z)
    if algo == "BFS":
        if distance == 0:
            return 0
        else:
            return 1
    else:
        if distance == 1:    # 10- straight
            return 10
        elif distance == 0:
            return 0
        else:
            return 14        # 14- diagonal


def boundary_check(maze_size, neighbour):
    if (neighbour[0] <= maze_size[0]) and (neighbour[1] <= maze_size[1]) and (neighbour[2] <= maze_size[2]):
        if (neighbour[0] >= 0) and (neighbour[1] >= 0) and (neighbour[2] >= 0):
            return True
        else:
            return False
    else:
        return False


def get_neighbouring_point(action, current):
    currAction=actionSteps[action]
    return ((current[0] + currAction[0]),(current[1] + currAction[1]),(current[2] + currAction[2]))


def final_result(entrance_point, exit_point, is_already_visited):
    # algo: Specifies which algorithm is to be used for calculating teh distance
    # entrance_point : specifies the entry
    # exit_point : specifies the exit
    # is_already_visited : dictionary storing the cost to get to that node and the parent of that node

    list_of_steps = []    # list of the steps that has to be written in output file
    total_cost = 0        # total cost of the search

    grid = [exit_point]
    cost_to_individual_points = []
    i = exit_point
    total_cost = total_cost + is_already_visited[i][0]    # total cost of the search
    cost_to_individual_points.append(is_already_visited[i][0])    # the cost to reach the final destination

    while (1):
        if i == entrance_point:
            break
        i = is_already_visited[i][1]  # considering the parent of the point next
        cost = is_already_visited[i][0]  # storing the cost to get to that point
        cost_to_individual_points.append(cost)
        grid.append(i)

    grid.reverse()
    cost_to_individual_points.reverse()
    number_of_steps = len(grid)

    k = 0
    for point in grid:
        if k == 0:
            individual_cost = 0
        else:
            individual_cost = cost_to_individual_points[k] - cost_to_individual_points[k-1]
        list_of_steps.append(str(point[0]) + " " + str(point[1]) + " " + str(point[2]) + " " + str(individual_cost))
        k = k + 1

    file = open("output.txt", "a")
    file.write(str(total_cost) + "\n")
    file.write(str(number_of_steps))
    for step in list_of_steps:
        file.write("\n" + step)
    file.close()


def failed_result():
    file = open("output.txt", "a")
    file.write("FAIL")
    file.close()


def astar_algo(entrance_point, exit_point, maze_size, points_provided):
    flag = False
    queue = []  # list to store the current nodes
    is_already_visited = {}  # dictionary to store the cost and parent of every node  
    heapq.heappush(queue, (0, entrance_point, None, 0))  # Each heap element consisting of (cost, current point, parent)
    boundaryCheckSatisfied = True

    if (not boundary_check(maze_size, exit_point)) or (not boundary_check(maze_size, entrance_point)):
        boundaryCheckSatisfied = False

    while queue and (boundaryCheckSatisfied == True):
        heuristic, current, parent, individual_cost = heapq.heappop(queue)

        # Path complete
        if (current[0] == exit_point[0]) and (current[1] == exit_point[1]) and (current[2] == exit_point[2]):
            flag = True  # flag to print output in file
            is_already_visited[current] = (individual_cost, parent, heuristic)
            break

        # to check whether element has already been added and current cost is more than the previous one
        if (is_already_visited.get(current) is not None) and (is_already_visited[current][2] < individual_cost):
            continue

        actions_allowed = points_provided[current]

        is_already_visited[current] = (individual_cost, parent, heuristic)

        # for all action from current node, find the neighbour and the cost to get there
        for action in actions_allowed:
            neighbour = get_neighbouring_point(action, current)
            gn = cost_calculation("A*", current, neighbour) + individual_cost
            heuristic_value = gn + round(math.sqrt(((neighbour[0] - exit_point[0])**2) + ((neighbour[1] - exit_point[1])**2) + ((neighbour[2] - exit_point[2])**2)), 1)
            # check whether the neighbour is within the boundary of the maze
            traverse_cost = gn

            if not boundary_check(maze_size, neighbour):
                continue
            else:
                # check if the neighbour has been considered before and if it was, the cost was more than the current
                if (is_already_visited.get(neighbour) is None) or (is_already_visited[neighbour][2] > heuristic_value):
                    neighbour_node = (heuristic_value, neighbour, current, traverse_cost)
                    heapq.heappush(queue, neighbour_node)

    if flag:
        final_result(entrance_point, exit_point, is_already_visited)
    else:
        failed_result()


def ucs_algo(entrance_point, exit_point, maze_size, points_provided):
    flag = False
    queue = []  # list to store the current nodes
    is_already_visited = {}  # dictionary to store the cost and parent of every node
    heapq.heappush(queue, (0, entrance_point, None))  # Each heap element consisting of (cost, current point, parent)
    boundaryCheckSatisfied = True

    if (not boundary_check(maze_size, exit_point)) or (not boundary_check(maze_size, entrance_point)):
        boundaryCheckSatisfied = False

    while queue and (boundaryCheckSatisfied == True):
        individual_cost, current, parent = heapq.heappop(queue)

        # Path complete
        if (current[0] == exit_point[0]) and (current[1] == exit_point[1]) and (current[2] == exit_point[2]):
            flag = True  # flag to print output in file
            is_already_visited[current] = (individual_cost, parent)
            break

        # to check whether element has already been added and current cost is more than the previous one
        if (is_already_visited.get(current) is not None) and (is_already_visited[current][0] < individual_cost):
            continue

        actions_allowed = points_provided[current]

        is_already_visited[current] = (individual_cost, parent)

        # for all action from current node, find the neighbour and the cost to get there
        for action in actions_allowed:
            neighbour = get_neighbouring_point(action, current)
            traverse_cost = cost_calculation("UCS", current, neighbour) + individual_cost

            # check whether the neighbour is within the boundary of the maze
            if not boundary_check(maze_size, neighbour):
                continue
            else:
                # check if the neighbour has been considered before and if it was, the cost was more than the current
                if (is_already_visited.get(neighbour) is None) or (is_already_visited[neighbour][0] > traverse_cost):
                    neighbour_node = (traverse_cost, neighbour, current)
                    heapq.heappush(queue, neighbour_node)

    if flag:
        final_result(entrance_point, exit_point, is_already_visited)
    else:
        failed_result()


def bfs_algo(entrance_point, exit_point, maze_size, points_provided):  # working
    flag = False
    queue = []           # list to store the current nodes
    is_already_explored = {}  # dictionary to store the cost and parent of every node
    heapq.heappush(queue, (0, entrance_point, None))  # Each heap element consisting of (cost, current point, parent)
    boundaryCheckSatisfied = True

    if (not boundary_check(maze_size, exit_point)) or (not boundary_check(maze_size, entrance_point)):
        boundaryCheckSatisfied = False

    while queue and (boundaryCheckSatisfied == True):
        cost, current, parent = heapq.heappop(queue)

        # Path complete
        if (current[0] == exit_point[0]) and (current[1] == exit_point[1]) and (current[2] == exit_point[2]):
            flag = True  # flag to print output in file
            is_already_explored[current] = (cost, parent)
            break

        if is_already_explored.get(current) is not None:
            continue

        actions_allowed = points_provided[current]

        is_already_explored[current] = (cost, parent)

        # for all action from current node, find the neighbour and the cost to get there
        for action in actions_allowed:
            neighbour = get_neighbouring_point(action, current)
            if neighbour == entrance_point:
                traverse_cost = cost + 0
            else:
                traverse_cost = cost + 1

            if not boundary_check(maze_size, neighbour):
                # check whether the neighbour is within the boundary of the maze
                continue
            else:
                if is_already_explored.get(neighbour) is None:
                    # check if the neighbour has been considered before or not
                    neighbour_node = (traverse_cost, neighbour, current)
                    heapq.heappush(queue, neighbour_node)

    if flag:
        final_result(entrance_point, exit_point, is_already_explored)
    else:
        failed_result()


def input_details():
    with open('input.txt') as f:
        input = f.readlines()
        #length = len(input)
        points_provided = {}
        specific_algo = input[0].rstrip()  # line1 : input the algorithm

        maze_size = input[1].split()       # line2 : input the boundary size of the maze
        maze_size = tuple(int(i) for i in maze_size[:3])

        entrance_point = input[2].split()  # line3 : input the entry point coordinates
        entrance_point = tuple(int(i) for i in entrance_point[:3])

        exit_point = input[3].split()      # line4 : input the exit point coordinates
        exit_point = tuple(int(i) for i in exit_point[:3])

        no_of_points_allowed = int(input[4])  # line5 : input the number of points allowed

        for i in range(5, no_of_points_allowed+5):
            new_point_details = input[i].split()    # line i : input the point coordinates along with the action numbers
            new_point = tuple(int(i) for i in new_point_details[:3])  # Split the input and store the coordinate values into variable
            actions = [int(i) for i in new_point_details[3:]]         # Split the input and create a list of the actions allowed

            # enter the point and actions into a dictionary, where key is the point coordinates and actions is the value
            points_provided[new_point] = actions
        f.close()
    return specific_algo, entrance_point, exit_point, maze_size, points_provided


if __name__ == '__main__':
    specific_algo, entrance_point, exit_point, maze_size, points_provided = input_details()
    if specific_algo == "BFS":
        bfs_algo(entrance_point, exit_point, maze_size, points_provided)
    elif specific_algo == "UCS":
        ucs_algo(entrance_point, exit_point, maze_size, points_provided)
    elif specific_algo == "A*":
        astar_algo(entrance_point, exit_point, maze_size, points_provided)
    else:
        failed_result()
