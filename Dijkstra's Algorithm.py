import numpy as np
import math

num_x_cells = 14
num_y_cells = 11

map_size_x = 80
map_size_y = 80

big_number = 255

prev_holder = np.zeros(num_x_cells*num_y_cells)

world_map = np.zeros((num_y_cells, num_x_cells))
goal_changed = True
goal_i = 3
goal_j = 3
source_i = 0
source_j = 5
index = 0
prev = None
path = None

#############################
# Dijkstra Helper Functions #
#############################

def is_not_empty(arr, length):
    for i in range(length):
        if (arr[i] >= 0):
            return True
    return False

def get_min_index(arr, length):
    min_idx = 0
    for i in range(length):
        if ((arr[min_idx] < 0) or ((arr[i] < arr[min_idx]) and (arr[i] >= 0))):
            min_idx = i
    if (arr[min_idx] == -1):
        return -1
    return min_idx

##################################
# Coordinate Transform Functions #
##################################

def vertex_index_to_ij_coordinates(v_idx, i, j):
    i = v_idx % num_x_cells
    j = v_idx / num_y_cells

    if (i < 0 or j < 0 or i >= num_x_cells or j >= num_y_cells):
        return False
    return True

def ij_coordinates_to_vertex_index(i, j):
    return (j * num_x_cells + i)

def ij_coordinates_to_xy_coordinates(i, j, x, y):
    if (i < 0 or j < 0 or i >= num_x_cells or j >= num_y_cells):
        return False

    x = (i+0.5) * (map_size_x/ num_x_cells)
    y = (j+0.5) * (map_size_y/ num_y_cells)
    return True

def xy_coordinates_to_ij_coordinates(x, y, i, j):
    if (x < 0 or y < 0 or x >= num_x_cells or y >= num_y_cells):
        return False

    i = int((x/map_size_x) * num_x_cells)
    j = int((y/map_size_y) * num_y_cells)
    return True


#############################
#Core Dijkstra Functions #
#############################

def get_travel_cost(vertex_source, vertex_dest):
    are_neighboring = 1

    #source i, j, dest i, j
    s_i = 0
    s_j = 0
    d_i = 0
    d_j = 0

    if (vertex_index_to_ij_coordinates(vertex_source, s_i, s_j) == False):
        return big_number

    if (vertex_index_to_ij_coordinates(vertex_dest, d_i, d_j) == False):
        return big_number

    s_i = vertex_source % num_x_cells
    s_j = vertex_source / num_y_cells
    d_i = vertex_dest % num_x_cells
    d_j = vertex_dest / num_y_cells

##    print "kjaslkdjalkclxzkcklnalcn"
##    print s_i, s_j, d_i, d_j
##    print "laskd;;;;;;;;;;;;;;;lkck"

    are_neighboring = (abs(s_i - d_i) + abs(s_j - d_j) <= 1); # 4-Connected world
##    print are_neighboring
    if (world_map[d_j][d_i] == 1):
        return big_number
    elif (are_neighboring == 1):
        return 1
    else:
        return big_number

def run_dijkstra(world_map, source_vertex):
    #print world_map
    print source_vertex
    cur_vertex = -1;
    cur_cost = -1;
    min_index = -1;
    travel_cost = -1;
    dist = np.zeros(num_x_cells*num_y_cells)
    Q_cost = np.zeros(num_y_cells*num_x_cells)
    prev = prev_holder;
    #print prev

    #Initialize our variables
    for j in range(0, num_y_cells):
        for i in range(0, num_x_cells):
            Q_cost[j*num_x_cells + i] = big_number
            dist[j*num_x_cells + i] = big_number
            prev[j*num_x_cells + i] = -1
    #print Q_cost, dist, prev
    dist[source_vertex] = 0
    Q_cost[source_vertex] = 0

    while (is_not_empty(Q_cost, num_x_cells*num_y_cells) == True):
##        print "MERP"
        min_index = get_min_index(Q_cost, num_x_cells*num_y_cells)
        #print min_index
        if (min_index < 0):
            break; # Queue is empty!

        cur_vertex = min_index # Vertex ID is same as array indices
        cur_cost = Q_cost[cur_vertex]# Current best cost for reaching vertex
        #print cur_cost
        Q_cost[cur_vertex] = -1 # Remove cur_vertex from the queue

        #print Q_cost, prev

        # Iterate through all of node's neighbors and see if cur_vertex provides a shorter
        # path to the neighbor than whatever route may exist currently.
        for neighbor_idx in range(0, num_x_cells*num_y_cells):
            if (Q_cost[neighbor_idx] == -1):
                continue # Skip neighboring nodes that are not in the Queue

            alt = -1
            travel_cost = get_travel_cost(cur_vertex, neighbor_idx)
            #print travel_cost
            if (travel_cost == big_number):
                continue # Nodes are not neighbors/cannot traverse... skip!
            
            alt = dist[cur_vertex] + travel_cost
            if (alt < dist[neighbor_idx]):
                # New shortest path to node at neighbor_idx found!
                dist[neighbor_idx] = alt
                if (Q_cost[neighbor_idx] > 0):
                    Q_cost[neighbor_idx] = alt
                prev[neighbor_idx] = cur_vertex;
                #print neighbor_idx, cur_vertex
                #print prev

    return prev;

def reconstruct_path(prev, source_vertex, dest_vertex):
##    print source_vertex, dest_vertex
##    print prev
    path = np.zeros((num_x_cells * num_y_cells)) # Max-length that path could be
##    print path
    final_path = []
    last_idx = 0

    path[last_idx] = dest_vertex # Start at goal, work backwards
##    print path
    last_vertex = prev[dest_vertex]
##    print last_vertex
    while (last_vertex != -1):
        last_idx += 1
        path[last_idx] = last_vertex
##        print path
        last_vertex = prev[int(last_vertex)]

    for i in range(last_idx + 1):
##        print last_idx
##        print path[last_idx-i]
        final_path.append(path[last_idx-i])# Reverse path so it goes source->dest
##        print final_path   
    final_path.append(-1) # End-of-array marker
##    print final_path
    return final_path;

##world_map[2][0] = 1 #added obstacles for testing purposes
##world_map[0][1] = 1
##world_map[1][3] = 1
print world_map
prev = run_dijkstra(world_map, ij_coordinates_to_vertex_index(source_i, source_j))
print prev
path = reconstruct_path(prev, ij_coordinates_to_vertex_index(source_i, source_j), ij_coordinates_to_vertex_index(goal_i, goal_j))
print path

