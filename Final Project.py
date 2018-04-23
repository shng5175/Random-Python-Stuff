import cv2
import numpy as np
import serial
import math
import matplotlib.pyplot as plt
import PySparkiBluetooth as psb

num_x_cells = 14
num_y_cells = 11

map_size_x = 80
map_size_y = 80

big_number = 255

prev_holder = np.zeros(num_x_cells*num_y_cells)

world_map = np.zeros((num_y_cells, num_x_cells))
#Location of goals, hardcoded in to make things easier
world_map[0][0] = 1
world_map[10][0] = 1
world_map[0][13] = 1
world_map[10][13] = 1
world_map[0][6] = 1
world_map[10][6] = 1
world_map[5][7] = 1

goals = [[0,0], [10,0], [0,13], [10,13], [0,6], [10, 6], [5, 7]]
#print goals

goal_changed = True
blue_has_block = False
red_has_block = False
goal_blue_i = 0
goal_blue_j = 0
goal_red_i = 0
goal_red_j = 0
source_blue_i = 0
source_blue_j = 5
source_red_i = 13
source_red_j = 5
index = 0
prev = None
path = None
blue_closest = big_number
red_closest = big_number
blue_blocks = []
red_blocks = []

def distance(source_i, source_j, goal_i, goal_j):
    dist = math.sqrt(math.pow((source_i-goal_i), 2)+ math.pow((source_j-goal_j), 2))
    return dist

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
    #print source_vertex
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
##print world_map
##prev = run_dijkstra(world_map, ij_coordinates_to_vertex_index(source_i, source_j))
##print prev
##path = reconstruct_path(prev, ij_coordinates_to_vertex_index(source_i, source_j), ij_coordinates_to_vertex_index(goal_i, goal_j))
##print path

#This is how we find path
##prev = run_dijkstra(world_map, ij_coordinates_to_vertex_index(source_i, source_j))
##path = reconstruct_path(prev, ij_coordinates_to_vertex_index(source_i, source_j), ij_coordinates_to_vertex_index(goal_i, goal_j))

##blueSparki = PySparkiBluetooth.init('COM5')
##redSparki = PySparkiBluetooth.init('COM6')
#setting theta to 90 for now to keep things easy
theta = 90

blue_points = 0
red_points = 0

##################
# Blob Detection #
##################

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(1)

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Cannot Open Camera")

#camara size
cap.set(3,1120) #width
cap.set(4,880) #height
camera_width = 1120
camera_height = 880

#55cmx70cm
#map:11x14 approx 5 cm cells

width = height = 80

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # define range of blue color in RGB
    lower_blue = np.array([0, 7, 32]) #2,25, 77
    upper_blue = np.array([49, 124, 224]) #48,96,207
    blue_mask = cv2.inRange(rgb, lower_blue, upper_blue)

##    lower_green = np.array([1,18,2]) #1,18,2
##    upper_green = np.array([78,84,36]) # 78,84,36
##    green_mask = cv2.inRange(rgb, lower_green, upper_green)

    lower_red = np.array([24, 0, 1]) #41,6,10
    upper_red = np.array([179, 35, 41]) #90,30,32
    red_mask = cv2.inRange(rgb, lower_red, upper_red)

    lower_green = np.array([51, 71, 87]) #41,6,10
    upper_green = np.array([76, 100, 117]) #90,30,32
    green_mask = cv2.inRange(rgb, lower_green, upper_green)

    lower_yellow = np.array([192, 176, 82]) #41,6,10
    upper_yellow = np.array([214, 196, 129]) #90,30,32
    yellow_mask = cv2.inRange(rgb, lower_yellow, upper_yellow)

    # mask to get multi colors at once
    mask = blue_mask | red_mask

    # Bitwise-AND mask and original image
    #blue mask
    res_blue = cv2.bitwise_and(frame,frame, mask= blue_mask)
    #red mask
    res_red = cv2.bitwise_and(frame,frame, mask= red_mask)
    #green mask
    res_green = cv2.bitwise_and(frame,frame, mask= green_mask)
    #yellow_mask
    res_yellow = cv2.bitwise_and(frame, frame, mask = yellow_mask)
    #all mask
    res = cv2.bitwise_and(frame,frame, mask= mask)

    #blur the frames
    #normal blur
    ##blur = cv2.blur(res,(5,5))
    #gaussian blur
    ##blur = cv2.GaussianBlur(res,(5,5),0)
    #median blur
    blur = cv2.medianBlur(res,5)
    blur_blue = cv2.medianBlur(res_blue,5)
    blur_red = cv2.medianBlur(res_red,5)
    blur_green = cv2.medianBlur(res_green,5)
    blur_yellow = cv2.medianBlur(res_yellow, 5)

    # Set up the detector with parameters.
    #params = cv2.SimpleBlobDetector_Params()
    #detector = cv2.SimpleBlobDetector()
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 256;

    #Filter by Area
    params.filterByArea = True
    params.minArea = 250 #blobs of pixels

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.001

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    #params.filterByInertia = True
    #params.minInertiaRatio = 0.01

    #Filter by Color
    # 0 for dark blobs, 255 for light blobs
    #params.filterByColor= True
    #params.blobColor= 255

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else :
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints_blue = detector.detect(blur_blue)
    keypoints_red = detector.detect(blur_red)
    keypoints_green = detector.detect(blur_green)
    keypoints_yellow = detector.detect(blur_yellow)
    keypoints = detector.detect(blur)

##    for keyPoint_blue in keypoints_blue:
##      print np.argwhere(keyPoint_blue == blur_blue)
##      x_blue=keyPoint_blue.pt[0]
##      y_blue=keyPoint_blue.pt[1]
##      #s=keyPoint.size
##      print ("Blue", x_blue, y_blue)

##    for keyPoint_red in keypoints_red:
##        x_red=keyPoint_red.pt[0]
##        y_red=keyPoint_red.pt[1]
##        #s=keyPoint.size
##        print ("Red", x_red, y_red)

##    for keyPoint_green in keypoints_green:
##        x_green=keyPoint_green.pt[0]
##        y_green=keyPoint_green.pt[1]
##        #s=keyPoint.size
##        print ("Green", x_green, y_green)


    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    #blue
    image_with_keypoints_blue = cv2.drawKeypoints(blur_blue, keypoints_blue, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    revmask_blue=255-blur_blue
    key_blue=detector.detect(revmask_blue)
    image_key_blue=cv2.drawKeypoints(blur_blue, key_blue, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    for keyPoint_blue in key_blue:
      x_blue=keyPoint_blue.pt[0]
      y_blue=keyPoint_blue.pt[1]
      x_blue_data = int(x_blue/width)
      y_blue_data = int(y_blue/height)
      if(world_map[y_blue_data][x_blue_data] != 1):
        world_map[y_blue_data][x_blue_data] = 2
        blue_blocks.append([x_blue_data, y_blue_data])
      #s=keyPoint.size
      #print ("Blue", x_blue, y_blue)

    #red
    image_with_keypoints_red = cv2.drawKeypoints(blur_red, keypoints_red, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    revmask_red=255-blur_red
    key_red=detector.detect(revmask_red)
    image_key_red=cv2.drawKeypoints(blur_red, key_red, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    for keyPoint_red in key_red:
      x_red=keyPoint_red.pt[0]
      y_red=keyPoint_red.pt[1]
      x_red_data = int(x_red/width)
      y_red_data = int(y_red/height)
      if (world_map[y_red_data][x_red_data] != 1):
        red_blocks.append([x_red_data, y_red_data])
        world_map[y_red_data][x_red_data] = 3
      #s=keyPoint.size
      #print ("Red", x_red, y_red)

    #print blue_blocks, red_blocks

    #green
    image_with_keypoints_green = cv2.drawKeypoints(blur_green, keypoints_green, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    revmask_green=255-blur_green
    key_green=detector.detect(revmask_green)
    image_key_green=cv2.drawKeypoints(blur_green, key_green, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #yellow
    image_with_keypoints_yellow = cv2.drawKeypoints(blur_yellow, keypoints_yellow, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    revmask_yellow=255-blur_yellow
    key_yellow=detector.detect(revmask_yellow)
    image_key_yellow=cv2.drawKeypoints(blur_yellow, key_yellow, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #print world_map
################################################################
# Should implement path finding and sparki communications here #
################################################################
#First find closest block
    if (blue_has_block == False):
      for i in range(len(blue_blocks)):
        j = blue_blocks[i][0]
        k = blue_blocks[i][1]
        #print j, k
        blue_dist = distance(source_blue_i, source_blue_j, j, k)
        if (blue_dist < blue_closest):
          blue_closest = blue_dist
          blue_closest_idx = ((j * num_x_cells) +  k)
          goal_blue_i = j
          goal_blue_j = k
    elif (blue_has_block == True):
      for i in range(len(goals)):
        j = goals[i][0]
        k = goals[i][1]
        #print j, k
        blue_dist = distance(source_blue_i, source_blue_j, j, k)
        if (blue_dist < blue_closest):
          blue_c
          losest = blue_dist
          blue_closest_idx = ((j * num_x_cells) +  k)
          goal_blue_i = j
          goal_blue_j = k
      #print blue_closest_idx    
          
    if (red_has_block == False):
      for i in range(len(red_blocks)):
        j = red_blocks[i][0]
        k = red_blocks[i][1]
        #print j, k
        red_dist = distance(source_red_i, source_red_j, j, k)
        if (red_dist < red_closest):
          red_closest = red_dist
          red_closest_idx = ((j * num_x_cells) +  k)
          goal_red_i = j
          goal_red_j = k
    elif (red_has_block == True):
      for i in range(len(goals)):
        j = goals[i][0]
        k = goals[i][1]
        #print j, k
        blue_dist = distance(source_red_i, source_red_j, j, k)
        if (red_dist < red_closest):
          red_closest = red_dist
          red_closest_idx = ((j * num_x_cells) +  k)
          goal_red_i = j
          goal_red_j = k
      #print red_closest_idx

    for keyPoint_green in key_green:
      source_blue_x = keyPoint_green.pt[0]
      source_blue_y = keyPoint_green.pt[1]
      source_blue_i = int(source_blue_x/width)
      source_blue_j = int(source_blue_y/width)

    for keyPoint_yellow in key_yellow:
      source_red_x = keyPoint_yellow.pt[0]
      source_red_y = keyPoint_yellow.pt[1]
      source_red_i = int(source_red_x/width)
      source_red_j = int(source_red_y/width)
        
    #Calculate path
    blue_prev = run_dijkstra(world_map, ij_coordinates_to_vertex_index(source_blue_i, source_blue_j))
    blue_path = reconstruct_path(blue_prev, ij_coordinates_to_vertex_index(source_blue_i, source_blue_j), ij_coordinates_to_vertex_index(goal_blue_i, goal_blue_j))
    print "Blue Path:" , blue_path

    prev = []

    #print ij_coordinates_to_vertex_index(source_red_i, source_red_j)
    red_prev = run_dijkstra(world_map, ij_coordinates_to_vertex_index(source_red_i, source_red_j))
    red_path = reconstruct_path(red_prev, ij_coordinates_to_vertex_index(source_red_i, source_red_j), ij_coordinates_to_vertex_index(goal_red_i, goal_red_j))
    print "Red Path:" , red_path
    
    for i in range(len(blue_path)-1):
      if (blue_path[i+1] != -1):
        if(blue_path[i+1] == blue_path[i] + 1):
          theta = 0
        elif(blue_path[i+1] == blue_path[i] - 1):
          theta = 180
        elif(blue_path[i+1] == blue_path[i] + 14):
          theta = 270
        elif(blue_path[i+1] == blue_path[i] - 14):
          theta = 90
      #print theta
      #blueSparki.move(blue_path[i], theta)

    for i in range(len(red_path)-1):
      if (red_path[i+1] != -1):
        if(red_path[i+1] == red_path[i] + 1):
          theta = 0
        elif(red_path[i+1] == red_path[i] - 1):
          theta = 180
        elif(red_path[i+1] == red_path[i] + 14):
          theta = 270
        elif(red_path[i+1] == red_path[i] - 14):
          theta = 90
##      redSparki.move(red_path[i], theta)      
      
    blue_blocks = []
    red_blocks = []
    #print blue_blocks, red_blocks

##    #all
##    image_with_keypoints = cv2.drawKeypoints(blur, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
##    revmask=255-blur
##    key=detector.detect(revmask)
##    image_key=cv2.drawKeypoints(blur, key, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    #display results
    #cv2.imshow("Color Mask and Blob Detection BETTER BLUE", image_key_blue)
    #cv2.imshow("Color Mask and Blob Detection BETTER RED", image_key_red)
    cv2.imshow("Color Mask and Blob Detection BETTER GREEN", image_key_green)
    cv2.imshow("Color Mask and Blob Detection BETTER YELLOW", image_key_yellow)
    #cv2.imshow("Color Mask and Blob Detection BETTER ALL", image_key)
    cv2.imshow('Frame',frame)
    #higher wait means 'slow-mo"
    k = cv2.waitKey(5) & 0xFF
    #if k == 27:
    #    break

cv2.destroyAllWindows()
