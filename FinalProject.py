import cv2
import numpy as np
import serial
import math
import matplotlib.pyplot as plt
import time
from collections import deque
import imutils

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
        final_path.append(int(path[last_idx-i]))# Reverse path so it goes source->dest
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


blueSparki = serial.Serial(str('COM6'),int(9600),write_timeout = 1, timeout = 1)
redSparki = serial.Serial(str('COM4'),int(9600),write_timeout = 1, timeout = 1)
#setting theta to 90 for now to keep things easy
theta = 90

blue_points = 0
red_points = 0

# start timer for reference
time.clock()
time_counter = 0
time_between_path_planning = 60

##################
# Blob Detection #
##################

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(0)

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

  # initialize the list of tracked points, the frame counter,
  # and the coordinate delta
  green_pts = deque(maxlen=200)
  green_counter = 0
  (green_dX, green_dY) = (0, 0)
  green_direction = ""

  yellow_pts = deque(maxlen=200)
  yellow_counter = 0
  (yellow_dX, yellow_dY) = (0, 0)
  yellow_direction = ""
  
  if ret == True:

    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #upper and lower bounds of colors need to be changed depending on lighting
    # define range of blue color in RGB
    lower_blue = np.array([0, 7, 32]) 
    upper_blue = np.array([49, 124, 224]) 
    blue_mask = cv2.inRange(rgb, lower_blue, upper_blue)

    lower_red = np.array([24, 0, 1]) 
    upper_red = np.array([179, 35, 41]) 
    red_mask = cv2.inRange(rgb, lower_red, upper_red)

    lower_green = np.array([66, 84, 87]) 
    upper_green = np.array([76, 95, 95])
    green_mask = cv2.inRange(rgb, lower_green, upper_green)
    other_green_mask = cv2.erode(green_mask, None, iterations=2)
    other_green_mask = cv2.dilate(green_mask, None, iterations=2)

    lower_yellow = np.array([209, 191, 84]) 
    upper_yellow = np.array([216, 197, 96]) 
    yellow_mask = cv2.inRange(rgb, lower_yellow, upper_yellow)
    other_yellow_mask = cv2.erode(yellow_mask, None, iterations=2)
    other_yellow_mask = cv2.dilate(yellow_mask, None, iterations=2)

    # mask to get multi colors at once
    mask = blue_mask | red_mask | green_mask | yellow_mask

    # Bitwise-AND mask and original image
    #blue mask
    res_blue = cv2.bitwise_and(frame,frame, mask = blue_mask)
    #red mask
    res_red = cv2.bitwise_and(frame,frame, mask = red_mask)
    #green mask
    res_green = cv2.bitwise_and(frame,frame, mask = green_mask)
    #yellow_mask
    res_yellow = cv2.bitwise_and(frame, frame, mask = yellow_mask)
    #all mask
    res = cv2.bitwise_and(frame,frame, mask = mask)

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

    #Draw detected blobs as red circles.
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


########################################################################
# Finding Blue Sparki's direction #
########################################################################

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    green_cnts = cv2.findContours(other_green_mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    green_center = None
        # only proceed if at least one contour was found
    if len(green_cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        green_c = max(green_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(green_c)
        green_M = cv2.moments(green_c)
        green_center = (int(green_M["m10"] / green_M["m00"]), int(green_M["m01"] / green_M["m00"]))

        ########################################################################
        # This is where we get Blue Sparki's location #
        source_blue_x = int(green_M["m10"] / green_M["m00"])
        source_blue_y = int(green_M["m01"] / green_M["m00"])
        source_blue_i = int(source_blue_x/width)
        source_blue_j = int(source_blue_y/width)
        ########################################################################
      
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, green_center, 5, (0, 0, 255), -1)
            green_pts.appendleft(green_center)
                # loop over the set of tracked points
    for i in np.arange(1, len(green_pts)):
        # if either of the tracked points are None, ignore
        # them
        if green_pts[i - 1] is None or green_pts[i] is None:
            continue
 
        # check to see if enough points have been accumulated in
        # the buffer
        if green_counter >= 10 and i == 1 and green_pts[-10] is not None:
            # compute the difference between the x and y
            # coordinates and re-initialize the direction
            # text variables
            green_dX = green_pts[-10][0] - green_pts[i][0]
            green_dY = green_pts[-10][1] - green_pts[i][1]
            (green_dirX, green_dirY) = ("", "")
 
            # ensure there is significant movement in the
            # x-direction
            if np.abs(dX) > 20:
                green_dirX = "East" if np.sign(green_dX) == 1 else "West"
 
            # ensure there is significant movement in the
            # y-direction
            if np.abs(dY) > 20:
                green_dirY = "North" if np.sign(green_dY) == 1 else "South"
 
            # handle when both directions are non-empty
            if green_dirX != "" and green_dirY != "":
                green_direction = "{}-{}".format(green_dirY, green_dirX)
 
            # otherwise, only one direction is non-empty
            else:
                green_direction = green_dirX if green_dirX != "" else green_dirY

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        green_thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, green_pts[i - 1], green_pts[i], (0, 0, 255), green_thickness)

########################################################################
# Finding Red Sparki's direction #
########################################################################
    yellow_cnts = cv2.findContours(other_yellow_mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    yellow_center = None

    if len(yellow_cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        yellow_c = max(yellow_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(yellow_c)
        yellow_M = cv2.moments(yellow_c)
        yellow_center = (int(yellow_M["m10"] / yellow_M["m00"]), int(yellow_M["m01"] / yellow_M["m00"]))

        ########################################################################
        # This is where we get Red Sparki's location #
        source_red_x = int(yellow_M["m10"] / yellow_M["m00"])
        source_red_y = int(yellow_M["m01"] / yellow_M["m00"])
        source_red_i = int(source_red_x/width)
        source_red_j = int(source_red_y/width)
        ########################################################################
        
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, yellow_center, 5, (0, 0, 255), -1)
            yellow_pts.appendleft(yellow_center)
                # loop over the set of tracked points
    for i in np.arange(1, len(yellow_pts)):
        # if either of the tracked points are None, ignore
        # them
        if yellow_pts[i - 1] is None or yellow_pts[i] is None:
            continue
 
        # check to see if enough points have been accumulated in
        # the buffer
        if yellow_counter >= 10 and i == 1 and yellow_pts[-10] is not None:
            # compute the difference between the x and y
            # coordinates and re-initialize the direction
            # text variables
            yellow_dX = yellow_pts[-10][0] - yellow_pts[i][0]
            yellow_dY = yellow_pts[-10][1] - yellow_pts[i][1]
            (yellow_dirX, yellow_dirY) = ("", "")
 
            # ensure there is significant movement in the
            # x-direction
            if np.abs(dX) > 20:
                yellow_dirX = "East" if np.sign(yellow_dX) == 1 else "West"
 
            # ensure there is significant movement in the
            # y-direction
            if np.abs(dY) > 20:
                yellow_dirY = "North" if np.sign(yellow_dY) == 1 else "South"
 
            # handle when both directions are non-empty
            if yellow_dirX != "" and yellow_dirY != "":
                yellow_direction = "{}-{}".format(yellow_dirY, yellow_dirX)
 
            # otherwise, only one direction is non-empty
            else:
                yellow_direction = yellow_dirX if yellow_dirX != "" else yellow_dirY

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        yellow_thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, yellow_pts[i - 1], yellow_pts[i], (0, 0, 255), yellow_thickness)
################################################################################

################################################################
# Should implement path finding and sparki communications here #
################################################################
#First find closest block
    blue_closest = 999
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
          blue_closest = blue_dist
          blue_closest_idx = ((j * num_x_cells) +  k)
          goal_blue_i = j
          goal_blue_j = k
      #print blue_closest_idx
    red_closest = 999
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
        red_dist = distance(source_red_i, source_red_j, j, k)
        if (red_dist < red_closest):
          red_closest = red_dist
          red_closest_idx = ((j * num_x_cells) +  k)
          goal_red_i = j
          goal_red_j = k
      #print red_closest_idx

##    for keyPoint_green in key_green:
##      source_blue_x = keyPoint_green.pt[0]
##      source_blue_y = keyPoint_green.pt[1]
##      source_blue_i = int(source_blue_x/width)
##      source_blue_j = int(source_blue_y/width)
##
##    for keyPoint_yellow in key_yellow:
##      source_red_x = keyPoint_yellow.pt[0]
##      source_red_y = keyPoint_yellow.pt[1]
##      source_red_i = int(source_red_x/width)
##      source_red_j = int(source_red_y/width)
        
###########################
#                         #
#      Sparki Commands    #
#                         #
###########################
    #if enough time hasn't elapsed, don't send commands
    if (time.clock() > time_between_path_planning * time_counter):
        time_counter = time_counter + 1
    
    #Calculate path
        blue_prev = run_dijkstra(world_map, ij_coordinates_to_vertex_index(source_blue_i, source_blue_j))
        blue_path = reconstruct_path(blue_prev, ij_coordinates_to_vertex_index(source_blue_i, source_blue_j), ij_coordinates_to_vertex_index(goal_blue_i, goal_blue_j))
        print "Blue Path:" , blue_path

        prev = []

        #print ij_coordinates_to_vertex_index(source_red_i, source_red_j)
        red_prev = run_dijkstra(world_map, ij_coordinates_to_vertex_index(source_red_i, source_red_j))
        red_path = reconstruct_path(red_prev, ij_coordinates_to_vertex_index(source_red_i, source_red_j), ij_coordinates_to_vertex_index(goal_red_i, goal_red_j))
        print "Red Path:" , red_path

        blueSparki.write('r')
        print "reset to Blue Sparki"
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
            else:
                theta = 0

            if blue_path[i] < 10:
                blue_path[i] = "00"+str(blue_path[i])
            elif blue_path[i]>9 and blue_path[i]<100:
                blue_path[i] = "0"+str(blue_path[i])
            else:
                blue_path[i] = str(blue_path[i])
            print "Output to blue sparki:", blue_path[i]

            if theta < 10:
                theta = "00"+str(theta)
            elif theta>9 and theta<100:
                theta = "0"+str(theta)
            else:
                theta = str(theta)
            print theta
            blueSparki.write('3,'+str(blue_path[i])+','+str(green_direction)+'n')
            time.sleep(1)
        if (blue_has_block):
            blueSparki.write('2,000,000n')
            print ("Sparki has given up a Block")
            blue_has_block = False

        else:
            blueSparki.write('1,000,000n')
            print ("Sparki has the block")
            blue_has_block = True

        #redSparki.write('r')
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
            if red_path[i] < 10:
                red_path[i] = "00"+str(red_path[i])
            elif red_path[i]>9 and red_path[i]<100:
                red_path[i] = "0"+str(red_path[i])
            else:
                red_path[i] = str(red_path[i])
            print "Output to red Sparki"

            if theta < 10:
                theta = "00"+str(theta)
            elif theta>9 and theta<100:
                theta = "0"+str(theta)
            else:
                theta = str(theta)
            print theta
            #redSparki.write('3,'+str(red_path[i])+','+str(theta)+'n')
            time.sleep(1)

        if (red_has_block):
            #redSparki.write('2,000,000n')
            red_has_block = False

        else:
            #redSparki.write('1,000,000n')
            red_has_block = True

    

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
    
    cv2.putText(frame, green_direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 255), 3)
    cv2.putText(frame, "dx: {}, dy: {}".format(green_dX, green_dY),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)
    cv2.putText(frame, yellow_direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 255), 3)
    cv2.putText(frame, "dx: {}, dy: {}".format(yellow_dX, yellow_dY),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)
    cv2.imshow("Color Mask and Blob Detection BETTER BLUE", image_key_blue)
    cv2.imshow("Color Mask and Blob Detection BETTER RED", image_key_red)
    cv2.imshow("Color Mask and Blob Detection BETTER GREEN", image_key_green)
    cv2.imshow("Color Mask and Blob Detection BETTER YELLOW", image_key_yellow)
    #cv2.imshow("Color Mask and Blob Detection BETTER ALL", image_key)
    cv2.imshow('Frame',frame)
    green_counter += 1
    #higher wait means 'slow-mo"
    k = cv2.waitKey(5) & 0xFF
    #if k == 27:
    #    break
    
    
cv2.destroyAllWindows()
blueSparki.close()
#redSparki.close()
