import serial

def init(com_port,baudrate=9600):
    self = serial.Serial(str(com_port),int(9600),write_timeout = 1, timeout = 1)

def close():
    self.write('1,000,000n')

def release():
    self.write('2,000,000n')

def move(map_index, theta):
    if map_index < 10:
        map_index = "00"+str(map_index)
    elif map_index>9 and map_index<100:
        map_index = "0"+str(map_index)
    else:
        map_index = str(map_index)
    print map_index
    
    if theta < 10:
        theta = "00"+str(theta)
    elif theta>9 and theta<100:
        theta = "0"+str(theta)
    else:
        theta = str(theta)
    print theta
    self.write('3,'+map_index+','+theta+'n')

def update_pos(map_index,theta):
    if map_index < 10:
        map_index = "00"+str(map_index)
    elif map_index>9 and map_index<100:
        map_index = "0"+str(map_index)
    else:
        map_index = str(map_index)
    print map_index
    
    if theta < 10:
        theta = "00"+str(theta)
    elif map_index>9 and map_index<100:
        theta = "0"+str(theta)
    else:
        theta = str(theta)
    print theta
    self.write('4,'+map_index+','+theta+'n')
    
def reset():
    self.write('r')
