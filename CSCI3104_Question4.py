import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#create a directed graph
G = nx.DiGraph()

#adding an edge also adds the node
G.add_edge('Spider', 'A', weight=1.0)
G.add_edge('Spider', 'H', weight=1.0)
G.add_edge('Spider', 'J', weight=1.0)

G.add_edge('H', 'G', weight=1.0)
G.add_edge('H', 'K', weight=1.0)

G.add_edge('G', 'L', weight=1.0)
G.add_edge('G', 'F', weight=1.0)

G.add_edge('F', 'E', weight=1.0)

G.add_edge('E', 'Fly', weight=1.0)

G.add_edge('J', 'S', weight=1.0)
G.add_edge('J', 'K', weight=1.0)

G.add_edge('K', 'L', weight=1.0)
G.add_edge('L', 'M', weight=1.0)
G.add_edge('M', 'N', weight=1.0)
G.add_edge('M', 'F', weight=1.0)

G.add_edge('N', 'O', weight=1.0)
G.add_edge('N', 'E', weight=1.0)

G.add_edge('O', 'Fly', weight=1.0)

G.add_edge('A', 'S', weight=1.0)
G.add_edge('A', 'B', weight=1.0)

G.add_edge('B', 'R', weight=1.0)
G.add_edge('B', 'C', weight=1.0)

G.add_edge('S', 'R', weight=1.0)
G.add_edge('R', 'Q', weight=1.0)

G.add_edge('Q', 'C', weight=1.0)
G.add_edge('Q', 'P', weight=1.0)

G.add_edge('C', 'D', weight=1.0)
G.add_edge('D', 'Fly', weight=1.0)
G.add_edge('P', 'D', weight=1.0)
G.add_edge('P', 'O', weight=1.0)
G.add_edge('O', 'Fly', weight=1.0)

G.add_edge('T', 'Q', weight=1.0)
G.add_edge('T', 'P', weight=1.0)
G.add_edge('T', 'O', weight=1.0)
G.add_edge('T', 'N', weight=1.0)
G.add_edge('T', 'M', weight=1.0)

G.add_edge('R', 'T', weight=1.0)
G.add_edge('S', 'T', weight=1.0)
G.add_edge('J', 'T', weight=1.0)
G.add_edge('K', 'T', weight=1.0)
G.add_edge('L', 'T', weight=1.0)

#each edge has a weight of 1. The shortest path is the fewest edges.
#Use this to verify that your graph built correctly.
t = nx.shortest_path(G, 'Spider', 'Fly', weight='weight')

#print(t)

#Tried to get a visual of the graph
#val_map = {'Spider': 1.0,
#           'J': 0.2,
#           'T': 0.4,
#           'O' : 0.6,
#           'Fly': 0.0}
#values = [val_map.get(node, 0.25) for node in G.nodes()]
#nx.draw(G, cmap = plt.get_cmap('jet'), node_color = values)
#plt.show()

#Messing around with networkx and learning about the built-in functions that might be useful
#print G.node()
#print G.edges()
#print G.edges('Spider')
#print G.in_edges('Fly')
#print G.has_edge('O', 'Fly')
#print len(G)

###############################################################################
#This is where I started implementing my algorithm
length = len(G)#number of nodes in G
nodes = ['Fly']#list of nodes to be traversed starting with the node 'Fly'
#print len(G.in_edges('Fly'))

#Find order in which nodes should be traversed
#There's probably a better way to do this, but if it works, that's all that matters to me right now
##for j in range(length):
##    for i in G.nodes:
##        if (G.has_edge(i, nodes[j])):
##            print j, nodes[j], i
##            nodes.append(i)
##            l = len(nodes)
##            print l
##            for k in range(l - 1):
##                print 'kjakljflsajfljslkvdjlsavjls', k, nodes[k], i
##                if nodes[k] == i:
##                    print k, nodes[k], i
##                    nodes.pop(k)
##                    if (k < j):
##                        j = j-1
##                    break
##    print nodes
##    print j
##print nodes   
###print len(nodes)
##if(len(nodes)!= len(G)):
##    print "ERROR"

#guess my algorithm actually starts here, sorry about that
#Part of the list this generates isn't quite correct, I'll try working on it some more if I have time
#Should be something like:
node_list = ['Fly', 'E', 'D', 'O', 'F', 'C', 'P', 'N', 'Q', 'M',
             'T', 'R', 'L', 'K', 'S', 'B', 'G', 'H', 'A', 'J', 'Spider']
node_count = np.zeros(len(node_list))#number of paths from 'Fly' to itself should be 0

for i in range(1, len(node_list)):
    if G.has_edge(node_list[i], node_list[0]):#if has edge to goal, increment number of paths for that node
        node_count[i] += 1
    #Check to see if has edge from any or the nodes that we've already solved for number of paths to the goal
    for j in range(i, 0, -1):
        #If there is an edge, then the number of paths from that node is equal to itself plus the number of paths it has an edge with
        if G.has_edge(node_list[i], node_list[j]):
            node_count[i] += node_count[j]
print node_list
print node_count
print 'Total number of paths from Spider to Fly: ', int(node_count[i]), ' paths'

###############################################################################
#Answer I should get
paths = nx.all_simple_paths(G, source='Spider', target='Fly', cutoff=21)
#print len(list(paths))
if (int(node_count[i]) == len(list(paths))):
    print 'Correct!'



