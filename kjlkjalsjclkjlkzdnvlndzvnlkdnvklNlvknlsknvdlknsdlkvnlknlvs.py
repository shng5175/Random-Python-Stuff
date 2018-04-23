def BFS(graph, s, t, parent):
    '''Returns true if there is a path from source 's' to sink 't' in
    residual graph. Also fills parent[] to store the path '''

    # Mark all the vertices as not visited
    nodes = graph.nodes()
    visited = {node: False for node in nodes}

    # Create a queue for BFS
    queue = collections.deque()

    # Mark the source node as visited and enqueue it
    queue.append(s)
    visited[s] = True

    # Standard BFS Loop
    while queue:
        u = queue.popleft()

        # Get all adjacent vertices's of the dequeued vertex u
        # If a adjacent has not been visited, then mark it
        # visited and enqueue it
        for ind in graph.adj[u]:
            val = graph[u][ind]['capacity']
            if visited[ind] == False and val > 0 :
                queue.append(ind)
                visited[ind] = True
                parent[ind] = u
 
        # If we reached sink in BFS starting from source, then return
        # true, else false
    return visited[t]

# Returns the maximum flow from s to t in the given graph
def FordFulkerson(graph, source, sink):

    # This array is filled by BFS and to store path
    parent = dict()

    max_flow = 0 # There is no flow initially

    # Augment the flow while there is path from source to sink
    while BFS(graph, source, sink, parent) :

        # Find minimum residual capacity of the edges along the
        # path filled by BFS. Or we can say find the maximum flow
        # through the path found.
        path_flow = float("Inf")
        s = sink
        while s !=  source:
            path_flow = min (path_flow, graph[parent[s]][s]['capacity'])
            s = parent[s]

        # Add path flow to overall flow
        max_flow +=  path_flow

        # update residual capacities of the edges and reverse edges
        # along the path
        v_node = sink
        while v_node !=  source:
            u_node = parent[v_node]
            graph[u_node][v_node]['capacity'] -= path_flow
            graph[v_node][u_node]['capacity'] += path_flow
            v_node = parent[v_node]

    return max_flow

#problem2_c
#input a graph with 
def problem2_c(graph, issue_param, person_param):
    """
    convert the graph G{P ∪ I , E} to G^ = { V^, E^}
    
    """
    new_graph = nx.DiGraph()

    #--------------Part1-----------------------------
    #convert the graph G{P ∪ I , E} to G' = { V', E'}
    # V' = {budget, survay}∪ P ∪ I
    # E' = {(budget, p)| p ∈ P} ∪ {(i, survay) | i ∈ I} ∪ E}

    #Add edges E
    new_graph.add_edges_from([(edge[0], edge[1], {'capacity':1}) for edge in graph.edges])

    # Add edges  {(budget, p)| p ∈ P} , 
    new_graph.add_edges_from([('budget', person, {'capacity': bp_tp[1] - bp_tp[0]}) for person, bp_tp in person_param.items()])
    # Add edges {(i, survay) | i ∈ I} ∪ E} 
    new_graph.add_edges_from([(issue, 'survay', {'capacity': li_ui[1] - li_ui[0]}) for issue, li_ui in issue_param.items()])

    
    #---------------part2----------------------------
    #convert the graph G' = {V' , E'} to G^ = { V^, E^}
    # V^ = {s,t}∪ V'
    # E^ = {(s, p), ('budget',t)| ('budget', p) ∈ E' }
    #      ∪{(s, 'survay') , (i, t) | (i, survay) ∈ E' }
    #      ∪E'

    # add edges (s,p) and ('budget',t)
    # capacity (s,p) and capacity ('budget', t) both are b_p
    new_graph.add_edges_from([('s', person, {'capacity': bp_tp[0]}) for person, bp_tp in person_param.items()])
    new_graph.add_edge('survay', 'budget', capacity = float('Inf'))
    cap = 0
    for person, bp_tp in person_param.items():
        cap += bp_tp[0]
    new_graph.add_edge('budget', 't', capacity = cap)

    # add edges (i,t) and (s, 'survay')
    # capacity (s, 'survay') and capacity(i,t) both are l_i
    new_graph.add_edges_from([(issue, 't', {'capacity': li_ui[0]}) for issue, li_ui in issue_param.items()])
    
    cap = 0
    for issue, li_ui in issue_param.items():
        cap += li_ui[0]
    new_graph.add_edge('s', 'survay', capacity = cap)
    return new_graph

def problem2_d(graph):
    residual_graph = nx.DiGraph()

    residual_graph.add_edges_from([(edge[1], edge[0], {'capacity':0}) for edge in graph.edges()])
    residual_graph.add_edges_from([(edge[0], edge[1], {'capacity': graph[edge[0]][edge[1]]['capacity']}) for edge in graph.edges()])

    max_flow = FordFulkerson(residual_graph, 's', 't')
    full_flow = graph['s']['survay']['capacity']
    if max_flow == full_flow:
        print "Success, the parameters are feasible."
    else:
        print "Failure, the parameters are unfeasible."
    
def problem2_e():
    """Generate test case for problem2_c
    Return:
    G --A graph G = {P ∪ I , E}
    person_param --A dict for {issue_id : (l_i, u_i)}
    issure_param --A dict for {person_id : (b_p, t_p) }
    """
    G = nx.DiGraph()
    issue_size = 10  #n = 10
    issue_id = ['i'+str(i) for i in range(0,issue_size)]
    issue_param = dict() # A dict for {issure_id:(l_i,u_i)}
    
    person_size = 1000  #m = 1000
    person_id = ['p'+str(i) for i in range(0,person_size)]
    person_param = dict() # A dict for {person_id: (b_p, t_p)}
    
    for person in person_id:
        t_p = 0
        for issue in issue_id:
            # a person has a probability of 50% to have a opition aboult the issue
           if random.randint(0, 1) == 0:
               G.add_edge(person, issue)
               t_p += 1
        b_p =  t_p / 2
        person_param [person] = (b_p, t_p)

    for issue in issue_id:
        l_i = random.randint(300,400)   # l_i is drawn unifromly from [300,400]
        u_i = random.randint(500,700)   # u_i is drawn uniformly from [500,700]
        issue_param[issue] = (l_i, u_i)
    return G, issue_param, person_param

    
graph, issue_param, person_param = problem2_e()
graph = problem2_c(graph, issue_param, person_param)

""" if you want to test problem2, uncomment next line."""
problem2_d(graph)
