import gurobipy as gp
import itertools as it
import networkx as nx
import pandas as pd
import numpy as np
import math
import re

"""
implementation of solving profit maximization problem
author(s): 
yk796@cornell.edu
nd396@cornell.edu
"""

# link to https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls

network_df = pd.read_csv("../data/SiouxFalls/SiouxFalls_net.txt", sep='\t', comment=';')
node_df = pd.read_csv("../data/SiouxFalls/SiouxFalls_node.txt", sep='\t', comment=';')
od_df = pd.read_csv("../data/SiouxFalls/SiouxFalls_od.csv")

"""
configuration
"""
# od_df = od_df[:5]
od_df = od_df.sample(n=20, random_state=42) 


fuel_cost_per_min = 1/10 # $5/hr # change it to 1/12
vot = 20 #$/hr
p_sen = 1/vot*60 # cost to min
Transit_ASC = -10

file_name = "output_transit_sampled_20.txt" # here you can save the solution to the text file, and do analysis with analyze_result.ipynb.
# # scenario 1 
# transit_line = []  

# scenario 2
transit_line = [(4, 11), (11, 14), (14, 23), (23, 24), (5, 9), (9, 10), (10, 15), (15, 22), (22, 21),
                (11, 4), (14, 11), (23, 14), (24, 23), (9, 5), (10, 9), (15, 10), (22, 15), (21, 22)]  

# # scenario 3
# transit_line = [(1, 3), (3, 12), (12, 13), (4, 11), (11, 14), (14, 23), (23, 24), (5, 9), (9, 10), (10, 15), (15, 22), (22, 21), (2, 6), (6, 8), (8, 16), (16, 17), (17, 19), (19, 20),
#                 (3, 1), (12, 3), (13, 12), (11, 4), (14, 11), (23, 14), (24, 23), (9, 5), (10, 9), (15, 10), (22, 15), (21, 22), (6, 2), (8, 6), (16, 8), (17, 16), (19, 17), (20, 19)]  


"""
end
"""

network_df = network_df[['init_node', 'term_node', 'capacity', 'length', 'free_flow_time', 'b',
       'power', 'speed', 'toll', 'link_type']]
network_df[['init_node', 'term_node']] = network_df[['init_node', 'term_node']].astype(int)
node_df = node_df[['Node', 'X', 'Y']]

print("Number of nodes:", len(node_df))
print("Number of links:", len(network_df))
print("Number of od pairs:", len(od_df))

def generate_route_sets_link_elimination(graph, source, target, num_routes):
    route_sets = []
    for i in range(num_routes):
        path = nx.shortest_path(graph, source=source, target=target, weight='weight')
        if path not in route_sets:
            route_sets.append(path)
        
        if len(path) > 2:
            edge_to_remove = (path[1], path[2])
            original_weight = graph[edge_to_remove[0]][edge_to_remove[1]]['weight']
            graph[edge_to_remove[0]][edge_to_remove[1]]['weight'] = float('inf')
            
        else:
            break
        
    # Reset graph weights for future usage
    nx.set_edge_attributes(graph, original_weight, 'weight')
    
    return route_sets

# link penalty approach
def generate_route_sets_link_penalty(graph, source, target, num_routes, penalty_factor=1.5):
    route_sets = []
    for i in range(num_routes):
        path = nx.shortest_path(graph, source=source, target=target, weight='weight')
        if path not in route_sets:
            route_sets.append(path)
        
        for j in range(len(path) - 1):
            edge = (path[j], path[j+1])
            graph[edge[0]][edge[1]]['weight'] *= penalty_factor 
            
    # Reset graph weights for future usage
    nx.set_edge_attributes(graph, 1, 'weight')
    
    return route_sets


transit_link = []
for index, row in network_df.iterrows():
    init_node, term_node = int(row['init_node']), int(row['term_node'])
    length = row['length']
    if (init_node, term_node) in transit_line:
        transit_link.append((init_node, term_node, length*0.5))
    else:
        transit_link.append((init_node, term_node, length*3))

n_nodes = len(node_df)


n_alternative = 1
n_routes = 3


# road_bpr_dict = {(row['init_node'], row['term_node']): lambda flow: row['free_flow_time']*(1+row['b']*(flow/row['capacity'])**row['power']) for index, row in network_df.iterrows()}

bpr_func = {}

for index, row in network_df.iterrows():
    init_node, term_node = int(row['init_node']), int(row['term_node'])
    free_flow_time, b, capacity, power = row['free_flow_time'], row['b'], row['capacity'], row['power']
    
    bpr_func[(init_node, term_node)] = lambda flow, f=free_flow_time, b=b, c=capacity, p=power: f * (1 + b * (flow / c)) # 1 should be substitue with power
    



nodes = node_df['Node'].to_list()
alternatives = list(range(1, n_alternative+1))
arcs = list(network_df[['init_node', 'term_node']].to_records(index=False))
#ods = list(it.permutations(nodes, 2))
# ods = [(id1+1, id2+1) for id1, o in enumerate(O_demand) for id2, d in enumerate(D_demand) if o>0 or d>0 if id1 != id2]

demand = {(int(row['O']), int(row['D'])): row['Ton'] for index, row in od_df.iterrows()}
ods = list(demand.keys() )
road_link = [(int(row['init_node']), int(row['term_node']), row['length']) for _, row in network_df.iterrows()]
oper_cost = {(int(row['init_node']), int(row['term_node'])): row['length'] * fuel_cost_per_min for _, row in network_df.iterrows()}


# Find shortest path travel time
OD_travel_time = {}
G = nx.DiGraph()
G.add_weighted_edges_from(road_link)
for od_pair_index in range(len(od_df)):
    i,j = od_df['O'].iloc[od_pair_index], od_df['D'].iloc[od_pair_index]
    # generate shortest path
    path_tt = nx.shortest_path_length(G, i, j, weight='weight')
    OD_travel_time[(i,j)] = path_tt

# Find shortest path travel time
OD_transit_time = {}
G = nx.DiGraph()
G.add_weighted_edges_from(transit_link)
for od_pair_index in range(len(od_df)):
    i,j = od_df['O'].iloc[od_pair_index], od_df['D'].iloc[od_pair_index]
    # generate shortest path
    path_tt = nx.shortest_path_length(G, i, j, weight='weight')
    OD_transit_time[(i,j)] = path_tt


OD_route = {}
for od_pair_index in range(len(od_df)):
    i,j = od_df['O'].iloc[od_pair_index], od_df['D'].iloc[od_pair_index]
    G = nx.DiGraph()
    G.add_weighted_edges_from(road_link)
    route_sets = generate_route_sets_link_penalty(G, i, j, n_routes) 
    OD_route[(i,j)] = route_sets
    
T = {}
for (s, t) in ods:
    T[(s, t), 1] = OD_travel_time[(s,t)] # MoD
    T[(s, t), 2] = OD_transit_time[(s,t)] # Transit


ASC = {}
for (s, t) in ods:
    ASC[(s, t), 1] = 0 # MoD
    ASC[(s, t), 2] = Transit_ASC # Transit


def create_route(od):
    (o, d) = od
    # find all possible routes
    return (o, d)
    

def indicator(arc, route): 
    '''
    To check if an arc is in route
    '''
    # Check if arc is a tuple and has 2 elements
    if not isinstance(arc, tuple) or len(arc) != 2:
        raise ValueError("Arc must be a tuple with 2 elements")

    # Iterate through the route and check each pair
    for i in range(len(route) - 1):
        if route[i] == arc[0] and route[i+1] == arc[1]:
            return True
    return False


def profit_maximization(n_nodes, arcs, routes, n_alternative, ods, demand, T, ASC, bpr_func):
    eps = 1e-3

    m = gp.Model()
    m.Params.NonConvex = 2 
    m.Params.DualReductions = 0 # to determine if the model is infeasible or unbounded
    m.setParam("MIPGap", 0.01) # Set the gap to 1 % 0.2
    # m.Params.OutputFlag = 0
    # m.Params.NonConvex = 2 # because of profit_extracting_log term is concave, we need to tell gurobi
    # that this is not a concave model, although it is. 
    #m.Params.MIPGap = 0


    m._T = T
    m._ASC = ASC
    m._bpr_func = bpr_func
    m._nodes = list(range(1, n_nodes+1))
    m._alternatives = list(range(1, n_alternative+1))
    m._arcs = [tuple(int(a) for a in arc) for arc in arcs]
    m._ods = ods #[(id1+1, id2+1) for id1, o in enumerate(O_demand) for id2, d in enumerate(D_demand) if o>0 or d>0 if id1 != id2]
    #m._routes = routes# {key:create_route(key) for key in m._ods}

    m._routes = routes

    for (o, d) in m._ods:
        # Convert to list of tuples with native Python types
        m._routes[(o, d)] = [tuple(int(r) for r in route) for route in routes[(o, d)]]

    m._theta_vars = m.addVars(list(it.product(m._ods, m._alternatives)), vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name='theta')
    # when theta is small, the problem domain is ill-defined 
    # m._ln_theta_vars = m.addVars(list(it.product(m._ods, m._alternatives)), vtype=gp.GRB.CONTINUOUS, lb = -float('inf'), ub=0, name='ln_theta')
    # m._ln_theta_n = m.addVars(m._ods, vtype=gp.GRB.CONTINUOUS, lb = -float('inf'), ub=0, name='ln_theta_n')

    # m._pi_vars = m.addVars(list(it.product(m._ods, m._alternatives)), vtype=gp.GRB.CONTINUOUS, lb=0, ub=3, name='pi')
    m._y_vars = m.addVars(list(it.product(m._arcs, m._ods, m._alternatives)), vtype=gp.GRB.CONTINUOUS, lb=0, name='y') # in fully connected graph, y=z
    
    # for theta_var in m._theta_vars.values():
    #     theta_var.start = 0.5
    
    z_vars = []
    for (o, d) in m._ods:
        # Create a list of tuples with native Python types for each (o, d) pair and append it to z_vars
        z_vars.extend([((o, d), tuple(int(r) for r in route), int(alt))
                            for route, alt in it.product(m._routes[(o, d)], m._alternatives)])
        
    m._z_vars = m.addVars(z_vars, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'z')
    
    # create auxiliary variables to deal with non-linear objective function
    m._f_vars = m.addVars(m._arcs, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'f')
    
    
    # # Dictionary for initial values
    # initial_values = {(1, 2): 100, (2, 4): 100, (1, 3):100, (3, 4): 100}

    # # Loop to set initial values
    # for arc, init_val in initial_values.items():
    #     m._f_vars[arc].Start = init_val
    
    m._theta_lntheta_vars = m.addVars(list(it.product(m._ods, m._alternatives)), lb = -float('inf'), ub = 0, vtype=gp.GRB.CONTINUOUS, name='theta_ln_theta') #define theta * ln(theta)
    m._congest_tt = m.addVars(list(it.product(m._ods, m._alternatives)), lb = 0, vtype=gp.GRB.CONTINUOUS, name='congested_travel_time') 


    m._ind = m.addVars(m._ods, vtype=gp.GRB.BINARY, name="ind")
    m._theta_n = m.addVars(m._ods, vtype=gp.GRB.CONTINUOUS, lb = 0, ub=1, name='theta_n')
    m._profit_extracting_log = m.addVars(m._ods, vtype=gp.GRB.CONTINUOUS, lb = -float('inf'), ub=0, name='extracting_log')
    m._profit_extracting_term = m.addVars(list(it.product(m._ods, m._alternatives)), lb = -float('inf'), ub = 0, vtype=gp.GRB.CONTINUOUS, name='extracting_term')
    
    
    m._F = m.addVars(m._arcs, vtype = gp.GRB.CONTINUOUS, name = 'F')
    m._G = m.addVars(m._arcs, vtype = gp.GRB.CONTINUOUS, name = 'G')
    

    """add constraints"""
    # relationship between theta and z
    for j in m._alternatives:
        for (s, t) in m._ods:
            lhs = gp.quicksum(m._z_vars[(s, t), r, j] for r in m._routes[(s, t)])
            rhs = m._theta_vars[(s, t), j]
            m.addConstr(lhs == rhs, name = "constraint Q (a)")
            
         
    
    # sum of theta's <= 1
    for (s, t) in m._ods:
        lhs = gp.quicksum(m._theta_vars[(s, t), j] for j in m._alternatives)
        rhs = 1 #- eps
        m.addConstr(lhs <= rhs, name = "constraint Q (b)") 
    
      
        
    # load on arc needed  
    for j in m._alternatives:
        for a in m._arcs:
            for (s, t) in m._ods:
                m.addConstr(gp.quicksum([demand[s,t] * m._z_vars[(s, t), r, j] for r in m._routes[(s, t)] if indicator(a, r)]) <= m._y_vars[a, (s, t), j], name = "constraint Q (c)") 

                            
            
    for a in m._arcs:
        lhs = m._f_vars[a]
        rhs = gp.quicksum(m._y_vars[a, (s, t), j] for (s, t) in m._ods for j in m._alternatives)
        m.addConstr(lhs == rhs, name = "equation Q (d)")
    

    # flow conservationå
    for j in m._alternatives:
        for v in m._nodes:
            lhs = gp.quicksum(m._y_vars[(a, b) , (s, t), j] for (a,b) in m._arcs if v == a for (s, t) in m._ods)
            rhs = gp.quicksum(m._y_vars[(a, b) , (s, t), j] for (a,b) in m._arcs if v == b for (s, t) in m._ods)
            m.addConstr(lhs == rhs, name = "constraint Q (e)")




    # bins = 100
    # xs = [1/bins*i for i in range(bins+1)]
    # ys = [math.log(p) if p != 0 else 0 for p in xs]
    # # objective function
    # for j in m._alternatives:
    #     m.addGenConstrPWL(m._theta_n[(s, t)], m._ln_theta_n[(s, t)], xs, ys, "pwl")
    #     for (s, t) in m._ods:
    #         m.addGenConstrPWL(m._theta_vars[(s, t), j], m._ln_theta_vars[(s, t), j], xs, ys, "pwl")

    # # restrict domain for price
    # for (s, t) in m._ods:
    #     for j in m._alternatives:
    #         lhs = m._pi_vars[(s, t), j]
    #         rhs = -1/p_sen * (m._ln_theta_vars[(s, t), j] - m._ln_theta_n[(s, t)] + ASC[(s,t), 2] - T[(s,t), 2] - ASC[(s,t), j] + T[(s,t), j] + l[(s,t), j])
    #         m.addConstr(lhs == rhs, name = "price domain")


    bins = 100
    xs = [1/bins*i for i in range(bins+1)]
    ys = [p*math.log(p) if p != 0 else 0 for p in xs]
    # objective function
    for j in m._alternatives:
        for (s, t) in m._ods:
            m.addGenConstrPWL(m._theta_vars[(s, t), j], m._theta_lntheta_vars[(s, t), j], xs, ys, "pwl")

    # constraints for profit extracting term
    for (s, t) in m._ods:
        m.addConstr(m._theta_n[s,t] == 1 - gp.quicksum(m._theta_vars[(s, t), j] for j in [1]), name ='extract') # add except transit mode
        m.addGenConstrLog(m._theta_n[s,t], m._profit_extracting_log[s,t], name = "ln_profit")
        for j in m._alternatives:
            m.addConstr(m._profit_extracting_term[(s, t), j] == m._theta_vars[(s, t), j] * m._profit_extracting_log[s,t])
            

    for (s, t) in m._ods:
        m.addConstr(m._congest_tt[(s, t), j] == gp.quicksum(gp.quicksum(m._z_vars[(s, t), r, j]*m._F[a] for a in m._arcs if indicator(a, r)) for r in m._routes[(s, t)] for j in m._alternatives))



    for a in m._arcs:
        lhs = m._F[a]
        rhs = m._bpr_func[a](m._f_vars[a])
        m.addConstr(lhs == rhs, name = "F_function")

        lhs = m._G[a]
        rhs = m._bpr_func[a](m._f_vars[a]) * fuel_cost_per_min #value of time + fuel cost
        m.addConstr(lhs == rhs, name = "G_function")
            
    
    obj_util = gp.quicksum(demand[s,t]/p_sen * gp.quicksum(m._theta_vars[(s, t), j] * (- m._ASC[(s, t), j] + m._T[(s, t), j] + ASC[(s, t), 2] - T[(s, t), 2]) for j in m._alternatives) for (s, t) in m._ods) # objective function (A)
    obj_congest = gp.quicksum(demand[s,t]/p_sen * m._congest_tt[(s, t), j] for j in m._alternatives for (s, t) in m._ods)
    obj_entropy = gp.quicksum(demand[s,t]/p_sen * gp.quicksum(m._theta_lntheta_vars[(s, t), j] for j in m._alternatives) for (s, t) in m._ods) # objective function (B)
    obj_profit_extracting = - gp.quicksum(demand[s,t]/p_sen * gp.quicksum(m._profit_extracting_term[(s, t), j] for j in m._alternatives) for (s, t) in m._ods) # objective function (C) 
    obj_oper_cost = gp.quicksum(gp.quicksum((m._G[a] + oper_cost[a]) * m._y_vars[a , (s, t), j] for a in m._arcs) for j in m._alternatives for (s, t) in m._ods) # objective function (D)
    #TODO: add constant operating cost 
    # define objective function
    m.setObjective(obj_util + obj_congest + obj_entropy + obj_oper_cost + obj_profit_extracting)  #

    m.update()
    m.optimize()

    # m.computeIIS() # this helps us to identify constraints that are responsible to make the model infeasible.
    # m.write("model.ilp")

    for v in m.getVars():
        print(f"{v.VarName} = {v.X}")


    if m.Status == 3:
        return None, None
    else:
        return m
    

result = profit_maximization(n_nodes, arcs, OD_route, n_alternative, ods, demand, T, ASC, bpr_func)


# open a file in write mode
with open("{}".format(file_name), "w") as file:
    for v in result.getVars():
        # Write each line to the file
        file.write(f"{v.VarName} = {v.X}\n")
