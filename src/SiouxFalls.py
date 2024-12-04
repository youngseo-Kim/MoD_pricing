import gurobipy as gp
import itertools as it
import networkx as nx
import pandas as pd
import numpy as np
import random
import math
import re
from scipy.sparse import diags
import argparse
from fractions import Fraction

"""
implementation of solving profit maximization problem
author(s): 
yk796@cornell.edu
nd396@cornell.edu
"""
# link to https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument

parser.add_argument('--n_ods', required=True) 
parser.add_argument('--dist', required=True) 
parser.add_argument('--alpha', required=True)
parser.add_argument('--beta', required=True)
parser.add_argument('--n_alter', required=True)
parser.add_argument('--transit_scenario', required=True) 
parser.add_argument('--n_bins', required=True)
parser.add_argument('--oper_cost', required=True)
parser.add_argument('--mip_gap', required=True) 
parser.add_argument('--time_limit', required=True) 
parser.add_argument('--fleet_size', required=False) 
parser.add_argument('--tolling', required=False) 
parser.add_argument('--exo_private', required=True) 

args = parser.parse_args()
n_ods = int(args.n_ods)
dist = int(args.dist)
alpha = float(args.alpha)
beta = float(args.beta)
n_alter = int(args.n_alter)
transit_scenario = int(args.transit_scenario)
n_bins = int(args.n_bins)
oper_cost = float(args.oper_cost)
mip_gap = float(args.mip_gap) 
time_limit = int(args.time_limit) # unit : sec
exo_private = float(args.exo_private)

if args.fleet_size is None:
    fleet_size = None  # Default value if the argument is not provided
else:
    fleet_size = int(args.fleet_size)  
    file_name = "ods_{}_dist_{}_alpha_{}_beta_{}_n_alter_{}_transit_{}_bin_{}_opercost_{}_gap_{}_timelimit_{}_exo_private_{}_fleet_size_{}".format(n_ods, dist, alpha, beta, n_alter, transit_scenario, n_bins, oper_cost, mip_gap, time_limit, exo_private, fleet_size)

if args.tolling is None:
    tolling_roads = []  
else:
    # first trial
    # tolling_roads = [(6, 8), (8, 16), (16, 17), (17, 19), (19, 20), 
    #                  (8, 6), (16, 8), (17, 16), (19, 17), (20, 19)]

    if exo_private == 0.33:
        tolling_roads = [(10, 16), (11, 14), (16, 10), (16, 17), (17, 19)] # exo_private = 0.33        
    elif exo_private == 0: 
        tolling_roads = [(10, 16), (11, 14), (14, 11), (16, 10), (17, 19)] # exo_private = 1 #TODO: need to be edited
   
    # top 10 [(6, 8), (8, 6), (10, 16), (11, 14), (14, 11), (16, 10), (16, 17), (17, 16), (17, 19), (19, 17)]

    toll = int(args.tolling) # USD
    file_name = "ods_{}_dist_{}_alpha_{}_beta_{}_n_alter_{}_transit_{}_bin_{}_opercost_{}_gap_{}_timelimit_{}_exo_private_{}_toll_{}".format(n_ods, dist, alpha, beta, n_alter, transit_scenario, n_bins, oper_cost, mip_gap, time_limit, exo_private, toll)


network_df = pd.read_csv("../data/SiouxFalls/SiouxFalls_net.txt", sep='\t', comment=';')
node_df = pd.read_csv("../data/SiouxFalls/SiouxFalls_node.txt", sep='\t', comment=';')

bkf_df = pd.read_csv('../data/SiouxFalls/SiouxFalls_flow.tntp', delim_whitespace=True)

"""
configuration
"""
if args.fleet_size is None and args.tolling is None:
    file_name = "ods_{}_dist_{}_alpha_{}_beta_{}_n_alter_{}_transit_{}_bin_{}_opercost_{}_gap_{}_timelimit_{}_exo_private_{}".format(n_ods, dist, alpha, beta, n_alter, transit_scenario, n_bins, oper_cost, mip_gap, time_limit, exo_private) # here you can save the solution to the text file, and do analysis with analyze_result.ipynb.
fuel_cost_per_km = oper_cost # USD per km 
vot = 20 #$/hr
p_sen = 1/vot*60 # cost to min
Transit_ASC = -10 # -10, -20, .., -100


if transit_scenario == 1:
    transit_line = []  
elif transit_scenario == 2:
    transit_line = [(4, 11), (11, 14), (14, 23), (23, 24), (5, 9), (9, 10), (10, 15), (15, 22), (22, 21),
                (11, 4), (14, 11), (23, 14), (24, 23), (9, 5), (10, 9), (15, 10), (22, 15), (21, 22)]  
elif transit_scenario == 3:
    transit_line = [(1, 3), (3, 12), (12, 13), (4, 11), (11, 14), (14, 23), (23, 24), (5, 9), (9, 10), (10, 15), (15, 22), (22, 21), (2, 6), (6, 8), (8, 16), (16, 17), (17, 19), (19, 20),
                    (3, 1), (12, 3), (13, 12), (11, 4), (14, 11), (23, 14), (24, 23), (9, 5), (10, 9), (15, 10), (22, 15), (21, 22), (6, 2), (8, 6), (16, 8), (17, 16), (19, 17), (20, 19)]  
elif transit_scenario == 4:
    transit_line = [(2, 6), (6, 8), (8, 16), (16, 17), (17, 19), (19, 20), (20, 21), (21, 24), (24, 13), (12, 11), (11, 10), (10, 16),
                    (6, 2), (8, 6), (16, 8), (17, 16), (19, 17), (20, 19), (21, 20), (24, 21), (13, 24), (11, 12), (10, 11), (16, 10)]  
elif transit_scenario == 5: # additional transit lines
    transit_line = [(2, 6), (6, 8), (8, 16), (16, 17), (17, 19), (19, 20), (20, 21), (21, 24), (24, 13), (12, 11), (11, 10), (10, 16), (24, 23), (23, 14), (14, 11), (11, 10), (10, 16), 
                    (6, 2), (8, 6), (16, 8), (17, 16), (19, 17), (20, 19), (21, 20), (24, 21), (13, 24), (11, 12), (10, 11), (16, 10), (23, 24), (14, 23), (11, 14), (10, 11), (16, 10)]  
    

if dist == 1:
    od_df = pd.read_csv("../data/SiouxFalls/SiouxFalls_od.csv")
elif dist == 2:
    od_df = pd.read_csv("../data/SiouxFalls/SiouxFalls_od_dist2.csv")

if n_ods < len(od_df):
    od_df = od_df.sample(n=n_ods, random_state=42) 


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
def generate_route_sets_link_penalty(graph, source, target, num_routes, penalty_factor=3):
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


n_alternative = n_alter
r_dim = 3

bpr_func = {}
link_capacity = {}

for index, row in network_df.iterrows():
    init_node, term_node = int(row['init_node']), int(row['term_node'])
    free_flow_time, b, capacity, power = row['free_flow_time'], row['b'], row['capacity'], row['power']
    
    bpr_func[(init_node, term_node)] = lambda flow, f=free_flow_time, c=capacity, p=power: f * (1 + alpha * (flow / c)**beta) # 1 should be substitue with power
    link_capacity[(init_node, term_node)] = capacity

nodes = node_df['Node'].to_list()
alternatives = list(range(1, n_alternative+1))
arcs = list(network_df[['init_node', 'term_node']].to_records(index=False))

car_flow = {(int(row['From']),int(row['To'])): float(row['Volume'])*exo_private for index, row in bkf_df.iterrows()} # Note: 1/3 of vehicle is used
demand = {(int(row['O']),int(row['D'])): row['Ton']*1*(1-exo_private) for index, row in od_df.iterrows()} # Note: 2/3 of original demand is used
#TODO: change 100 to 1. This is just for troubleshooting

ods = list(demand.keys())
road_link = [(int(row['init_node']), int(row['term_node']), row['length']) for _, row in network_df.iterrows()]
arc_length = {(int(row['init_node']), int(row['term_node'])): row['length'] for _, row in network_df.iterrows()}

if tolling_roads:
    toll_cost = {(int(row['init_node']), int(row['term_node'])): toll if (int(row['init_node']), int(row['term_node'])) in tolling_roads else 0 for _, row in network_df.iterrows()}
else:
    toll_cost = {(int(row['init_node']), int(row['term_node'])): 0 for _, row in network_df.iterrows()}



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
    route_sets = generate_route_sets_link_penalty(G, i, j, r_dim) 
    OD_route[(i,j)] = route_sets
    

# if j = 2, we need to change the following   

if n_alternative == 2:  
    T = {}
    for (s, t) in ods:
        T[(s, t), 1] = OD_travel_time[(s,t)] # MoD
        T[(s, t), 2] = OD_travel_time[(s,t)] # MoD
        T[(s, t), 3] = OD_transit_time[(s,t)] # Transit

    ASC = {}
    for (s, t) in ods:
        ASC[(s, t), 1] = 0 # MoD
        ASC[(s, t), 2] = 0 # MoD
        ASC[(s, t), 3] = Transit_ASC # Transit

elif n_alternative == 1: 
    T = {}
    for (s, t) in ods:
        T[(s, t), 1] = OD_travel_time[(s,t)] # MoD
        T[(s, t), 2] = OD_transit_time[(s,t)] # Transit

    ASC = {}
    for (s, t) in ods:
        ASC[(s, t), 1] = 0 # MoD
        ASC[(s, t), 2] = Transit_ASC # Transit





# def create_route(od):
#     (o, d) = od
#     # find all possible routes
#     return (o, d)
    

def indicator(arc, route): 
    '''
    To check if an arc is in route
    '''
    # Check if arc is a tuple and has 2 elements
    if not isinstance(arc, tuple) or len(arc) != 2:
        raise ValueError("Arc must be a tuple with 2 elements")

    # Iterate through the route and check each pair
    if route is None:
        return False
    
    for i in range(len(route) - 1):
        if route[i] == arc[0] and route[i+1] == arc[1]:
            return True
    return False


def profit_maximization(n_nodes, arcs, routes, n_alternative, ods, demand, car_flow, T, ASC, arc_length, bpr_func, link_capacity):
    eps = 1e-3

    m = gp.Model()
    m.Params.NonConvex = 2 
    m.Params.DualReductions = 0 # to determine if the model is infeasible or unbounded
    m.setParam("MIPGap", mip_gap) # Set the gap to 5%
    m.Params.LogFile = "../log/"+file_name + ".txt"
    m.setParam('TimeLimit', time_limit)
    # m.Params.OutputFlag = 0
    # m.Params.NonConvex = 2 # because of profit_extracting_log term is concave, we need to tell gurobi
    # that this is not a concave model, although it is. 
    #m.Params.MIPGap = 0


    m._T = T
    m._ASC = ASC
    m._bpr_func = bpr_func
    m._link_capacity = link_capacity
    m._car_flow = car_flow
    m._nodes = list(range(1, n_nodes+1))
    m._alternatives = list(range(1, n_alternative+1))
    m._arcs = [tuple(int(a) for a in arc) for arc in arcs]

    

    m._ods = ods #[(id1+1, id2+1) for id1, o in enumerate(O_demand) for id2, d in enumerate(D_demand) if o>0 or d>0 if id1 != id2]
    #m._routes = routes# {key:create_route(key) for key in m._ods}

    od_dim = len(m._ods)
    j_dim = len(m._alternatives)
    a_dim = len(m._arcs)
    n_dim = len(m._nodes)

    m._routes = routes

    for (o, d) in m._ods:
        # Convert to list of tuples with native Python types
        m._routes[(o, d)] = [tuple(int(r) for r in route) for route in routes[(o, d)]]

    # Note: we define variables as matrix to fasten computation
    m._theta_vars = m.addMVar((j_dim, od_dim), vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name='theta')
    m._y_vars = m.addMVar((j_dim, a_dim, od_dim), vtype=gp.GRB.CONTINUOUS, lb=0, name='y')
    m._z_vars = m.addMVar((j_dim, r_dim, od_dim), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'z')
    m._f_vars = m.addMVar((a_dim), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'f')
    m._total_f_vars = m.addMVar((a_dim), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'total_f')
    m._theta_lntheta_vars = m.addVars(list(it.product(m._ods, m._alternatives)), lb = -float('inf'), ub = 0, vtype=gp.GRB.CONTINUOUS, name='theta_ln_theta') #define theta * ln(theta)
    m._congest_tt = m.addMVar((j_dim, od_dim), lb = 0, vtype=gp.GRB.CONTINUOUS, name='congested_travel_time') 
    m._ind = m.addVars(m._ods, vtype=gp.GRB.BINARY, name="ind")
    m._theta_n = m.addVars(m._ods, vtype=gp.GRB.CONTINUOUS, lb = 0, ub=1, name='theta_n')
    m._profit_extracting_log = m.addVars(m._ods, vtype=gp.GRB.CONTINUOUS, lb = -float('inf'), ub=0, name='extracting_log')
    m._profit_extracting_term = m.addVars(list(it.product(m._ods, m._alternatives)), lb = -float('inf'), ub = 0, vtype=gp.GRB.CONTINUOUS, name='extracting_term')
    
    
    m._F = m.addMVar((a_dim), vtype = gp.GRB.CONTINUOUS, name = 'F')
    m._G = m.addMVar((a_dim), vtype = gp.GRB.CONTINUOUS, name = 'G')

    m.addConstrs((gp.quicksum(m._z_vars[j, r, od] for r in range(r_dim)) == m._theta_vars[j, od] for j in range(j_dim) for od in range(od_dim)), name="constraintQ(a)")
    m.addConstrs((gp.quicksum(m._theta_vars[j, od] for j in range(j_dim)) <= 1 for od in range(od_dim)), name="constraintQ(b)")
    
    indicator_matrix = np.zeros((j_dim, od_dim, a_dim, r_dim), dtype=int)
    for j_ind in range(j_dim):
        for (od_ind, (s,t)) in enumerate(m._ods):
            # Fill in the matrix
            for a_ind, a in enumerate(m._arcs):
                for r_ind, r in enumerate(m._routes[(s,t)]):
                    indicator_matrix[j_ind, od_ind, a_ind, r_ind] = 1 if indicator(a, r) else 0

    no_route_matrix = np.ones((j_dim, r_dim, a_dim), dtype=int) # if the route is not exist set it as 1.
    for j_ind in range(j_dim):
        for (od_ind, (s,t)) in enumerate(m._ods):
            for a_ind, a in enumerate(m._arcs):
                for (r_ind, r) in enumerate(m._routes[(s, t)]):
                    no_route_matrix[j_ind, r_ind, a_ind] = 0 if indicator(a, r) else 1


    # [Option 1]
    # # Caveat: we need to add constraints to ensure 
    # m.addConstrs((demand[s,t] * (indicator_matrix[j_ind, od_ind, :, :] @ m._z_vars[j_ind, :, od_ind]) <= m._y_vars[j_ind, :, od_ind] for j_ind in range(j_dim) for (od_ind, (s,t)) in enumerate(m._ods)), name="constraint Q (c)")
    # m.addConstrs((no_route_matrix[j_ind, r_ind, a_ind] * m._z_vars[j_ind, r_ind, a_ind] == 0 for j_ind in range(j_dim) for r_ind in range(r_dim) for a_ind in range(a_dim)))
    
    # [Option 2]
    m.addConstrs((gp.quicksum([demand[s,t] * m._z_vars[j_ind, r_ind, od_ind] for (r_ind, r) in enumerate(m._routes[(s, t)]) if indicator(a, r)]) <= m._y_vars[j_ind, a_ind, od_ind] for j_ind in range(j_dim) for (a_ind, a) in enumerate(m._arcs) for (od_ind, (s, t)) in enumerate(m._ods)), name = "constraintQ(c)") 
   

    m.addConstrs((m._f_vars[a_ind] == gp.quicksum(m._y_vars[j_ind, a_ind, od_ind] for j_ind in range(j_dim) for od_ind in range(od_dim)) for a_ind in range(a_dim)), name = "equationQ(d)")


    m.addConstrs((m._total_f_vars[a_ind] == m._f_vars[a_ind] + m._car_flow[a] for (a_ind, a) in enumerate(m._arcs)), name = "private vehicles")



    # Create a mapping from node to index
    node_to_index = {node: idx for idx, node in enumerate(m._nodes)}

    # Initialize the node-arc incidence matrix with zeros
    incidence_matrix = np.zeros((n_dim, a_dim), dtype=int)

    # Populate the incidence matrix
    for a_ind, (s, t) in enumerate(m._arcs):
        incidence_matrix[node_to_index[s], a_ind] = -1  # Arc leaves the origin node
        incidence_matrix[node_to_index[t], a_ind] = 1  # Arc enters the destination node

    m.addConstrs((gp.quicksum(incidence_matrix[n_ind, :] @ m._y_vars[j_ind, :, od_ind] for od_ind in range(od_dim)) == 0 for j_ind in range(j_dim) for n_ind, node in enumerate(m._nodes)), name="constraintQ (e)")
    
    # fleet size constraints 
    if fleet_size: # if fleet size is given, add the constraints
        m.addConstr(gp.quicksum(m._y_vars[j_ind, a_ind, od_ind] * arc_length[a] for j_ind in range(j_dim) for (a_ind, a) in enumerate(m._arcs) for od_ind in range(od_dim)) <= fleet_size, name="constraintQ(f)")


            
    bins = n_bins
    xs = [1/bins*i for i in range(bins+1)]
    ys = [p*math.log(p) if p != 0 else 0 for p in xs]
    # objective function
    for (j_ind, j) in enumerate(m._alternatives):
        for (od_ind, (s, t)) in enumerate(m._ods):
            m.addGenConstrPWL(m._theta_vars[j_ind, od_ind], m._theta_lntheta_vars[(s, t), j], xs, ys, "pwl_entropy")


    ys = [math.log(p) if p > 0 else -1e3 for p in xs] # -1e5 represents -infty. Extremely large value cause numerical stability issues

    # constraints for profit extracting term
    for (od_ind, (s, t)) in enumerate(m._ods):
        #m.addGenConstrLog(m._theta_n[s,t], m._profit_extracting_log[s,t], name = "ln_profit")
        m.addGenConstrPWL(m._theta_n[s,t], m._profit_extracting_log[s,t], xs, ys, "ln_profit")
    m.addConstrs((m._theta_n[s,t] == 1 - gp.quicksum(m._theta_vars[j_ind, od_ind] for j_ind in range(j_dim)) for (od_ind, (s, t)) in enumerate(m._ods)), name ='extract')
    m.addConstrs((m._profit_extracting_term[(s, t), j] == m._theta_vars[j_ind, od_ind] * m._profit_extracting_log[s,t] for (od_ind, (s, t)) in enumerate(m._ods) for (j_ind, j) in enumerate(m._alternatives)), name = "profit_extracting")


    m.addConstrs((m._congest_tt[j_ind, od_ind] == indicator_matrix[j_ind, od_ind, :, :] @ m._z_vars[j_ind, :, od_ind] @ m._F for j_ind in range(j_dim) for od_ind in range(od_dim)), name = "congest_tt")

    m.addConstrs((m._G[a_ind] == m._bpr_func[a](0) for (a_ind, a) in enumerate(m._arcs)), name = "G_function")  # G can be set as different function or the same as F
    # Note: Gurobi cannot handle quadratic function as it is. We need piecewise linear approximation as below. 

    # Piecewise linear approximation of BPR function
    # TODO: check if this implementation is correct
    bins = n_bins
    for (a_ind, a) in enumerate(m._arcs):
        xs = [(m._link_capacity[a]) * 1.0 /bins*i for i in range(bins+1)] # the upper bound of volumn to capacity (V/C) ratio is set to 1. 
        ys = [m._bpr_func[a](p) for p in xs] 
        m.addGenConstrPWL(m._total_f_vars[a_ind], m._F[a_ind], xs, ys, "F_function")







    # bins = 12
    # for (a_ind, a) in enumerate(m._arcs):
    #     xs = [(m._link_capacity[a]) * 1.2 /bins*i for i in range(bins+1)] # the upper bound of volumn to capacity (V/C) ratio is set to 1. 
    #     ys = [m._bpr_func[a](p) for p in xs] 
    #     m.addGenConstrPWL(m._f_vars[a_ind], m._F[a_ind], xs, ys, "F_function")



    obj_util = gp.quicksum(demand[s,t]/p_sen * gp.quicksum(m._theta_vars[j_ind, od_ind] * (- m._ASC[(s, t), j] + m._T[(s, t), j] + m._ASC[(s, t), n_alternative+1] - m._T[(s, t), n_alternative+1] - p_sen*2.5) for (j_ind, j) in enumerate(m._alternatives)) for (od_ind, (s, t)) in enumerate(m._ods)) # objective function (A)
    obj_congest = gp.quicksum(demand[s,t]/p_sen * m._congest_tt[j_ind, od_ind] for j_ind in range(j_dim) for (od_ind, (s, t)) in enumerate(m._ods))
    obj_entropy = gp.quicksum(demand[s,t]/p_sen * gp.quicksum(m._theta_lntheta_vars[(s, t), j] for j in m._alternatives) for (s, t) in m._ods) # objective function (B)
    obj_profit_extracting = - gp.quicksum(demand[s,t]/p_sen * gp.quicksum(m._profit_extracting_term[(s, t), j] for j in m._alternatives) for (s, t) in m._ods) # objective function (C) 
    obj_oper_cost = gp.quicksum(gp.quicksum((m._F[a_ind] * fuel_cost_per_km + toll_cost[a]) * m._y_vars[j_ind, a_ind, od_ind] for (a_ind, a) in enumerate(m._arcs)) for (j_ind, j) in enumerate(m._alternatives) for (od_ind, (s, t)) in enumerate(m._ods)) # objective function (D)
    # define objective function
    m.setObjective(obj_util + obj_congest + obj_entropy + obj_oper_cost + obj_profit_extracting) #  # #

    m.update()
    m.write("../model/" + file_name + ".mps")
    m.optimize()

    # m.computeIIS() # this helps us to identify constraints that are responsible to make the model infeasible.
    # m.write("model.ilp")

    for v in m.getVars():
        print(f"{v.VarName} = {v.X}")


    if m.Status == 3:
        return None, None
    else:
        return m
    

open("{}".format("../log/" + file_name + ".txt"), "w") # make sure to overwrite previous file 

result = profit_maximization(n_nodes, arcs, OD_route, n_alternative, ods, demand, car_flow, T, ASC, arc_length, bpr_func, link_capacity)


# open a file in write mode
with open("{}".format("../output/"+file_name + ".txt"), "w") as file:
    for v in result.getVars():
        # Write each line to the file
        file.write(f"{v.VarName} = {v.X}\n")
