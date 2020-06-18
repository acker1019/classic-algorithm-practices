# -*- coding: utf-8 -*-

import numpy as np
from time import process_time_ns
import random as rd


'''
Two versions of bi-connected components searching algorithm are implemented in the file,
which are adjacency matrix and m-way tree with backtracking line.

It also finds articulation points when searching.
'''

class NumberIterator:
    '''
    Generate number iterately.
    Its object will be passed between recursive functions
    '''
    
    def __iter__(self, val=1):
        self.val = val
        return self

    def __next__(self):
        v = self.val
        self.val += 1
        return v

class Node():
    '''
    The node element of dfs-spanning-backtracking-tree
    '''
    
    def __init__(self, parent=None, data=None):
        self.parent = parent
        self.data = data
        self.children = []
        self.backtrack = []
        self.DFN = None
        self.L = None
        
    
    def __str__(self):
        return "Node({}; parent={}, len(children)={}, len(backtrack)={}, DFN={}, L={})".format(
            self.data,
            None if self.parent is None else self.parent.data,
            len(self.children),
            len(self.backtrack),
            self.DFN,
            self.L)
    
    
    def __repr__(self):
        return self.__str__()


def dfs_spanning_tree(nodes, graph, visited, tree_graph):
    '''
    generate dfs-spanning backtracking tree from a given graph which is represented by adjacency matrix
    '''
    root_id = 0
    root = nodes[root_id]
    num_giver = iter(NumberIterator())
    walk_toBuild(nodes, graph, visited, root, None, root_id, tree_graph, num_giver)
    
    return root
    

def walk_toBuild(nodes, graph, visited, node, parent_j, j, tree_graph, num_giver):
    '''
    The recursive part of dfs_spanning_tree()
    '''
    node.DFN = next(num_giver)
    visited[j] = 1
    edges = graph[j]
    for i in range(len(edges)):
        if edges[i]==1:
            if visited[i]==0:
                # build tree structure
                child = nodes[i]
                child.parent = node
                node.children.append(child)
                
                # build tree graph
                tree_graph[j][i] = 1
                tree_graph[i][j] = 1
                
                walk_toBuild(nodes, graph, visited, child, j, i, tree_graph, num_giver)
                
            elif visited[i]==1 and parent_j!=i and i < j:
                # build tree structure
                if nodes[i].DFN > node.DFN:
                    offspring = nodes[i]
                    ancestor = node
                else:
                    offspring = node
                    ancestor = nodes[i]
                    
                offspring.backtrack.append(ancestor)
                
                # build tree graph
                tree_graph[j][i] = -1
                tree_graph[i][j] = -1


def draw(node, graph):
    '''
    Transform the generated dfs-spanning backtracking tree to check the correctness
    '''
    for a in node.backtrack:
        if graph[node.data][a.data] == 0:
            graph[node.data][a.data] = -1
        else:
            graph[node.data][a.data] = 3
            
        if graph[a.data][node.data] == 0:
            graph[a.data][node.data] = -1
        else:
            graph[a.data][node.data] = 3
    
    for c in node.children:
        if graph[node.data][c.data] == 0:
            graph[node.data][c.data] = 1
        else:
            graph[node.data][c.data] = 2
            
        if graph[c.data][node.data] == 0:
            graph[c.data][node.data] = 1
        else:
            graph[c.data][node.data] = 2

        
        draw(c, graph)


def search_BiConnected(graph, j, DFN, L, edge_stack, joints, subnets):
    '''
    Version of adjacency matrix
    
    If the root gets branches more than two, the root is an articulation point, because:
    
    DFS make the nodes concentrate at the left-bottom of the tree,
    which means all branches of the root never connect to each other.
    '''
    
    num_giver = iter(NumberIterator())
    num = next(num_giver)
    root_id = 0
    DFN[root_id] = num
    L[root_id] = num
    
    edges = graph[root_id]
    for i, v in enumerate(edges):
        if v == 1 and DFN[i] == 0:
            edge_stack.append((root_id, i))
            walk_toSearch(graph, i, root_id, DFN, L, num_giver, edge_stack, joints, subnets)
            
            subnet = []
            while len(edge_stack) > 0:
                x, y = edge_stack.pop()
                subnet.append((x, y))
                if (x==i and y==root_id) or (x==root_id and y==i):
                    break
            if len(subnet) > 0:
                subnets.append(subnet)
    
    path_count = 0
    edges = graph[root_id]
    for i, v in enumerate(edges):
        if v == 1 and DFN[i] == 0:
            path_count += 1
    
    if path_count > 0:
        joints.append(root_id)
            
    

def walk_toSearch(graph, j, src_j, DFN, L, num_giver, edge_stack, joints, subnets):
    '''
    Recursion part of search_BiConnected()
    '''
    num = next(num_giver)
    DFN[j] = num
    L[j] = num
    edges = graph[j]
    isArticulation = False
    for i, v in enumerate(edges):
        if v == 1:
            if DFN[i] < DFN[j] and i != src_j:
                edge_stack.append((j, i))
            if DFN[i] == 0:
                walk_toSearch(graph, i, j, DFN, L, num_giver, edge_stack, joints, subnets)
                L[j] = min([L[j], L[i]])
                
                if L[i] >= DFN[j]:
                    isArticulation = True
                    subnet = []
                    while len(edge_stack) > 0:
                        x, y = edge_stack.pop()
                        subnet.append((x, y))
                        if (x==i and y==j) or (x==j and y==i):
                            break
                    subnets.append(subnet)
                
            else:
                if i != src_j:
                    L[j] = min([L[j], DFN[i]])
                    
    if isArticulation:
        joints.append(j)
        

def search_BiConnected_tree(root, edge_stack, joints, subnets):
    '''
    The version of depth-first-search backtracking tree
    '''
    
    root.L = root.DFN
    
    for child in root.children:
        edge_stack.append((root.data, child.data))
        walk_toSearch_tree(child, edge_stack, joints, subnets)
        
        subnet = []
        while len(edge_stack) > 0:
            x, y = edge_stack.pop()
            subnet.append((x, y))
            if (x==child.data and y==root.data) or (x==root.data and y==child.data):
                break
        if len(subnet) > 0:
            subnets.append(subnet)
    
    if len(root.children) > 1:
        joints.append(root.data)



def walk_toSearch_tree(node, edge_stack, joints, subnets):
    '''
    Recursion part of search_BiConnected_tree()
    '''
    node.L = node.DFN
    isArticulation = False
    for child in node.children:
        edge_stack.append((node.data, child.data))
        walk_toSearch_tree(child, edge_stack, joints, subnets)
        node.L = min([node.L, child.L])
        
        if child.L >= node.DFN:
            isArticulation = True
            subnet = []
            while len(edge_stack) > 0:
                x, y = edge_stack.pop()
                subnet.append((x, y))
                if (x==child.data and y==node.data) or (x==node.data and y==child.data):
                    break
            subnets.append(subnet)
    
    if isArticulation:
        joints.append(node.data)
    
    for ancestor in node.backtrack:
        edge_stack.append((node.data, ancestor.data))
        node.L = min([node.L, ancestor.DFN])
        

def walk_forValue_tree(node, DFN, L):
    '''
    Gethering the values from the tree nodes into arrays
    '''
    DFN[node.data] = node.DFN
    L[node.data] = node.L
    for child in node.children:
        walk_forValue_tree(child, DFN, L)


def genAdjacencyMatrix(node_size, edge_size_rate=0.2):
    '''
    Generate a adjacency matrix randomly
    
    node_size: number of node in the generated graph
    edge_size_rate:
        Value domain is [0.0, 1.0].
        No edge is in the graph when 0.0.
        If it's 1.0, a full connected graph will be generated.
    '''
    edge_domain_size = int(node_size * (node_size - 1) / 2)
    edge_size = int(edge_domain_size * edge_size_rate)
    matrix = np.zeros((node_size, node_size), dtype=int)

    edge_scale = edge_domain_size - node_size + 1

    distributor = np.zeros(edge_scale, dtype=int)
    
    
    for i in range(edge_size):
        pos = rd.randrange(0, edge_scale)
        while distributor[pos] == 1:
            pos = (pos + 1) % edge_scale
        distributor[pos] = 1
    
    for j in range(node_size-1):
        pos = 1 + j + rd.randrange(0, node_size - j - 1)
        matrix[j][pos] = 1
        matrix[pos][j] = 1

    k = 0
    for j in range(node_size-1):
        for i in range(1 + j, node_size):
            if matrix[j][i] == 0:
                v = distributor[k] 
                k += 1
                matrix[j][i] = v
                matrix[i][j] = v
    
    return matrix


def readMatrix(fpath):
    '''
    read an exsisted matrix from file
    '''
    str_matrix = []
    with open(fpath, 'r') as file:
        for line in file:
            str_matrix.append(line)
    
    matrix = [str_matrix[0][3:-2].split(' ')]
    
    for line in str_matrix[1:-1]:
        matrix.append(line[2:-2].split(' '))
    
    matrix.append(str_matrix[-1][2:-2].split(' '))
    
    matrix = [[int(s) for s in line] for line in matrix]
    
    graph = np.array(matrix, dtype=int)
    
    return len(graph), graph



' generate graph'


randgen = False # generate a graph randomly?
if randgen:
    num_point = 20
    graph = genAdjacencyMatrix(
        node_size=num_point,
        edge_size_rate=0.03)
else:
    # read a graph from file
    num_point, graph = readMatrix('matrix_in.txt')

print('\ngraph: (It can be drawn in https://graphonline.ru/en/create_graph_by_matrix)\n', graph)



' build dfs-spanning tree'

nodes = []
for i in range(num_point):
    nodes.append(Node(data=i))

tree_graph = np.zeros((num_point, num_point), dtype=int)
visited = np.zeros(num_point, dtype=int)

start_time = process_time_ns()
tree = dfs_spanning_tree(nodes, graph, visited, tree_graph)
time_usage = process_time_ns() - start_time

print('\ntree_graph\n', tree_graph)

print('\ntime_usage:{}\n'.format(time_usage))



' draw tree to check'

tree_draw_exam = np.zeros((num_point, num_point), dtype=int)
draw(tree, tree_draw_exam)
print('\ntree_draw_exam:\n', tree_draw_exam)

print('\ntree built correctly? ', end='')
if np.allclose(tree_graph, tree_draw_exam):
    print( (tree_graph == tree_draw_exam).all() )



' bi-connected components searching with only graph as input (adjacency matrix)'

print("\n\nbi-connected components searching with only graph as input (adjacency matrix) [O(V^2)]")

DFN = np.zeros(num_point, dtype=int) # depth-first number
L = np.zeros(num_point, dtype=int) # DFN(self -> offsprings -> oldest_ancestor)

edge_stack = []
joints = []
subnets = []

start_time = process_time_ns()
search_BiConnected(graph, 0, DFN, L, edge_stack, joints, subnets)
time_usage = process_time_ns() - start_time

print('\nDFN:\n', DFN)
print('\nL:\n', L)
print('\njoints:\n', joints)
print('\nsubnets:\n', subnets)

print('\nedge_stack:\n', edge_stack)

print('\ntime_usage:{}\n'.format(time_usage))



' bi-connected components searching with DFS-backtracking-tree'

print("\n\nbi-connected components searching with DFS-backtracking-tree [O(E+V)]\n")

edge_stack = []
joints = []
subnets = []

start_time = process_time_ns()
search_BiConnected_tree(tree, edge_stack, joints, subnets)
time_usage = process_time_ns() - start_time

DFN = np.zeros(num_point, dtype=int) # depth-first number
L = np.zeros(num_point, dtype=int) # DFN(self -> offsprings -> oldest_ancestor)

walk_forValue_tree(tree, DFN, L)

print('\nDFN:\n', DFN)
print('\nL:\n', L)
print('\njoints:\n', joints)
print('\nsubnets:\n', subnets)

print('\nedge_stack:\n', edge_stack)

print('\ntime_usage:{}\n'.format(time_usage))





