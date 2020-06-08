# -*- coding: utf-8 -*-

import numpy as np
from time import process_time_ns

class NumberIterator:
  def __iter__(self, val=1):
    self.val = val
    return self

  def __next__(self):
    v = self.val
    self.val += 1
    return v

class Node():
    
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
    root_id = 0
    root = nodes[root_id]
    walk_toBuild(nodes, graph, visited, root, None, root_id, tree_graph)
    
    return root
    

def walk_toBuild(nodes, graph, visited, node, parent_j, j, tree_graph):
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
                
                walk_toBuild(nodes, graph, visited, child, j, i, tree_graph)
                
            elif visited[i]==1 and parent_j!=i and i < j:
                # build tree structure
                ancestor = nodes[i]
                node.backtrack.append(ancestor)
                
                # build tree graph
                tree_graph[j][i] = -1
                tree_graph[i][j] = -1


def draw(node, graph):
    for a in node.backtrack:
        graph[node.data][a.data] = -1
        graph[a.data][node.data] = -1
    
    for c in node.children:
        graph[node.data][c.data] = 1
        graph[c.data][node.data] = 1

        
        draw(c, graph)


def search_BiConnected(graph, j, DFN, L, edge_stack, joints, subnets):
    '''
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
                subnet.insert(0, (x, y))
                if (x==i and y==root_id) or (x==root_id and y==i):
                    break
            if len(subnet) > 0:
                subnets.insert(0, subnet)
    
    path_count = 0
    edges = graph[root_id]
    for i, v in enumerate(edges):
        if v == 1 and DFN[i] == 0:
            path_count += 1
    
    if path_count > 0:
        joints.insert(0, root_id)
            
    

def walk_toSearch(graph, j, src_j, DFN, L, num_giver, edge_stack, joints, subnets):
    num = next(num_giver)
    DFN[j] = num
    L[j] = num
    edges = graph[j]
    for i, v in enumerate(edges):
        if v == 1 and i != src_j:
            if DFN[i] < DFN[j]:
                edge_stack.append((j, i))
            if DFN[i] == 0:
                walk_toSearch(graph, i, j, DFN, L, num_giver, edge_stack, joints, subnets)
                L[j] = min([L[j], L[i]])
                
                if L[i] >= DFN[j]:
                    joints.insert(0, j)
                    subnet = []
                    while len(edge_stack) > 0:
                        x, y = edge_stack.pop()
                        subnet.insert(0, (x, y))
                        if (x==i and y==j) or (x==j and y==i):
                            break
                    subnets.insert(0, subnet)
                
            else:
                L[j] = min([L[j], DFN[i]])
        

def search_BiConnected_tree(root, edge_stack, joints, subnets):
    '''
    The version of depth-first-search backtracking tree
    '''
    
    num_giver = iter(NumberIterator())
    num = next(num_giver)
    root.DFN = num
    root.L = num
    print(root)
    
    for child in root.children:
        edge_stack.append((root.data, child.data))
        walk_toSearch_tree(child, num_giver, edge_stack, joints, subnets)
        
        subnet = []
        while len(edge_stack) > 0:
            x, y = edge_stack.pop()
            subnet.insert(0, (x, y))
            if (x==child.data and y==root.data) or (x==root.data and y==child.data):
                break
        if len(subnet) > 0:
            subnets.insert(0, subnet)
    
    if len(root.children) > 1:
        joints.insert(0, root.data)



def walk_toSearch_tree(node, num_giver, edge_stack, joints, subnets):
    num = next(num_giver)
    node.DFN = num
    node.L = num
    print(node)
    for child in node.children:
        edge_stack.append((node.data, child.data))
        walk_toSearch_tree(child, num_giver, edge_stack, joints, subnets)
        node.L = min([node.L, child.L])
        
        if child.L >= node.DFN:
            joints.insert(0, node.data)
            subnet = []
            while len(edge_stack) > 0:
                x, y = edge_stack.pop()
                subnet.insert(0, (x, y))
                if (x==child.data and y==node.data) or (x==node.data and y==child.data):
                    break
            subnets.insert(0, subnet)
    
    for ancestor in node.backtrack:
        edge_stack.append((node.data, ancestor.data))
        node.L = min([node.L, ancestor.DFN])
        

def walk_forValue_tree(node, DFN, L):
    #print(node, node.data, node.DFN, node.L)
    DFN[node.data] = node.DFN
    L[node.data] = node.L
    for child in node.children:
        walk_forValue_tree(child, DFN, L)


' generate graph'
        
num_point = 6   #input()
num_edge = 6    #input()

graph = np.zeros((num_point, num_point), dtype=int)
visited = np.zeros(num_point, dtype=int)

in_ = [(1, 2), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5), (5, 6)]

for a, b in in_:
    a -= 1
    b -= 1
    graph[a][b] = 1
    graph[b][a] = 1

print('\ngraph:\n', graph)



' build dfs-spanning tree'

nodes = []
for i in range(num_point):
    nodes.append(Node(data=i))

tree_graph = np.zeros((num_point, num_point), dtype=int)

start_time = process_time_ns()
tree = dfs_spanning_tree(nodes, graph, visited, tree_graph)
time_usage = process_time_ns() - start_time

print('\ntree_graph\n', tree_graph)

print('\ntime_usage:{}\n'.format(time_usage))



' draw tree to check'

tree_draw_exam = np.zeros((num_point, num_point), dtype=int)
draw(tree, tree_draw_exam)
print('\ntree_draw_exam:\n', tree_draw_exam)



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

joints = [v+1 for v in joints]
subnets = [[(x+1, y+1) for x, y in subnet] for subnet in subnets]
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
joints = [v+1 for v in joints]
subnets = [[(x+1, y+1) for x, y in subnet] for subnet in subnets]

print('\nDFN:\n', DFN)
print('\nL:\n', L)
print('\njoints:\n', joints)
print('\nsubnets:\n', subnets)

print('\nedge_stack:\n', edge_stack)

print('\ntime_usage:{}\n'.format(time_usage))





