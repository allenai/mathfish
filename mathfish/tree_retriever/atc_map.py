"""
Functions for creating the ATC coherence map 
as a graph and for calculating distances between nodes.
"""
from collections import defaultdict
import json
import networkx as nx

class ATCMap(): 
    def __init__(self, standards_path: str):
        self.connections = defaultdict(dict) # { origin standard : {relation : [destination standards] } 
        with open(standards_path, 'r') as infile:
            for line in infile:
                d = json.loads(line)
                if d['level'] != 'Standard': continue
                if d['connections']: 
                    self.connections[d['id']] = d['connections']
        self.undir_graph = nx.Graph()
        self.dir_graph = nx.DiGraph()

    def create_undirected_graph(self):
        for vertex in self.connections: 
            if not self.undir_graph.has_node(vertex): 
                self.undir_graph.add_node(vertex)
            for reltype in self.connections[vertex]: 
                neighbors = self.connections[vertex][reltype]
                for n in neighbors: 
                    self.undir_graph.add_edge(vertex, n)

    def get_distance(self, start, end, directed=False): 
        if directed: 
            raise NotImplementedError
        else: 
            return nx.shortest_path_length(self.undir_graph, source=start, target=end)

    def create_directed_graph(self): 
        raise NotImplementedError