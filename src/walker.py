import random

import networkx as nx
import numpy as np


def alias_setup(probs):
    """
    Build alias tables (J, q) for O(1) sampling from a discrete distribution.

    Given a probability vector that sums to 1, returns:
      - q: probability of staying with the drawn bin
      - J: alias index to jump to when rejecting the bin
    """
    n = len(probs)
    q = np.zeros(n, dtype=np.float64)
    J = np.zeros(n, dtype=np.int32)

    smaller = []
    larger = []

    for i, prob in enumerate(probs):
        q[i] = n * prob
        if q[i] < 1.0:
            smaller.append(i)
        else:
            larger.append(i)

    while smaller and larger:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """Draw one sample index in O(1) using precomputed alias tables."""
    n = len(J)
    i = int(np.floor(np.random.rand() * n))
    if np.random.rand() < q[i]:
        return i
    return int(J[i])


class Node2VecWalker:
    def __init__(self, graph: nx.Graph, p: float, q: float):
        self.G = graph
        self.p = p
        self.q = q
        # Filled by preprocess_transition_probs()
        self.alias_nodes = {}  # node -> (J, q) for 1st-order (no previous)
        self.alias_edges = {}  # (t, v) -> (J, q) for 2nd-order biased step

    def get_unnormalized_weights(self, t, v):
        """
        Calculates alpha(t,x) * w(v,x) for all neighbors x of v.
        t: previous node
        v: current node
        """
        unnormalized_probs = []
        neighbors = list(self.G.neighbors(v))

        for x in neighbors:
            # Default weight to 1.0 if the graph is unweighted
            weight = self.G[v][x].get("weight", 1.0)

            if x == t:
                # Distance 0: Returning to the previous node
                unnormalized_probs.append(weight / self.p)
            elif self.G.has_edge(x, t):
                # Distance 1: Next node is connected to previous node (homophily)
                unnormalized_probs.append(weight)
            else:
                # Distance 2: Next node is NOT connected to previous node (exploration)
                unnormalized_probs.append(weight / self.q)

        return neighbors, unnormalized_probs

    def _normalize(self, weights):
        total = float(sum(weights))
        if total == 0.0:
            n = len(weights)
            return [1.0 / n] * n
        return [w / total for w in weights]

    def preprocess_transition_probs(self):
        """
        Precompute alias tables for:
          1) First step from each node (edge-weight / degree distribution)
          2) Second-order transitions for every directed edge (t -> v)
        """
        G = self.G
        alias_nodes = {}
        alias_edges = {}

        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            weights = [G[node][nbr].get("weight", 1.0) for nbr in neighbors]
            probs = self._normalize(weights)
            alias_nodes[node] = alias_setup(probs)

        for t, v in G.edges():
            # Undirected: store both directions
            neighbors_v, weights_tv = self.get_unnormalized_weights(t, v)
            alias_edges[(t, v)] = alias_setup(self._normalize(weights_tv))

            neighbors_t, weights_vt = self.get_unnormalized_weights(v, t)
            alias_edges[(v, t)] = alias_setup(self._normalize(weights_vt))

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

    def node2vec_walk(self, walk_length, start_node):
        """Generate one biased random walk of length walk_length starting at start_node."""
        G = self.G
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            neighbors = list(G.neighbors(cur))
            if not neighbors:
                break

            if len(walk) == 1:
                # No previous node: sample with 1st-order alias table
                J, q = self.alias_nodes[cur]
                next_idx = alias_draw(J, q)
                walk.append(neighbors[next_idx])
            else:
                prev = walk[-2]
                J, q = self.alias_edges[(prev, cur)]
                # Neighbor order must match get_unnormalized_weights(prev, cur)
                cur_neighbors = list(G.neighbors(cur))
                next_idx = alias_draw(J, q)
                walk.append(cur_neighbors[next_idx])

        return walk

    def simulate_walks(self, num_walks, walk_length, seed=None):
        """
        Generate num_walks walks of length walk_length starting from every node.

        Returns a list of walks (each walk is a list of node ids).
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if not self.alias_nodes or not self.alias_edges:
            self.preprocess_transition_probs()

        nodes = list(self.G.nodes())
        walks = []

        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks
