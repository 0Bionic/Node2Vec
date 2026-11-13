"""Phase 1 sanity check: alias sampling + walk generation on Les Misérables."""

import networkx as nx

from src.walker import Node2VecWalker


def main():
    G = nx.les_miserables_graph()
    print(f"Graph: Les Misérables | nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    p, q = 1.0, 1.0
    num_walks = 10
    walk_length = 20

    walker = Node2VecWalker(G, p=p, q=q)
    walker.preprocess_transition_probs()
    print(
        f"Alias tables ready | nodes={len(walker.alias_nodes)} "
        f"directed_edges={len(walker.alias_edges)}"
    )

    walks = walker.simulate_walks(num_walks=num_walks, walk_length=walk_length, seed=42)
    expected = num_walks * G.number_of_nodes()
    print(f"Generated walks: {len(walks)} (expected {expected})")

    # Length check (isolates would be shorter; Les Mis has none)
    lengths = [len(w) for w in walks]
    print(f"Walk length: min={min(lengths)} max={max(lengths)} target={walk_length}")

    # Every walk should start at a valid node; sample a few
    print("\nSample walks:")
    for i, walk in enumerate(walks[:5]):
        print(f"  [{i}] {' -> '.join(map(str, walk[:8]))} ... (len={len(walk)})")

    # Basic validity: every consecutive pair is an edge
    bad = 0
    for walk in walks:
        for a, b in zip(walk, walk[1:]):
            if not G.has_edge(a, b):
                bad += 1
    print(f"\nInvalid consecutive steps: {bad} (should be 0)")


if __name__ == "__main__":
    main()
