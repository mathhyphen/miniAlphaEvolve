"""
AlphaEvolve Example: Minimum Spanning Tree (MST) Algorithm Evolution

This example demonstrates how AlphaEvolve can optimize MST algorithms
like Kruskal's or Prim's algorithm.
"""

import logging
import time
from alphaevolve import (
    EvolutionConfig,
    MutationEngine,
    UnitTestEvaluator,
    Evolution,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Seed code: Kruskal's algorithm with naive sorting (no path compression)
SEED_CODE = """
def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)
    if root_x != root_y:
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        else:
            parent[root_y] = root_x
            if rank[root_x] == rank[root_y]:
                rank[root_x] += 1
        return True
    return False

def mst_kruskal(vertices, edges):
    '''
    Find minimum spanning tree using Kruskal's algorithm.

    Args:
        vertices: Number of vertices
        edges: List of (u, v, weight) tuples

    Returns:
        Total weight of MST
    '''
    # Sort edges by weight using bubble sort (inefficient)
    sorted_edges = list(edges)
    n = len(sorted_edges)
    for i in range(n):
        for j in range(0, n - i - 1):
            if sorted_edges[j][2] > sorted_edges[j + 1][2]:
                sorted_edges[j], sorted_edges[j + 1] = sorted_edges[j + 1], sorted_edges[j]

    parent = list(range(vertices))
    rank = [0] * vertices

    mst_weight = 0
    edges_count = 0

    for edge in sorted_edges:
        u, v, w = edge
        if union(parent, rank, u, v):
            mst_weight += w
            edges_count += 1
            if edges_count == vertices - 1:
                break

    return mst_weight
"""


def create_test_cases():
    """Create test cases for MST algorithm."""
    return [
        # Simple triangle: 3 vertices, 3 edges
        ((3, [(0, 1, 1), (1, 2, 2), (0, 2, 3)]), 3),  # MST: 1 + 2 = 3

        # Square: 4 vertices in a line
        ((4, [(0, 1, 1), (1, 2, 2), (2, 3, 3)]), 6),  # MST: 1 + 2 + 3 = 6

        # Single edge
        ((2, [(0, 1, 10)]), 10),

        # Square with diagonal - choose cheaper path
        ((4, [(0, 1, 1), (1, 2, 2), (2, 3, 3), (0, 2, 10)]), 6),  # MST: 1 + 2 + 3 = 6

        # Complete graph K4
        ((4, [(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 2, 4), (1, 3, 5), (2, 3, 6)]), 6),
    ]


def run_evolution():
    """Run MST algorithm evolution."""
    logger.info("=" * 60)
    logger.info("AlphaEvolve: MST Algorithm Evolution")
    logger.info("=" * 60)

    # Create test cases
    test_cases = create_test_cases()
    logger.info(f"Test cases: {len(test_cases)}")

    # Configure evolution
    config = EvolutionConfig(
        population_size=10,
        max_generations=5,
        elitism_count=2,
        mutation_rate=0.7,
        seed=42,
    )

    # Create evaluator
    evaluator = UnitTestEvaluator(
        test_cases=test_cases,
        function_name="mst_kruskal",
        partial_credit=True,
    )

    # Create mutation engine
    engine = MutationEngine()

    # Create evolution instance
    evolution = Evolution(
        config=config,
        mutation_engine=engine,
        evaluator=evaluator,
        seed_code=SEED_CODE,
    )

    # Run evolution
    logger.info("Starting evolution...")
    logger.info(f"Population size: {config.population_size}")
    logger.info(f"Max generations: {config.max_generations}")
    logger.info(f"Elitism count: {config.elitism_count}")

    best = evolution.evolve()

    # Report results
    logger.info("=" * 60)
    logger.info("Evolution Complete!")
    logger.info("=" * 60)
    logger.info(f"Best fitness: {best.fitness:.2f}")
    logger.info(f"Best code:\n{best.code}")

    # Show evolution history
    history = evolution.get_history()
    if history:
        logger.info("\nEvolution History:")
        logger.info("-" * 40)
        for gen in history:
            logger.info(
                f"Gen {gen['generation']}: "
                f"Best={gen['best_fitness']:.2f}, "
                f"Avg={gen['avg_fitness']:.2f}, "
                f"Pop={gen['population_size']}"
            )

    return best


def demo_prims_algorithm():
    """Demonstrate Prim's algorithm seed."""

    prims_seed = """
def mst_prim(vertices, edges):
    '''
    Find minimum spanning tree using Prim's algorithm.

    Args:
        vertices: Number of vertices
        edges: List of (u, v, weight) tuples

    Returns:
        Total weight of MST
    '''
    # Build adjacency list
    adj = [[] for _ in range(vertices)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    # Prim's algorithm with simple array (no heap)
    in_mst = [False] * vertices
    min_weight = [float('inf')] * vertices
    min_weight[0] = 0

    mst_weight = 0

    for _ in range(vertices):
        # Find minimum vertex not in MST (linear scan)
        u = -1
        for i in range(vertices):
            if not in_mst[i] and (u == -1 or min_weight[i] < min_weight[u]):
                u = i

        if u == -1 or min_weight[u] == float('inf'):
            break

        in_mst[u] = True
        mst_weight += min_weight[u]

        # Update neighbors
        for v, w in adj[u]:
            if not in_mst[v] and w < min_weight[v]:
                min_weight[v] = w

    return mst_weight
"""

    logger.info("\n" + "=" * 60)
    logger.info("Prim's Algorithm Seed")
    logger.info("=" * 60)
    logger.info(prims_seed)

    return prims_seed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AlphaEvolve MST Example")
    parser.add_argument(
        "--demo-prim",
        action="store_true",
        help="Show Prim's algorithm seed code"
    )

    args = parser.parse_args()

    if args.demo_prim:
        demo_prims_algorithm()
    else:
        run_evolution()
