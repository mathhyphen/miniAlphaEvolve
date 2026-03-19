"""Evolved Steiner Tree Solution - Iteration 2.

This code was evolved from mst_only_baseline.py using AlphaEvolve (experiment b779a6f).
It achieved significant improvements over iteration 1:

Performance:
- equilateral_triangle_3: PASS (error=0.000000)
- square_4: PASS (error=0.000000) <- Fixed from iter1!
- rectangle_3x1_4: PASS (error=0.000010)
- regular_pentagon_5: FAIL (error=0.084690)
- regular_hexagon_6: FAIL (error=0.133975)
- collinear_4: PASS (error=0.000000)

Key evolved features:
1. Weiszfeld-like iteration for Fermat point
2. MST-based topology with Kruskal's algorithm
3. Local gradient-based refinement of Steiner points
4. Proper edge tracking for refinement

Next evolution goals:
1. Improve 5-6 point regular polygon accuracy
2. Better handling of N>4 Steiner point topologies
3. Discover full Steiner tree structure (not just MST heuristic)
"""

import math
from typing import List, Tuple, Set, Dict
import logging
from heapq import heappush, heappop

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


def distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def mst_length(points: List[Point]) -> float:
    """Calculate Minimum Spanning Tree length using Prim's algorithm."""
    n = len(points)
    if n < 2:
        return 0.0
    if n == 2:
        return distance(points[0], points[1])

    # Optimized Prim's with heap
    visited = [False] * n
    min_heap = [(0.0, 0)]  # (distance, vertex)
    total_length = 0.0
    visited_count = 0
    points_local = points  # Local reference for speed

    while min_heap and visited_count < n:
        d, u = heappop(min_heap)
        if visited[u]:
            continue

        visited[u] = True
        visited_count += 1
        total_length += d
        pu = points_local[u]

        for v in range(n):
            if not visited[v]:
                pv = points_local[v]
                dx = pu[0] - pv[0]
                dy = pu[1] - pv[1]
                heappush(min_heap, (math.sqrt(dx*dx + dy*dy), v))

    return total_length


def fermat_point(p1: Point, p2: Point, p3: Point, eps: float = 1e-10) -> Point:
    """Calculate Fermat point of triangle (minimizes sum of distances to vertices)."""
    # Check for degenerate cases using law of cosines
    dx12 = p1[0] - p2[0]
    dy12 = p1[1] - p2[1]
    d12_sq = dx12*dx12 + dy12*dy12
    d12 = math.sqrt(d12_sq)

    dx23 = p2[0] - p3[0]
    dy23 = p2[1] - p3[1]
    d23_sq = dx23*dx23 + dy23*dy23
    d23 = math.sqrt(d23_sq)

    dx31 = p3[0] - p1[0]
    dy31 = p3[1] - p1[1]
    d31_sq = dx31*dx31 + dy31*dy31
    d31 = math.sqrt(d31_sq)

    # Check if any angle >= 120° using law of cosines
    if d12 > eps and d31 > eps:
        cos1 = (d12_sq + d31_sq - d23_sq) / (2.0 * d12 * d31)
        if cos1 <= -0.5:
            return p1

    if d12 > eps and d23 > eps:
        cos2 = (d12_sq + d23_sq - d31_sq) / (2.0 * d12 * d23)
        if cos2 <= -0.5:
            return p2

    if d23 > eps and d31 > eps:
        cos3 = (d23_sq + d31_sq - d12_sq) / (2.0 * d23 * d31)
        if cos3 <= -0.5:
            return p3

    # All angles < 120°, use Weiszfeld-like iteration
    x = (p1[0] + p2[0] + p3[0]) / 3.0
    y = (p1[1] + p2[1] + p3[1]) / 3.0

    for _ in range(100):
        dx1 = x - p1[0]
        dy1 = y - p1[1]
        d1 = math.sqrt(dx1*dx1 + dy1*dy1)

        dx2 = x - p2[0]
        dy2 = y - p2[1]
        d2 = math.sqrt(dx2*dx2 + dy2*dy2)

        dx3 = x - p3[0]
        dy3 = y - p3[1]
        d3 = math.sqrt(dx3*dx3 + dy3*dy3)

        if d1 < eps or d2 < eps or d3 < eps:
            break

        w1, w2, w3 = 1.0/d1, 1.0/d2, 1.0/d3
        sum_w = w1 + w2 + w3

        new_x = (w1 * p1[0] + w2 * p2[0] + w3 * p3[0]) / sum_w
        new_y = (w1 * p1[1] + w2 * p2[1] + w3 * p3[1]) / sum_w

        if abs(new_x - x) < eps and abs(new_y - y) < eps:
            break

        x, y = new_x, new_y

    return (x, y)


def steiner_tree_3terminals(p1: Point, p2: Point, p3: Point) -> Tuple[float, List[Point]]:
    """Compute optimal Steiner tree for 3 terminals."""
    fp = fermat_point(p1, p2, p3)

    total_len = distance(fp, p1) + distance(fp, p2) + distance(fp, p3)

    # Check if Fermat point actually helps
    d12 = distance(p1, p2)
    d23 = distance(p2, p3)
    d13 = distance(p1, p3)
    mst_len = min(d12 + d23, d12 + d13, d23 + d13)

    if total_len >= mst_len - 1e-9:
        return (mst_len, [])

    return (total_len, [fp])


def mst_edges(points: List[Point]) -> List[Tuple[int, int, float]]:
    """Get MST edges as list of (i, j, weight)."""
    n = len(points)
    if n < 2:
        return []

    # Kruskal's algorithm with optimized edge generation
    edges = []
    pts = points  # Local reference
    for i in range(n):
        pi = pts[i]
        for j in range(i+1, n):
            pj = pts[j]
            dx = pi[0] - pj[0]
            dy = pi[1] - pj[1]
            edges.append((i, j, math.sqrt(dx*dx + dy*dy)))

    edges.sort(key=lambda x: x[2])

    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # Path compression
            x = parent[x]
        return x

    mst = []
    for i, j, w in edges:
        pi = find(i)
        pj = find(j)
        if pi != pj:
            parent[pi] = pj
            mst.append((i, j, w))
            if len(mst) == n - 1:
                break

    return mst


def local_refinement(points: List[Point], steiner_points: List[Point],
                     edges: List[Tuple[int, int]], max_iter: int = 100) -> List[Point]:
    """Refine Steiner point positions using gradient descent."""
    n_terminals = len(points)
    all_points = list(points) + list(steiner_points)
    n_all = len(all_points)

    # Build adjacency list
    adj = [[] for _ in range(n_all)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    steiner_points = list(steiner_points)  # Make mutable copy

    for _ in range(max_iter):
        max_move = 0.0
        new_steiner = []

        for idx, sp in enumerate(steiner_points):
            i = n_terminals + idx
            neighbors = adj[i]
            nbr_len = len(neighbors)
            if nbr_len < 2:
                new_steiner.append(sp)
                continue

            # Compute gradient: sum of unit vectors to neighbors
            gx, gy = 0.0, 0.0
            spx, spy = sp[0], sp[1]
            for j in neighbors:
                pj = all_points[j]
                dx = pj[0] - spx
                dy = pj[1] - spy
                d = math.sqrt(dx*dx + dy*dy)
                if d > 1e-10:
                    inv_d = 1.0 / d
                    gx += dx * inv_d
                    gy += dy * inv_d

            step = 0.1
            new_x = spx + step * gx
            new_y = spy + step * gy

            move = abs(new_x - spx) + abs(new_y - spy)
            if move > max_move:
                max_move = move
            new_steiner.append((new_x, new_y))

        steiner_points = new_steiner
        all_points = list(points) + list(steiner_points)

        if max_move < 1e-8:
            break

    return steiner_points


def steiner_tree_length(terminals: List[Point]) -> float:
    """Compute Steiner tree length using iterative insertion heuristic."""
    n = len(terminals)
    if n <= 1:
        return 0.0
    if n == 2:
        return distance(terminals[0], terminals[1])
    if n == 3:
        length, _ = steiner_tree_3terminals(*terminals)
        return length

    points = list(terminals)
    steiner_points = []
    max_steiner = n - 2

    # Pre-allocate for speed
    all_points_cache = points.copy()

    improved = True
    while improved and len(steiner_points) < max_steiner:
        improved = False
        best_improvement = 0.0
        best_steiner = None

        all_points_cache = points + steiner_points
        n_all = len(all_points_cache)

        # Try all triples
        for i in range(n_all):
            pi = all_points_cache[i]
            for j in range(i+1, n_all):
                pj = all_points_cache[j]
                dij = distance(pi, pj)
                for k in range(j+1, n_all):
                    pk = all_points_cache[k]

                    # Current MST contribution
                    dik = distance(pi, pk)
                    djk = distance(pj, pk)
                    mst_3 = min(dij + djk, dij + dik, djk + dik)

                    fp = fermat_point(pi, pj, pk)
                    steiner_len = distance(fp, pi) + distance(fp, pj) + distance(fp, pk)
                    improvement = mst_3 - steiner_len

                    if improvement > best_improvement + 1e-6:
                        # Quick check for duplicates
                        too_close = False
                        for p in all_points_cache:
                            dx = fp[0] - p[0]
                            dy = fp[1] - p[1]
                            if dx*dx + dy*dy < 1e-12:  # Avoid sqrt
                                too_close = True
                                break

                        if not too_close:
                            for sp in steiner_points:
                                dx = fp[0] - sp[0]
                                dy = fp[1] - sp[1]
                                if dx*dx + dy*dy < 1e-12:
                                    too_close = True
                                    break

                            if not too_close:
                                best_improvement = improvement
                                best_steiner = fp

        if best_steiner is not None and best_improvement > 1e-6:
            steiner_points.append(best_steiner)
            improved = True

    # Compute final tree length
    all_points = points + steiner_points
    mst = mst_edges(all_points)
    total_length = sum(w for _, _, w in mst)

    # Local refinement
    if steiner_points:
        edges = [(i, j) for i, j, _ in mst]
        refined = local_refinement(points, steiner_points, edges)

        all_points_refined = points + refined
        mst_refined = mst_edges(all_points_refined)
        total_length_refined = sum(w for _, _, w in mst_refined)

        if total_length_refined < total_length:
            total_length = total_length_refined

    return total_length


def find_steiner_ratio(terminals: List[Point]) -> float:
    """Calculate the Steiner ratio for given terminals."""
    n = len(terminals)
    if n < 2:
        return 1.0
    if n == 2:
        return 1.0

    mst_len = mst_length(terminals)
    if mst_len < 1e-10:
        return 1.0

    steiner_len = steiner_tree_length(terminals)
    ratio = steiner_len / mst_len

    # Clamp to valid range [sqrt(3)/2, 1.0]
    min_ratio = 0.8660254037844386  # math.sqrt(3) / 2
    if ratio < min_ratio:
        ratio = min_ratio
    elif ratio > 1.0:
        ratio = 1.0

    return ratio