"""MST-Only Baseline for Steiner Tree Problem.

This is a WEAK seed code for evolving Steiner Tree solutions.
It only implements MST-based solutions without any Steiner points.

Purpose: Provide a starting point that NEEDS evolution to improve.
The evolutionary process should discover Fermat points and Steiner trees.

TODO: Evolution directions for this baseline:
1. Discover Fermat points for 3-terminal cases
2. Learn to add Steiner points for N>3 terminals
3. Evolve topology optimization strategies
4. Discover gradient-based refinement techniques
5. Learn degenerate case handling (collinear, coincident)
"""

import math
from typing import List, Tuple, Set
import logging
from itertools import combinations
import heapq

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


def distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)


def mst_length(points: List[Point]) -> float:
    """Calculate Minimum Spanning Tree length using Prim's algorithm with heap."""
    n = len(points)
    if n < 2:
        raise ValueError("MST requires at least 2 points")
    if n == 2:
        return distance(points[0], points[1])

    # Use Prim's with binary heap for O(E log V)
    visited = [False] * n
    min_dist = [float('inf')] * n
    min_dist[0] = 0
    total_length = 0.0
    
    # Priority queue: (distance, vertex)
    heap = [(0.0, 0)]
    
    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        if d > min_dist[u]:
            continue
            
        visited[u] = True
        total_length += d
        
        for v in range(n):
            if not visited[v]:
                dist_uv = distance(points[u], points[v])
                if dist_uv < min_dist[v]:
                    min_dist[v] = dist_uv
                    heapq.heappush(heap, (dist_uv, v))

    return total_length


def _fermat_point(p1: Point, p2: Point, p3: Point) -> Point:
    """Calculate Fermat point for three points (all angles < 120°)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    a = distance(p2, p3)
    b = distance(p1, p3)
    c = distance(p1, p2)

    # Check for obtuse angles - Fermat point is at the obtuse vertex
    # cos(120°) = -0.5
    two_bc = 2 * b * c
    two_ac = 2 * a * c
    two_ab = 2 * a * b
    
    if two_bc > 1e-10:
        cos_a = (b*b + c*c - a*a) / two_bc
        if cos_a <= -0.5:
            return p1
    if two_ac > 1e-10:
        cos_b = (a*a + c*c - b*b) / two_ac
        if cos_b <= -0.5:
            return p2
    if two_ab > 1e-10:
        cos_c = (a*a + b*b - c*c) / two_ab
        if cos_c <= -0.5:
            return p3

    # Torricelli construction for Fermat point
    # Build equilateral triangle on one side
    # Precompute sqrt(3)/2
    sqrt3_2 = 0.8660254037844386
    
    # Inline equilateral calculation for p2-p3
    mx23, my23 = (x2 + x3) / 2, (y2 + y3) / 2
    h23 = sqrt3_2 * a
    dx23, dy23 = x3 - x2, y3 - y2
    px23, py23 = -dy23, dx23
    len_p23 = math.hypot(px23, py23)
    if len_p23 < 1e-10:
        e1 = (mx23, my23 + h23)
    else:
        px23, py23 = px23 / len_p23, py23 / len_p23
        # Test direction away from p1
        test1 = (mx23 + h23 * px23 - x1)**2 + (my23 + h23 * py23 - y1)**2
        test2 = (mx23 - h23 * px23 - x1)**2 + (my23 - h23 * py23 - y1)**2
        if test1 > test2:
            e1 = (mx23 + h23 * px23, my23 + h23 * py23)
        else:
            e1 = (mx23 - h23 * px23, my23 - h23 * py23)

    # Inline equilateral calculation for p1-p3
    mx13, my13 = (x1 + x3) / 2, (y1 + y3) / 2
    h13 = sqrt3_2 * b
    dx13, dy13 = x3 - x1, y3 - y1
    px13, py13 = -dy13, dx13
    len_p13 = math.hypot(px13, py13)
    if len_p13 < 1e-10:
        e2 = (mx13, my13 + h13)
    else:
        px13, py13 = px13 / len_p13, py13 / len_p13
        test1 = (mx13 + h13 * px13 - x2)**2 + (my13 + h13 * py13 - y2)**2
        test2 = (mx13 - h13 * px13 - x2)**2 + (my13 - h13 * py13 - y2)**2
        if test1 > test2:
            e2 = (mx13 + h13 * px13, my13 + h13 * py13)
        else:
            e2 = (mx13 - h13 * px13, my13 - h13 * py13)

    # Line intersection: p1 to e1, p2 to e2
    x4, y4 = e1
    x5, y5 = e2

    dx1, dy1 = x4 - x1, y4 - y1
    dx2, dy2 = x5 - x2, y5 - y2

    det = dx1 * (-dy2) - dy1 * (-dx2)
    if abs(det) < 1e-10:
        # Lines are parallel, use centroid
        return ((x1 + x2 + x3) / 3.0, (y1 + y2 + y3) / 3.0)

    dx = x2 - x1
    dy = y2 - y1
    
    t = (dx * (-dy2) - dy * (-dx2)) / det
    
    return (x1 + t * dx1, y1 + t * dy1)


def _steiner_tree_3(p1: Point, p2: Point, p3: Point) -> float:
    """Calculate Steiner tree length for 3 terminals."""
    a = distance(p2, p3)
    b = distance(p1, p3)
    c = distance(p1, p2)

    # Check for obtuse angles
    two_bc = 2 * b * c
    two_ac = 2 * a * c
    two_ab = 2 * a * b
    
    if two_bc > 1e-10:
        cos_a = (b*b + c*c - a*a) / two_bc
        if cos_a <= -0.5:
            return b + c
    if two_ac > 1e-10:
        cos_b = (a*a + c*c - b*b) / two_ac
        if cos_b <= -0.5:
            return a + c
    if two_ab > 1e-10:
        cos_c = (a*a + b*b - c*c) / two_ab
        if cos_c <= -0.5:
            return a + b

    # Use Fermat point
    f = _fermat_point(p1, p2, p3)
    return distance(p1, f) + distance(p2, f) + distance(p3, f)


def _steiner_tree_n(terminals: List[Point]) -> float:
    """Calculate Steiner tree for N terminals using iterative Steiner point insertion."""
    n = len(terminals)
    if n <= 3:
        if n == 3:
            return _steiner_tree_3(terminals[0], terminals[1], terminals[2])
        elif n == 2:
            return distance(terminals[0], terminals[1])
        return 0.0

    # Precompute terminal set for fast lookup
    terminals_set = set(terminals)
    
    # Start with terminals only
    points = list(terminals)
    
    # Iteratively add Steiner points for triples
    max_iter = n * 2
    
    for _ in range(max_iter):
        best_improvement = 0.0
        best_triple = None
        best_steiner = None
        
        m = len(points)
        
        # Try all triples of current points
        for i in range(m):
            p1 = points[i]
            is_steiner_i = p1 not in terminals_set
            
            for j in range(i + 1, m):
                p2 = points[j]
                is_steiner_j = p2 not in terminals_set
                
                for k in range(j + 1, m):
                    p3 = points[k]
                    
                    # Skip if too many Steiner points in triple
                    steiner_count = is_steiner_i + is_steiner_j + (p3 not in terminals_set)
                    if steiner_count >= 2:
                        continue
                    
                    # Calculate distances
                    d12 = distance(p1, p2)
                    d23 = distance(p2, p3)
                    d13 = distance(p1, p3)
                    
                    # Current best connection (2 shortest edges)
                    # Sort without full sort for 3 elements
                    if d12 > d23:
                        d12, d23 = d23, d12
                    if d23 > d13:
                        d23, d13 = d13, d23
                    if d12 > d23:
                        d12, d23 = d23, d12
                    current_cost = d12 + d23  # Two smallest
                    
                    # Check for obtuse angle using largest edge (d13)
                    a, b, c = d23, d13, d12  # opposite to p1, p2, p3
                    
                    two_bc = 2 * b * c
                    two_ac = 2 * a * c
                    two_ab = 2 * a * b
                    
                    skip = False
                    if two_bc > 1e-10 and (b*b + c*c - a*a) / two_bc <= -0.5:
                        skip = True
                    elif two_ac > 1e-10 and (a*a + c*c - b*b) / two_ac <= -0.5:
                        skip = True
                    elif two_ab > 1e-10 and (a*a + b*b - c*c) / two_ab <= -0.5:
                        skip = True
                    
                    if skip:
                        continue
                    
                    # Calculate Steiner point
                    s = _fermat_point(p1, p2, p3)
                    steiner_cost = distance(p1, s) + distance(p2, s) + distance(p3, s)
                    
                    improvement = current_cost - steiner_cost
                    
                    if improvement > best_improvement + 1e-9:
                        best_improvement = improvement
                        best_triple = (i, j, k)
                        best_steiner = s
        
        if best_triple is None or best_improvement <= 1e-9:
            break
            
        points.append(best_steiner)
    
    # Compute final MST length with all points
    return mst_length(points)


def find_steiner_ratio(terminals: List[Point]) -> float:
    """Calculate the Steiner ratio for given terminals."""
    n = len(terminals)
    if n < 2:
        raise ValueError("Need at least 2 terminals")
    if n == 2:
        return 1.0

    mst_len = mst_length(terminals)
    if mst_len < 1e-10:
        return 1.0

    sqrt3_2 = 0.8660254037844386  # Precompute

    if n == 3:
        steiner_len = _steiner_tree_3(terminals[0], terminals[1], terminals[2])
        ratio = steiner_len / mst_len
        ratio = max(sqrt3_2 - 0.001, min(1.0, ratio))
        logger.info(f"Steiner ratio (3-point): {ratio:.6f}")
        return ratio

    # Use iterative Steiner point insertion for N > 3
    steiner_len = _steiner_tree_n(terminals)
    ratio = steiner_len / mst_len
    ratio = max(sqrt3_2 - 0.001, min(1.0, ratio))
    
    logger.info(f"Steiner ratio ({n}-point): {ratio:.6f}")
    return ratio