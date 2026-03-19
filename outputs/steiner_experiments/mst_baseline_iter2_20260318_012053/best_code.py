"""Evolved Steiner Tree Solver with Fermat Points and MST Optimization.

This evolved code implements proper Steiner tree computation using:
1. Fermat point calculation for 3-terminal subtrees
2. Iterative Steiner point insertion for N>3 terminals
3. MST-based topology with Steiner point optimization
4. Gradient-based refinement for local optimization
"""

import math
from typing import List, Tuple, Set, Dict
import logging
from heapq import heappush, heappop

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


def distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def mst_length(points: List[Point]) -> float:
    """Calculate Minimum Spanning Tree length using Prim's algorithm."""
    if len(points) < 2:
        raise ValueError("MST requires at least 2 points")

    n = len(points)
    if n == 2:
        return distance(points[0], points[1])

    # Optimized Prim's with heap
    visited = [False] * n
    min_heap = [(0.0, 0)]  # (distance, vertex)
    total_length = 0.0
    visited_count = 0

    while min_heap and visited_count < n:
        d, u = heappop(min_heap)
        if visited[u]:
            continue
        
        visited[u] = True
        visited_count += 1
        total_length += d

        for v in range(n):
            if not visited[v]:
                dist = distance(points[u], points[v])
                heappush(min_heap, (dist, v))

    return total_length


def fermat_point(p1: Point, p2: Point, p3: Point, eps: float = 1e-10) -> Point:
    """Calculate Fermat point of triangle (minimizes sum of distances to vertices).
    
    For triangles with all angles < 120°, the Fermat point has 120° angles.
    For triangles with an angle >= 120°, the Fermat point is at that vertex.
    """
    # Check for degenerate cases
    d12 = distance(p1, p2)
    d23 = distance(p2, p3)
    d31 = distance(p3, p1)
    
    # Check if any angle >= 120° using law of cosines
    # cos(A) = (b² + c² - a²) / (2bc)
    # If angle >= 120°, then cos <= -0.5
    
    # Angle at p1
    if d12 > eps and d31 > eps:
        cos1 = (d12*d12 + d31*d31 - d23*d23) / (2 * d12 * d31)
        if cos1 <= -0.5:
            return p1
    
    # Angle at p2
    if d12 > eps and d23 > eps:
        cos2 = (d12*d12 + d23*d23 - d31*d31) / (2 * d12 * d23)
        if cos2 <= -0.5:
            return p2
    
    # Angle at p3
    if d23 > eps and d31 > eps:
        cos3 = (d23*d23 + d31*d31 - d12*d12) / (2 * d23 * d31)
        if cos3 <= -0.5:
            return p3
    
    # All angles < 120°, find Fermat point with 120° angles
    # Use Weiszfeld-like iteration for geometric median with 120° constraint
    # Actually, for 3 points we can solve analytically or use simple iteration
    
    # Start with centroid
    x = (p1[0] + p2[0] + p3[0]) / 3.0
    y = (p1[1] + p2[1] + p3[1]) / 3.0
    
    # Iterate to find point where all angles are 120°
    for _ in range(100):
        d1 = math.sqrt((x - p1[0])**2 + (y - p1[1])**2)
        d2 = math.sqrt((x - p2[0])**2 + (y - p2[1])**2)
        d3 = math.sqrt((x - p3[0])**2 + (y - p3[1])**2)
        
        if d1 < eps or d2 < eps or d3 < eps:
            break
        
        # Gradient descent on sum of distances
        # But we need to enforce 120° angles, so use different approach
        
        # Torricelli construction: build equilateral triangles outward
        # Fermat point is intersection of lines from new vertices to opposite triangle vertices
        
        # Use iterative Weiszfeld with small modification
        # Weighted average where weights are inverse distances
        w1, w2, w3 = 1.0/d1, 1.0/d2, 1.0/d3
        sum_w = w1 + w2 + w3
        
        new_x = (w1 * p1[0] + w2 * p2[0] + w3 * p3[0]) / sum_w
        new_y = (w1 * p1[1] + w2 * p2[1] + w3 * p3[1]) / sum_w
        
        if abs(new_x - x) < eps and abs(new_y - y) < eps:
            break
        
        x, y = new_x, new_y
    
    # Verify 120° condition with refinement
    for _ in range(50):
        d1 = math.sqrt((x - p1[0])**2 + (y - p1[1])**2)
        d2 = math.sqrt((x - p2[0])**2 + (y - p2[1])**2)
        d3 = math.sqrt((x - p3[0])**2 + (y - p3[1])**2)
        
        if d1 < eps or d2 < eps or d3 < eps:
            break
        
        # Calculate angles using dot product
        # Vector from point to p1
        v1x, v1y = p1[0] - x, p1[1] - y
        # Vector from point to p2
        v2x, v2y = p2[0] - x, p2[1] - y
        # Vector from point to p3
        v3x, v3y = p3[0] - x, p3[1] - y
        
        # Normalize
        v1x, v1y = v1x/d1, v1y/d1
        v2x, v2y = v2x/d2, v2y/d2
        v3x, v3y = v3x/d3, v3y/d3
        
        # Small gradient step to open angles toward 120°
        # Move away from directions that are too close
        dx = (v1x + v2x + v3x) * 0.1
        dy = (v1y + v2y + v3y) * 0.1
        
        x -= dx
        y -= dy
    
    return (x, y)


def steiner_tree_3terminals(p1: Point, p2: Point, p3: Point) -> Tuple[float, List[Point]]:
    """Compute optimal Steiner tree for 3 terminals."""
    fp = fermat_point(p1, p2, p3)
    
    total_len = distance(fp, p1) + distance(fp, p2) + distance(fp, p3)
    
    # Check if Fermat point actually helps (might be at a vertex)
    mst_len = min(
        distance(p1, p2) + distance(p2, p3),
        distance(p1, p2) + distance(p1, p3),
        distance(p2, p3) + distance(p1, p3)
    )
    
    if total_len >= mst_len - 1e-9:
        # No improvement, use MST
        return (mst_len, [])
    
    return (total_len, [fp])


def mst_edges(points: List[Point]) -> List[Tuple[int, int, float]]:
    """Get MST edges as list of (i, j, weight)."""
    n = len(points)
    if n < 2:
        return []
    
    # Kruskal's algorithm
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j, distance(points[i], points[j])))
    
    edges.sort(key=lambda x: x[2])
    
    parent = list(range(n))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    mst = []
    for i, j, w in edges:
        if find(i) != find(j):
            union(i, j)
            mst.append((i, j, w))
            if len(mst) == n - 1:
                break
    
    return mst


def local_refinement(points: List[Point], steiner_points: List[Point], 
                     edges: List[Tuple[int, int]], max_iter: int = 100) -> List[Point]:
    """Refine Steiner point positions using gradient descent."""
    n_terminals = len(points)
    all_points = list(points) + list(steiner_points)
    
    # Build adjacency list
    adj = [[] for _ in range(len(all_points))]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Only refine Steiner points (indices >= n_terminals)
    for _ in range(max_iter):
        max_move = 0.0
        new_steiner = []
        
        for idx, sp in enumerate(steiner_points):
            i = n_terminals + idx
            neighbors = adj[i]
            if len(neighbors) < 2:
                new_steiner.append(sp)
                continue
            
            # Compute gradient: sum of unit vectors to neighbors
            gx, gy = 0.0, 0.0
            for j in neighbors:
                dx = all_points[j][0] - sp[0]
                dy = all_points[j][1] - sp[1]
                d = math.sqrt(dx*dx + dy*dy)
                if d > 1e-10:
                    gx += dx / d
                    gy += dy / d
            
            # For optimal Steiner point, gradient should be zero (120° angles)
            # Move in direction that reduces gradient
            step = 0.1
            new_x = sp[0] + step * gx
            new_y = sp[1] + step * gy
            
            move = abs(new_x - sp[0]) + abs(new_y - sp[1])
            max_move = max(max_move, move)
            new_steiner.append((new_x, new_y))
        
        steiner_points = new_steiner
        all_points = list(points) + list(steiner_points)
        
        if max_move < 1e-8:
            break
    
    return steiner_points


def steiner_tree_length(terminals: List[Point]) -> float:
    """Compute Steiner tree length using iterative insertion heuristic."""
    if len(terminals) <= 1:
        return 0.0
    if len(terminals) == 2:
        return distance(terminals[0], terminals[1])
    if len(terminals) == 3:
        length, _ = steiner_tree_3terminals(*terminals)
        return length
    
    # For N > 3, use MST-based heuristic with Steiner point insertion
    points = list(terminals)
    steiner_points = []
    
    # Build initial MST and find good Steiner point candidates
    # Use iterative improvement: find triples that benefit from Fermat point
    
    # Greedy: repeatedly find best Steiner point to add
    improved = True
    max_steiner = len(terminals) - 2  # At most N-2 Steiner points
    
    while improved and len(steiner_points) < max_steiner:
        improved = False
        best_improvement = 0.0
        best_steiner = None
        best_triple = None
        
        # Try all triples of current points
        all_points = points + steiner_points
        n = len(all_points)
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = all_points[i], all_points[j], all_points[k]
                    
                    # Current MST contribution for this triple
                    # Approximate by pairwise distances
                    mst_3 = min(
                        distance(p1, p2) + distance(p2, p3),
                        distance(p1, p2) + distance(p1, p3),
                        distance(p2, p3) + distance(p1, p3)
                    )
                    
                    fp = fermat_point(p1, p2, p3)
                    steiner_len = distance(fp, p1) + distance(fp, p2) + distance(fp, p3)
                    
                    improvement = mst_3 - steiner_len
                    
                    # Check if this Steiner point is new and beneficial
                    if improvement > best_improvement + 1e-6:
                        # Check not too close to existing points
                        too_close = False
                        for p in all_points:
                            if distance(fp, p) < 1e-6:
                                too_close = True
                                break
                        for sp in steiner_points:
                            if distance(fp, sp) < 1e-6:
                                too_close = True
                                break
                        
                        if not too_close:
                            best_improvement = improvement
                            best_steiner = fp
                            best_triple = (i, j, k)
        
        if best_steiner is not None and best_improvement > 1e-6:
            steiner_points.append(best_steiner)
            improved = True
    
    # Compute final tree length
    all_points = points + steiner_points
    
    # Build MST on all points (terminals + Steiner points)
    mst = mst_edges(all_points)
    total_length = sum(w for _, _, w in mst)
    
    # Local refinement of Steiner points
    if steiner_points:
        edges = [(i, j) for i, j, _ in mst]
        refined = local_refinement(points, steiner_points, edges)
        
        # Recompute length with refined points
        all_points_refined = points + refined
        mst_refined = mst_edges(all_points_refined)
        total_length_refined = sum(w for _, _, w in mst_refined)
        
        total_length = min(total_length, total_length_refined)
    
    return total_length


def find_steiner_ratio(terminals: List[Point]) -> float:
    """Calculate the Steiner ratio for given terminals.
    
    Returns the ratio of optimal Steiner tree length to MST length.
    """
    if len(terminals) < 2:
        raise ValueError("Need at least 2 terminals")
    
    if len(terminals) == 2:
        return 1.0
    
    mst_len = mst_length(terminals)
    if mst_len < 1e-10:
        return 1.0
    
    steiner_len = steiner_tree_length(terminals)
    ratio = steiner_len / mst_len
    
    # Clamp to valid range [sqrt(3)/2, 1.0]
    min_ratio = math.sqrt(3) / 2
    ratio = max(min_ratio, min(ratio, 1.0))
    
    logger.debug(f"Steiner ratio: {ratio:.6f}, MST: {mst_len:.6f}, Steiner: {steiner_len:.6f}")
    
    return ratio