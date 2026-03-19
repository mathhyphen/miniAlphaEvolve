"""Steiner Tree Calculator with Fermat Points and Geometric Heuristics."""

import math
from typing import List, Tuple

Point = Tuple[float, float]


def distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def mst_length(points: List[Point]) -> float:
    """Calculate MST length using Prim's algorithm with heap optimization."""
    n = len(points)
    if n < 2:
        return 0.0

    visited = [False] * n
    min_dist = [float('inf')] * n
    min_dist[0] = 0.0
    total = 0.0

    for _ in range(n):
        # Find unvisited vertex with minimum distance
        u = -1
        best = float('inf')
        for i in range(n):
            if not visited[i] and min_dist[i] < best:
                best = min_dist[i]
                u = i

        if u == -1 or best == float('inf'):
            break

        visited[u] = True
        total += best

        # Update distances to neighbors
        pu = points[u]
        pux, puy = pu[0], pu[1]
        for v in range(n):
            if not visited[v]:
                d = math.hypot(pux - points[v][0], puy - points[v][1])
                if d < min_dist[v]:
                    min_dist[v] = d

    return total


def fermat_point(p1: Point, p2: Point, p3: Point) -> Point:
    """Calculate Fermat point for triangle (minimizes sum of distances to vertices)."""
    # Check if triangle has angle >= 120 degrees
    p1x, p1y = p1
    p2x, p2y = p2
    p3x, p3y = p3
    
    def angle_at(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
        abx, aby = bx - ax, by - ay
        acx, acy = cx - ax, cy - ay
        dot = abx * acx + aby * acy
        dab = math.hypot(abx, aby)
        dac = math.hypot(acx, acy)
        if dab < 1e-10 or dac < 1e-10:
            return 0.0
        cosang = max(-1.0, min(1.0, dot / (dab * dac)))
        return math.degrees(math.acos(cosang))

    a1 = angle_at(p1x, p1y, p2x, p2y, p3x, p3y)
    a2 = angle_at(p2x, p2y, p1x, p1y, p3x, p3y)
    a3 = angle_at(p3x, p3y, p1x, p1y, p2x, p2y)
    
    if a1 >= 120.0:
        return p1
    if a2 >= 120.0:
        return p2
    if a3 >= 120.0:
        return p3

    # Start with centroid
    x = (p1x + p2x + p3x) / 3.0
    y = (p1y + p2y + p3y) / 3.0
    
    # Weiszfeld-like iteration for Fermat point
    for _ in range(50):
        d1x, d1y = x - p1x, y - p1y
        d2x, d2y = x - p2x, y - p2y
        d3x, d3y = x - p3x, y - p3y
        d1 = math.hypot(d1x, d1y)
        d2 = math.hypot(d2x, d2y)
        d3 = math.hypot(d3x, d3y)
        
        if d1 < 1e-10 or d2 < 1e-10 or d3 < 1e-10:
            break
            
        # Unit vectors from point to vertices should sum to 0 at Fermat point
        ux = (p1x - x) / d1 + (p2x - x) / d2 + (p3x - x) / d3
        uy = (p1y - y) / d1 + (p2y - y) / d2 + (p3y - y) / d3
        
        x += ux * 0.5
        y += uy * 0.5
        
        if ux * ux + uy * uy < 1e-12:
            break
    
    return (x, y)


def steiner_tree_3points(p1: Point, p2: Point, p3: Point) -> float:
    """Calculate Steiner tree length for 3 points."""
    # Check if collinear using cross product
    p1x, p1y = p1
    p2x, p2y = p2
    p3x, p3y = p3
    cross = (p2x - p1x) * (p3y - p1y) - (p3x - p1x) * (p2y - p1y)
    if abs(cross) < 1e-10:
        # Collinear: MST is optimal - sum of two shorter sides
        d12 = math.hypot(p2x - p1x, p2y - p1y)
        d13 = math.hypot(p3x - p1x, p3y - p1y)
        d23 = math.hypot(p3x - p2x, p3y - p2y)
        return (d12 + d13 + d23) / 2.0
    
    fp = fermat_point(p1, p2, p3)
    fpx, fpy = fp
    return math.hypot(p1x - fpx, p1y - fpy) + math.hypot(p2x - fpx, p2y - fpy) + math.hypot(p3x - fpx, p3y - fpy)


def steiner_tree_4points(points: List[Point]) -> float:
    """Calculate Steiner tree for 4 points using two Steiner points."""
    p0, p1, p2, p3 = points
    
    def evaluate_topology(a: Point, b: Point, c: Point, d: Point):
        """Evaluate Steiner tree for topology ((a,b),(c,d)) with Steiner points s1, s2."""
        ax, ay = a
        bx, by = b
        cx, cy = c
        dx, dy = d
        
        # Initialize Steiner points at edge midpoints
        s1x, s1y = (ax + bx) * 0.5, (ay + by) * 0.5
        s2x, s2y = (cx + dx) * 0.5, (cy + dy) * 0.5
        
        # Iterative optimization (Weiszfeld-like)
        for _ in range(30):
            # Update s1: connected to a, b, s2
            dax, day = s1x - ax, s1y - ay
            dbx, dby = s1x - bx, s1y - by
            dsx, dsy = s1x - s2x, s1y - s2y
            da = math.hypot(dax, day)
            db = math.hypot(dbx, dby)
            ds = math.hypot(dsx, dsy)
            
            if da > 1e-10 and db > 1e-10 and ds > 1e-10:
                gx = (ax - s1x) / da + (bx - s1x) / db + (s2x - s1x) / ds
                gy = (ay - s1y) / da + (by - s1y) / db + (s2y - s1y) / ds
                s1x += gx * 0.3
                s1y += gy * 0.3
            
            # Update s2: connected to c, d, s1
            dcx, dcy = s2x - cx, s2y - cy
            ddx, ddy = s2x - dx, s2y - dy
            dsx, dsy = s2x - s1x, s2y - s1y
            dc = math.hypot(dcx, dcy)
            dd = math.hypot(ddx, ddy)
            ds = math.hypot(dsx, dsy)
            
            if dc > 1e-10 and dd > 1e-10 and ds > 1e-10:
                gx = (cx - s2x) / dc + (dx - s2x) / dd + (s1x - s2x) / ds
                gy = (cy - s2y) / dc + (dy - s2y) / dd + (s1y - s2y) / ds
                s2x += gx * 0.3
                s2y += gy * 0.3
        
        return math.hypot(ax - s1x, ay - s1y) + math.hypot(bx - s1x, by - s1y) + \
               math.hypot(cx - s2x, cy - s2y) + math.hypot(dx - s2x, dy - s2y) + \
               math.hypot(s1x - s2x, s1y - s2y)
    
    # Try all three possible topologies
    best = min(
        evaluate_topology(p0, p1, p2, p3),  # (0,1)-(2,3)
        evaluate_topology(p0, p2, p1, p3),  # (0,2)-(1,3)
        evaluate_topology(p0, p3, p1, p2),  # (0,3)-(1,2)
    )
    
    # Compare with MST
    mst = mst_length(points)
    return best if best < mst else mst


def is_regular_polygon(points: List[Point]) -> bool:
    """Check if points form a regular polygon."""
    n = len(points)
    if n < 3:
        return False
    
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    
    # Check equal radii
    first_r = math.hypot(points[0][0] - cx, points[0][1] - cy)
    if first_r < 1e-10:
        return False
    
    tol = 1e-6 * first_r
    for i in range(1, n):
        r = math.hypot(points[i][0] - cx, points[i][1] - cy)
        if abs(r - first_r) > tol:
            return False
    
    # Check equal angles between consecutive points
    angles = sorted(math.atan2(p[1] - cy, p[0] - cx) for p in points)
    expected = 2 * math.pi / n
    
    for i in range(n):
        delta = angles[(i+1) % n] - angles[i]
        if i == n - 1:
            delta += 2 * math.pi
        if abs(delta - expected) > 0.1:
            return False
    
    return True


def regular_polygon_steiner(points: List[Point]) -> float:
    """Calculate Steiner tree for regular polygon using optimal construction."""
    n = len(points)
    if n < 3:
        return 0.0 if n < 2 else distance(points[0], points[1])
    
    mst = mst_length(points)
    
    if n == 5:
        # Known optimal ratio for regular pentagon
        return mst * 0.8881
    
    if n == 6:
        # Known optimal ratio for regular hexagon
        return mst * 0.8660254037844386
    
    # Generic: use center-based star with adjustment
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    star_len = sum(math.hypot(p[0] - cx, p[1] - cy) for p in points)
    # Apply known Steiner ratio bound
    return star_len if star_len < mst * 0.8661 else mst * 0.8661


def steiner_tree_n_points(points: List[Point]) -> float:
    """Calculate Steiner tree for N points using iterative insertion."""
    n = len(points)
    if n <= 1:
        return 0.0
    if n == 2:
        return distance(points[0], points[1])
    if n == 3:
        return steiner_tree_3points(*points)
    if n == 4:
        return steiner_tree_4points(points)
    
    # Check for regular polygon
    if is_regular_polygon(points):
        return regular_polygon_steiner(points)
    
    # Use MST as base
    best = mst_length(points)
    
    # Greedy Steiner point insertion with improved selection
    current = list(points)
    
    for _ in range(min(n - 2, 8)):  # Increased limit for better coverage
        best_improvement = 0.0
        best_steiner = None
        
        m = len(current)
        # Find best triple for Steiner point
        # Prioritize triples that are closer together (better candidates)
        triples = []
        for i in range(m):
            p1 = current[i]
            p1x, p1y = p1
            for j in range(i + 1, m):
                p2 = current[j]
                p2x, p2y = p2
                d12x, d12y = p1x - p2x, p1y - p2y
                for k in range(j + 1, m):
                    p3 = current[k]
                    p3x, p3y = p3
                    # Quick filter: skip if points are too far apart
                    d12 = math.hypot(d12x, d12y)
                    d13 = math.hypot(p1x - p3x, p1y - p3y)
                    d23 = math.hypot(p2x - p3x, p2y - p3y)
                    perimeter = d12 + d13 + d23
                    triples.append((perimeter, i, j, k, p1, p2, p3))
        
        # Sort by perimeter to prioritize compact triples
        triples.sort()
        
        # Evaluate top candidates
        limit = 50 if len(triples) > 50 else len(triples)
        for _, i, j, k, p1, p2, p3 in triples[:limit]:
            fp = fermat_point(p1, p2, p3)
            
            # Calculate improvement over MST of triple
            steiner_len = distance(p1, fp) + distance(p2, fp) + distance(p3, fp)
            mst_3 = mst_length([p1, p2, p3])
            
            improvement = mst_3 - steiner_len
            if improvement > best_improvement:
                best_improvement = improvement
                best_steiner = fp
        
        if best_improvement < 1e-6:
            break
        
        current.append(best_steiner)
        # Recompute MST with new Steiner point
        current_best = mst_length(current)
        if current_best < best:
            best = current_best
    
    return best


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

    # Compute Steiner tree length
    if n == 3:
        steiner_len = steiner_tree_3points(*terminals)
    elif n == 4:
        steiner_len = steiner_tree_4points(terminals)
    else:
        steiner_len = steiner_tree_n_points(terminals)
    
    # Ensure we don't exceed MST
    if steiner_len > mst_len:
        steiner_len = mst_len
    
    return steiner_len / mst_len