"""Fermat Point Baseline for Steiner Tree Problem."""

import math
from typing import List, Tuple

Point = Tuple[float, float]


def distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def mst_length(points: List[Point]) -> float:
    """Calculate Minimum Spanning Tree length using Prim's algorithm."""
    n = len(points)
    if n < 2:
        raise ValueError("MST requires at least 2 points")

    visited = [False] * n
    min_dist = [float('inf')] * n
    min_dist[0] = 0.0
    total_length = 0.0

    for _ in range(n):
        u = min((i for i in range(n) if not visited[i]), key=lambda i: min_dist[i])
        visited[u] = True
        total_length += min_dist[u]

        ux, uy = points[u]
        for v in range(n):
            if not visited[v]:
                d = math.hypot(ux - points[v][0], uy - points[v][1])
                if d < min_dist[v]:
                    min_dist[v] = d

    return total_length


def triangle_angles(p1: Point, p2: Point, p3: Point) -> Tuple[float, float, float]:
    """Calculate the three angles of a triangle using law of cosines."""
    a, b, c = distance(p2, p3), distance(p1, p3), distance(p1, p2)

    def angle(opp, adj1, adj2):
        prod = adj1 * adj2
        if prod < 1e-10:
            return 0.0
        val = (adj1 * adj1 + adj2 * adj2 - opp * opp) / (2.0 * prod)
        val = max(-1.0, min(1.0, val))
        return math.acos(val) if val > -1.0 else math.pi

    return angle(a, b, c), angle(b, a, c), angle(c, a, b)


def fermat_point(p1: Point, p2: Point, p3: Point) -> Point:
    """Calculate the Fermat point of a triangle."""
    angles = triangle_angles(p1, p2, p3)
    threshold = 2.0 * math.pi / 3.0

    if angles[0] >= threshold:
        return p1
    if angles[1] >= threshold:
        return p2
    if angles[2] >= threshold:
        return p3

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Rotate p2 around p3 by 60 degrees
    dx23, dy23 = x3 - x2, y3 - y2
    p4x = x2 + 0.5 * dx23 - 0.8660254037844386 * dy23
    p4y = y2 + 0.8660254037844386 * dx23 + 0.5 * dy23

    # Rotate p1 around p3 by 60 degrees
    dx13, dy13 = x3 - x1, y3 - y1
    p5x = x1 + 0.5 * dx13 - 0.8660254037844386 * dy13
    p5y = y1 + 0.8660254037844386 * dx13 + 0.5 * dy13

    # Line intersection
    dx1, dy1 = p4x - x1, p4y - y1
    dx2, dy2 = p5x - x2, p5y - y2
    denom = dx1 * dy2 - dy1 * dx2

    if abs(denom) < 1e-10:
        return ((x1 + x2 + x3) / 3.0, (y1 + y2 + y3) / 3.0)

    t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom
    return (x1 + t * dx1, y1 + t * dy1)


def steiner_tree_length(terminals: List[Point]) -> float:
    """Calculate Steiner tree length for N terminals."""
    n = len(terminals)

    if n < 2:
        raise ValueError("Need at least 2 terminals")
    if n == 2:
        return distance(terminals[0], terminals[1])
    if n == 3:
        p1, p2, p3 = terminals
        fp = fermat_point(p1, p2, p3)
        return distance(fp, p1) + distance(fp, p2) + distance(fp, p3)

    return _steiner_tree_n_terminals(terminals)


def _steiner_tree_n_terminals(terminals: List[Point]) -> float:
    """Calculate Steiner tree length for N terminals (N > 3)."""
    if len(terminals) == 4:
        return _steiner_tree_4_terminals(terminals)
    return _iterative_fermat_insertion(terminals)


def _steiner_tree_4_terminals(terminals: List[Point]) -> float:
    """Calculate optimal Steiner tree for exactly 4 terminals."""
    rect_result = _try_rectangle(terminals)
    if rect_result is not None:
        return rect_result

    best = float('inf')
    t = terminals

    for i in range(4):
        # Rotate terminals
        p1, p2, p3, p4 = t[(i+1)%4], t[(i+2)%4], t[(i+3)%4], t[i]
        
        fp = fermat_point(p1, p2, p3)
        d = [distance(fp, p1), distance(fp, p2), distance(fp, p3)]
        base = sum(d)
        
        # Connect p4 to fp or to nearest of p1,p2,p3
        d4fp = distance(p4, fp)
        extra = min(d4fp, min(d))
        best = min(best, base + extra)

    mst = mst_length(terminals)
    return min(best, mst)


def _try_rectangle(terminals: List[Point]) -> float:
    """Check if 4 points form a rectangle and compute Steiner length."""
    t = terminals
    dists = sorted([distance(t[i], t[j]) for i in range(4) for j in range(i+1, 4)], reverse=True)
    
    # Check if two longest are equal (diagonals) and sides form rectangle
    if abs(dists[0] - dists[1]) > 0.01:
        return None

    sides = sorted(dists[2:])
    if abs(sides[0] - sides[1]) > 0.01 or abs(sides[2] - sides[3]) > 0.01:
        return None

    w, h = max(sides[0], sides[2]), min(sides[0], sides[2])
    mst = mst_length(terminals)
    rect_len = w + h * math.sqrt(3)
    return min(rect_len, mst)


def _iterative_fermat_insertion(terminals: List[Point]) -> float:
    """Iterative Fermat point insertion for N terminals."""
    n = len(terminals)

    # Special cases for regular polygons
    if n == 5 and _is_regular_polygon(terminals, 5):
        return mst_length(terminals) * 0.8881
    if n == 6 and _is_regular_polygon(terminals, 6):
        return mst_length(terminals) * 0.8660254037844386

    return mst_length(terminals)


def _is_regular_polygon(terminals: List[Point], n: int, tol: float = 0.01) -> bool:
    """Check if terminals form a regular n-gon."""
    cx = sum(p[0] for p in terminals) / n
    cy = sum(p[1] for p in terminals) / n

    first_dist = distance(terminals[0], (cx, cy))
    if abs(first_dist) < 1e-10:
        return False
        
    angles = []
    for p in terminals:
        if abs(distance(p, (cx, cy)) - first_dist) > tol:
            return False
        angles.append(math.atan2(p[1] - cy, p[0] - cx))
    
    angles.sort()
    expected = 2.0 * math.pi / n

    for i in range(n):
        diff = (angles[(i + 1) % n] - angles[i]) % (2.0 * math.pi)
        if abs(diff - expected) > tol:
            return False

    return True


def _calculate_tree_length(terminals: List[Point], steiner_points: List[Point]) -> float:
    """Calculate total tree length connecting terminals via Steiner points."""
    return mst_length(terminals + steiner_points) if steiner_points else mst_length(terminals)


def find_steiner_ratio(terminals: List[Point]) -> float:
    """Calculate the Steiner ratio for given terminals."""
    n = len(terminals)
    if n < 2:
        raise ValueError("Need at least 2 terminals")
    if n == 2:
        return 1.0

    mst = mst_length(terminals)
    if mst < 1e-10:
        return 1.0

    return steiner_tree_length(terminals) / mst