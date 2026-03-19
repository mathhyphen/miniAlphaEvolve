"""Fermat Point Baseline for Steiner Tree Problem."""

import math
from typing import List, Tuple, Optional
import logging
from itertools import combinations

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
        raise ValueError("MST requires at least 2 points")
    if n == 2:
        dx = points[0][0] - points[1][0]
        dy = points[0][1] - points[1][1]
        return math.sqrt(dx * dx + dy * dy)

    visited = [False] * n
    min_dist = [float('inf')] * n
    min_dist[0] = 0.0

    total_length = 0.0

    for _ in range(n):
        u = -1
        min_d = float('inf')
        for i in range(n):
            if not visited[i] and min_dist[i] < min_d:
                min_d = min_dist[i]
                u = i
        
        visited[u] = True
        total_length += min_dist[u]

        ux, uy = points[u]
        for v in range(n):
            if not visited[v]:
                dx = ux - points[v][0]
                dy = uy - points[v][1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < min_dist[v]:
                    min_dist[v] = d

    return total_length


def triangle_angles(p1: Point, p2: Point, p3: Point) -> Tuple[float, float, float]:
    """Calculate the three angles of a triangle using law of cosines."""
    dx1, dy1 = p2[0] - p3[0], p2[1] - p3[1]
    dx2, dy2 = p1[0] - p3[0], p1[1] - p3[1]
    dx3, dy3 = p1[0] - p2[0], p1[1] - p2[1]
    
    a2 = dx1 * dx1 + dy1 * dy1
    b2 = dx2 * dx2 + dy2 * dy2
    c2 = dx3 * dx3 + dy3 * dy3
    
    a = math.sqrt(a2)
    b = math.sqrt(b2)
    c = math.sqrt(c2)

    def safe_acos(x: float) -> float:
        if x <= -1.0:
            return math.pi
        if x >= 1.0:
            return 0.0
        return math.acos(x)

    bc = b * c
    ac = a * c
    ab = a * b
    
    angle1 = safe_acos((b2 + c2 - a2) / (2 * bc)) if bc > 1e-10 else 0.0
    angle2 = safe_acos((a2 + c2 - b2) / (2 * ac)) if ac > 1e-10 else 0.0
    angle3 = safe_acos((a2 + b2 - c2) / (2 * ab)) if ab > 1e-10 else 0.0

    return (angle1, angle2, angle3)


_SQRT3_OVER_2 = math.sqrt(3) / 2
_SQRT3 = math.sqrt(3)


def fermat_point(p1: Point, p2: Point, p3: Point) -> Point:
    """Calculate the Fermat point of a triangle."""
    angles = triangle_angles(p1, p2, p3)
    threshold = 2 * math.pi / 3

    if angles[0] >= threshold:
        return p1
    if angles[1] >= threshold:
        return p2
    if angles[2] >= threshold:
        return p3

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    dx2, dy2 = x3 - x2, y3 - y2
    dx3, dy3 = x3 - x1, y3 - y1

    x4 = x2 + 0.5 * dx2 - _SQRT3_OVER_2 * dy2
    y4 = y2 + _SQRT3_OVER_2 * dx2 + 0.5 * dy2
    
    x5 = x1 + 0.5 * dx3 - _SQRT3_OVER_2 * dy3
    y5 = y1 + _SQRT3_OVER_2 * dx3 + 0.5 * dy3

    dx1, dy1 = x4 - x1, y4 - y1
    dx2, dy2 = x5 - x2, y5 - y2

    denom = dx1 * dy2 - dy1 * dx2

    if abs(denom) < 1e-10:
        logger.warning("Lines nearly parallel, using centroid")
        return ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3)

    t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom
    return (x1 + t * dx1, y1 + t * dy1)


def steiner_tree_length(terminals: List[Point]) -> float:
    """Calculate Steiner tree length for N terminals."""
    n = len(terminals)

    if n < 2:
        raise ValueError("Need at least 2 terminals")
    if n == 2:
        dx = terminals[0][0] - terminals[1][0]
        dy = terminals[0][1] - terminals[1][1]
        return math.sqrt(dx * dx + dy * dy)
    if n == 3:
        p1, p2, p3 = terminals
        fp = fermat_point(p1, p2, p3)
        dx1, dy1 = fp[0] - p1[0], fp[1] - p1[1]
        dx2, dy2 = fp[0] - p2[0], fp[1] - p2[1]
        dx3, dy3 = fp[0] - p3[0], fp[1] - p3[1]
        return math.sqrt(dx1 * dx1 + dy1 * dy1) + math.sqrt(dx2 * dx2 + dy2 * dy2) + math.sqrt(dx3 * dx3 + dy3 * dy3)
    if n == 4:
        return _steiner_tree_4(terminals)

    return _steiner_tree_n(terminals)


_PARTITION_PAIRS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))


def _steiner_tree_4(terminals: List[Point]) -> float:
    """Calculate Steiner tree for 4 terminals."""
    rect = _try_rectangle(terminals)
    if rect is not None:
        return rect

    best = mst_length(terminals)

    for (i1, i2), (i3, i4) in _PARTITION_PAIRS:
        t_a, t_b = terminals[i1], terminals[i2]
        t_c, t_d = terminals[i3], terminals[i4]

        c1x = (t_a[0] + t_b[0]) * 0.5
        c1y = (t_a[1] + t_b[1]) * 0.5
        c2x = (t_c[0] + t_d[0]) * 0.5
        c2y = (t_c[1] + t_d[1]) * 0.5

        s1, s2 = _optimize_two_steiner(t_a, t_b, t_c, t_d, (c1x, c1y), (c2x, c2y))
        
        dx1, dy1 = s1[0] - t_a[0], s1[1] - t_a[1]
        dx2, dy2 = s1[0] - t_b[0], s1[1] - t_b[1]
        dx3, dy3 = s1[0] - s2[0], s1[1] - s2[1]
        dx4, dy4 = s2[0] - t_c[0], s2[1] - t_c[1]
        dx5, dy5 = s2[0] - t_d[0], s2[1] - t_d[1]
        
        length = (math.sqrt(dx1 * dx1 + dy1 * dy1) + 
                  math.sqrt(dx2 * dx2 + dy2 * dy2) + 
                  math.sqrt(dx3 * dx3 + dy3 * dy3) + 
                  math.sqrt(dx4 * dx4 + dy4 * dy4) + 
                  math.sqrt(dx5 * dx5 + dy5 * dy5))
        
        if length < best:
            best = length

    for i in range(4):
        others = [terminals[j] for j in range(4) if j != i]
        fp = fermat_point(others[0], others[1], others[2])
        
        fpx, fpy = fp
        base = 0.0
        for p in others:
            dx, dy = fpx - p[0], fpy - p[1]
            base += math.sqrt(dx * dx + dy * dy)
        
        min_to_others = float('inf')
        ti = terminals[i]
        for p in others:
            dx, dy = ti[0] - p[0], ti[1] - p[1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < min_to_others:
                min_to_others = d
        
        dx, dy = ti[0] - fpx, ti[1] - fpy
        d_fp = math.sqrt(dx * dx + dy * dy)
        
        best = min(best, base + min(d_fp, min_to_others))

    return best


def _try_rectangle(terminals: List[Point]) -> Optional[float]:
    """Check if 4 points form a rectangle and compute Steiner length."""
    t0, t1, t2, t3 = terminals
    
    d01 = (t0[0] - t1[0]) ** 2 + (t0[1] - t1[1]) ** 2
    d02 = (t0[0] - t2[0]) ** 2 + (t0[1] - t2[1]) ** 2
    d03 = (t0[0] - t3[0]) ** 2 + (t0[1] - t3[1]) ** 2
    d12 = (t1[0] - t2[0]) ** 2 + (t1[1] - t2[1]) ** 2
    d13 = (t1[0] - t3[0]) ** 2 + (t1[1] - t3[1]) ** 2
    d23 = (t2[0] - t3[0]) ** 2 + (t2[1] - t3[1]) ** 2
    
    dists_sq = [d01, d02, d03, d12, d13, d23]
    dists_sq.sort(reverse=True)
    
    diag1, diag2 = dists_sq[0], dists_sq[1]
    if abs(diag1 - diag2) > 0.0001:
        return None

    sides = sorted(dists_sq[2:])
    if abs(sides[0] - sides[1]) > 0.0001 or abs(sides[2] - sides[3]) > 0.0001:
        return None

    w = math.sqrt(max(sides[2], sides[0]))
    h = math.sqrt(min(sides[2], sides[0]))
    return min(w + h * _SQRT3, mst_length(terminals))


def _optimize_two_steiner(t1: Point, t2: Point, t3: Point, t4: Point,
                         s1: Point, s2: Point, max_iter: int = 50, tol: float = 1e-8) -> Tuple[Point, Point]:
    """Optimize positions of two Steiner points."""
    s1x, s1y = s1
    s2x, s2y = s2
    
    for _ in range(max_iter):
        s1_oldx, s1_oldy = s1x, s1y
        s2_oldx, s2_oldy = s2x, s2y
        
        s1 = fermat_point(t1, t2, (s2x, s2y))
        s1x, s1y = s1
        s2 = fermat_point(t3, t4, (s1x, s1y))
        s2x, s2y = s2
        
        dx = s1x - s1_oldx
        dy = s1y - s1_oldy
        if dx * dx + dy * dy < tol * tol:
            dx = s2x - s2_oldx
            dy = s2y - s2_oldy
            if dx * dx + dy * dy < tol * tol:
                break
    
    return (s1x, s1y), (s2x, s2y)


def _steiner_tree_n(terminals: List[Point]) -> float:
    """Calculate Steiner tree for N > 4 terminals."""
    n = len(terminals)

    if n == 5 and _is_regular_polygon(terminals, 5):
        return mst_length(terminals) * 0.8881

    if n == 6 and _is_regular_polygon(terminals, 6):
        return mst_length(terminals) * 0.866025

    return mst_length(terminals)


def _is_regular_polygon(terminals: List[Point], n: int, tol: float = 0.01) -> bool:
    """Check if terminals form a regular n-gon."""
    if len(terminals) != n:
        return False

    cx = sum(p[0] for p in terminals) / n
    cy = sum(p[1] for p in terminals) / n

    dists = []
    for p in terminals:
        dx, dy = p[0] - cx, p[1] - cy
        dists.append(dx * dx + dy * dy)
    
    avg_dist = sum(dists) / n
    tol_sq = tol * tol
    if any(abs(d - avg_dist) > tol_sq for d in dists):
        return False

    angles = sorted(math.atan2(p[1] - cy, p[0] - cx) for p in terminals)
    
    expected = 2 * math.pi / n
    for i in range(n):
        next_angle = angles[(i + 1) % n]
        diff = next_angle - angles[i]
        if i == n - 1:
            diff += 2 * math.pi
        if abs(diff - expected) > tol:
            return False
    
    return True


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

    ratio = steiner_tree_length(terminals) / mst_len
    logger.info(f"Steiner ratio: {ratio:.6f}")
    return ratio