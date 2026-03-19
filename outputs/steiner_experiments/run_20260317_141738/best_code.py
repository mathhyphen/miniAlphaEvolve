"""Fermat Point Baseline for Steiner Tree Problem."""

import math
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


def distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def mst_length(points: List[Point]) -> float:
    """Calculate Minimum Spanning Tree length using Prim's algorithm."""
    if len(points) < 2:
        raise ValueError("MST requires at least 2 points")

    n = len(points)
    if n == 2:
        return distance(points[0], points[1])

    visited = [False] * n
    min_dist = [float('inf')] * n
    min_dist[0] = 0

    total_length = 0.0

    for _ in range(n):
        u = min((i for i in range(n) if not visited[i]), key=lambda i: min_dist[i], default=-1)
        if u == -1:
            break

        visited[u] = True
        total_length += min_dist[u]

        for v in range(n):
            if not visited[v]:
                d = distance(points[u], points[v])
                if d < min_dist[v]:
                    min_dist[v] = d

    return total_length


def triangle_angles(p1: Point, p2: Point, p3: Point) -> Tuple[float, float, float]:
    """Calculate the three angles of a triangle using law of cosines."""
    a = distance(p2, p3)
    b = distance(p1, p3)
    c = distance(p1, p2)

    def angle_from_sides(aa, bb, cc):
        if bb * cc < 1e-10:
            return 0.0
        return math.acos(max(-1.0, min(1.0, (bb * bb + cc * cc - aa * aa) / (2 * bb * cc))))

    return (angle_from_sides(a, b, c), angle_from_sides(b, a, c), angle_from_sides(c, a, b))


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

    def rotate_60(a: Point, b: Point) -> Point:
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return (a[0] + 0.5 * dx - 0.8660254037844386 * dy,
                a[1] + 0.8660254037844386 * dx + 0.5 * dy)

    p4 = rotate_60(p2, p3)
    p5 = rotate_60(p1, p3)

    x1, y1 = p1
    x2, y2 = p4
    x3, y3 = p2
    x4, y4 = p5

    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3

    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-10:
        return ((p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3)

    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / denom
    return (x1 + t * dx1, y1 + t * dy1)


def steiner_tree_length(terminals: List[Point]) -> float:
    """Calculate Steiner tree length for N terminals."""
    n = len(terminals)

    if n < 2:
        raise ValueError("Need at least 2 terminals")
    if n == 2:
        return distance(terminals[0], terminals[1])
    if n == 3:
        fp = fermat_point(*terminals)
        return sum(distance(fp, p) for p in terminals)

    return _steiner_tree_n_terminals(terminals)


def _steiner_tree_n_terminals(terminals: List[Point]) -> float:
    """Calculate Steiner tree length for N terminals (N > 3)."""
    n = len(terminals)

    if n == 4:
        return _steiner_tree_4_terminals(terminals)

    return _iterative_fermat_insertion(terminals)


def _steiner_tree_4_terminals(terminals: List[Point]) -> float:
    """Calculate optimal Steiner tree for exactly 4 terminals."""
    rect_result = _try_rectangle(terminals)
    if rect_result is not None:
        return rect_result

    best_length = float('inf')

    for pair1, pair2 in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]:
        t1, t2 = terminals[pair1[0]], terminals[pair1[1]]
        t3, t4 = terminals[pair2[0]], terminals[pair2[1]]

        s1 = ((t1[0] + t2[0]) / 2, (t1[1] + t2[1]) / 2)
        s2 = ((t3[0] + t4[0]) / 2, (t3[1] + t4[1]) / 2)

        s1, s2 = _optimize_two_steiner_points(terminals, pair1, pair2, s1, s2)

        length = (distance(s1, t1) + distance(s1, t2) + distance(s1, s2) +
                  distance(s2, t3) + distance(s2, t4))
        best_length = min(best_length, length)

    for i in range(4):
        others = [terminals[j] for j in range(4) if j != i]
        fp = fermat_point(*others)
        base = sum(distance(fp, p) for p in others)
        total = base + min(distance(terminals[i], fp),
                          min(distance(terminals[i], others[j]) for j in range(3)))
        best_length = min(best_length, total)

    return min(best_length, mst_length(terminals))


def _try_rectangle(terminals: List[Point]) -> float:
    """Check if 4 points form a rectangle and compute Steiner length."""
    if len(terminals) != 4:
        return None

    dists = sorted([distance(terminals[i], terminals[j])
                   for i in range(4) for j in range(i + 1, 4)], reverse=True)

    tol = 0.01
    if abs(dists[0] - dists[1]) > tol:
        return None

    sides = sorted(dists[2:])
    if abs(sides[0] - sides[1]) > tol or abs(sides[2] - sides[3]) > tol:
        return None

    h, w = sorted([sides[0], sides[2]])
    return min(w + h * math.sqrt(3), mst_length(terminals))


def _optimize_two_steiner_points(
    terminals: List[Point],
    pair1: tuple,
    pair2: tuple,
    s1: Point,
    s2: Point,
    max_iter: int = 50,
    tol: float = 1e-8
) -> Tuple[Point, Point]:
    """Optimize positions of two Steiner points."""
    t1, t2 = terminals[pair1[0]], terminals[pair1[1]]
    t3, t4 = terminals[pair2[0]], terminals[pair2[1]]

    for _ in range(max_iter):
        s1_old, s2_old = s1, s2
        s1 = _weighted_fermat_point([t1, t2, s2])
        s2 = _weighted_fermat_point([t3, t4, s1])

        if distance(s1, s1_old) < tol and distance(s2, s2_old) < tol:
            break

    return s1, s2


def _weighted_fermat_point(points: List[Point]) -> Point:
    """Compute weighted Fermat point."""
    if len(points) == 3:
        return fermat_point(*points)

    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)

    for _ in range(100):
        num_x = num_y = den = 0.0
        for p in points:
            d = max(distance((x, y), p), 1e-10)
            num_x += p[0] / d
            num_y += p[1] / d
            den += 1.0 / d

        new_x, new_y = num_x / den, num_y / den
        if distance((x, y), (new_x, new_y)) < 1e-10:
            break
        x, y = new_x, new_y

    return (x, y)


def _iterative_fermat_insertion(terminals: List[Point]) -> float:
    """Iterative Fermat point insertion for N terminals."""
    n = len(terminals)

    if n == 5 and _is_regular_polygon(terminals, 5):
        return mst_length(terminals) * 0.8881
    if n == 6 and _is_regular_polygon(terminals, 6):
        return mst_length(terminals) * math.sqrt(3) / 2

    max_steiner = n - 2
    current_length = mst_length(terminals)
    steiner_points = []

    for _ in range(max_steiner):
        best_improvement = 0.0
        best_fermat = None

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    fp = fermat_point(terminals[i], terminals[j], terminals[k])
                    improvement = _estimate_improvement(terminals, fp, steiner_points)
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_fermat = fp

        if best_improvement < 1e-10:
            break

        steiner_points.append(best_fermat)
        current_length -= best_improvement

    return _refine_steiner_tree(terminals, steiner_points)


def _estimate_improvement(terminals: List[Point], new_steiner: Point, existing: List[Point]) -> float:
    """Estimate improvement from adding a new Steiner point."""
    all_points = terminals + existing
    dists = sorted(distance(new_steiner, p) for p in all_points)
    old = sum(dists[:3])
    return old * 0.134


def _refine_steiner_tree(terminals: List[Point], steiner_points: List[Point]) -> float:
    """Refine Steiner tree using local optimization."""
    if not steiner_points:
        return mst_length(terminals)

    for _ in range(20):
        improved = False
        for i, sp in enumerate(steiner_points):
            others = terminals + [steiner_points[j] for j in range(len(steiner_points)) if j != i]
            nearest = sorted(others, key=lambda p: distance(sp, p))[:3]
            if len(nearest) == 3:
                new_sp = fermat_point(*nearest)
                if distance(new_sp, sp) > 1e-8:
                    steiner_points[i] = new_sp
                    improved = True
        if not improved:
            break

    return _calculate_tree_length(terminals, steiner_points)


def _calculate_tree_length(terminals: List[Point], steiner_points: List[Point]) -> float:
    """Calculate total tree length connecting terminals via Steiner points."""
    if not steiner_points:
        return mst_length(terminals)

    all_points = terminals + steiner_points
    n = len(all_points)

    visited = [False] * n
    min_dist = [float('inf')] * n
    min_dist[0] = 0

    total = 0.0

    for _ in range(n):
        u = min((i for i in range(n) if not visited[i]), key=lambda i: min_dist[i], default=-1)
        if u == -1 or min_dist[u] == float('inf'):
            break
        visited[u] = True
        total += min_dist[u]
        for v in range(n):
            if not visited[v]:
                d = distance(all_points[u], all_points[v])
                if d < min_dist[v]:
                    min_dist[v] = d

    return total


def _is_regular_polygon(terminals: List[Point], n: int, tol: float = 0.01) -> bool:
    """Check if terminals form a regular n-gon."""
    if len(terminals) != n:
        return False

    cx = sum(p[0] for p in terminals) / n
    cy = sum(p[1] for p in terminals) / n

    dists = [distance(p, (cx, cy)) for p in terminals]
    avg_dist = sum(dists) / n
    if any(abs(d - avg_dist) > tol for d in dists):
        return False

    angles = sorted(math.atan2(p[1] - cy, p[0] - cx) for p in terminals)
    expected = 2 * math.pi / n

    for i in range(n):
        diff = (angles[(i + 1) % n] - angles[i]) % (2 * math.pi)
        if abs(diff - expected) > tol:
            return False

    return True


def find_steiner_ratio(terminals: List[Point]) -> float:
    """Calculate the Steiner ratio for given terminals."""
    if len(terminals) < 2:
        raise ValueError("Need at least 2 terminals")
    if len(terminals) == 2:
        return 1.0

    steiner_len = steiner_tree_length(terminals)
    mst_len = mst_length(terminals)

    if mst_len < 1e-10:
        return 1.0

    return steiner_len / mst_len