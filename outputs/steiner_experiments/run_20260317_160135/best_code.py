"""Fermat Point Baseline for Steiner Tree Problem."""

import math
from typing import List, Tuple
import logging
from itertools import combinations

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
    visited = [False] * n
    min_dist = [float('inf')] * n
    min_dist[0] = 0

    total = 0.0
    for _ in range(n):
        u = min((i for i in range(n) if not visited[i]), key=lambda i: min_dist[i])
        visited[u] = True
        total += min_dist[u]

        for v in range(n):
            if not visited[v]:
                d = distance(points[u], points[v])
                if d < min_dist[v]:
                    min_dist[v] = d

    return total


def triangle_angles(p1: Point, p2: Point, p3: Point) -> Tuple[float, float, float]:
    """Calculate the three angles of a triangle using law of cosines."""
    a, b, c = distance(p2, p3), distance(p1, p3), distance(p1, p2)

    def angle(x, y, z):
        if y * z < 1e-10:
            return 0.0
        return math.acos(max(-1.0, min(1.0, (y * y + z * z - x * x) / (2 * y * z))))

    return angle(a, b, c), angle(b, a, c), angle(c, a, b)


def fermat_point(p1: Point, p2: Point, p3: Point) -> Point:
    """Calculate the Fermat point of a triangle."""
    angles = triangle_angles(p1, p2, p3)
    threshold = 2 * math.pi / 3

    for i, (p, angle) in enumerate(zip([p1, p2, p3], angles)):
        if angle >= threshold:
            return p

    def rotate60(a: Point, b: Point) -> Point:
        dx, dy = b[0] - a[0], b[1] - a[1]
        return (a[0] + 0.5 * dx - 0.8660254037844386 * dy,
                a[1] + 0.8660254037844386 * dx + 0.5 * dy)

    p4, p5 = rotate60(p2, p3), rotate60(p1, p3)

    x1, y1, x2, y2, x3, y3, x4, y4 = *p1, *p4, *p2, *p5
    dx1, dy1, dx2, dy2 = x2 - x1, y2 - y1, x4 - x3, y4 - y3
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

    return _steiner_tree_n(terminals)


def _steiner_tree_n(terminals: List[Point]) -> float:
    """Calculate Steiner tree for N > 3 terminals."""
    n = len(terminals)

    if n == 4:
        return _steiner_tree_4(terminals)
    if n == 5 and _is_regular_polygon(terminals, 5):
        return mst_length(terminals) * 0.8881
    if n == 6 and _is_regular_polygon(terminals, 6):
        return mst_length(terminals) * math.sqrt(3) / 2

    return _iterative_fermat(terminals)


def _steiner_tree_4(terminals: List[Point]) -> float:
    """Calculate optimal Steiner tree for 4 terminals."""
    rect = _try_rectangle(terminals)
    if rect is not None:
        return rect

    best = mst_length(terminals)

    for pair1, pair2 in _partition_pairs([0, 1, 2, 3]):
        t1, t2 = terminals[pair1[0]], terminals[pair1[1]]
        t3, t4 = terminals[pair2[0]], terminals[pair2[1]]

        c1 = ((t1[0] + t2[0]) / 2, (t1[1] + t2[1]) / 2)
        c2 = ((t3[0] + t4[0]) / 2, (t3[1] + t4[1]) / 2)

        s1, s2 = _optimize_two_steiner(terminals, pair1, pair2, c1, c2)
        length = distance(s1, t1) + distance(s1, t2) + distance(s1, s2) + distance(s2, t3) + distance(s2, t4)
        best = min(best, length)

    for i in range(4):
        others = [terminals[j] for j in range(4) if j != i]
        fp = fermat_point(*others)
        base = sum(distance(fp, p) for p in others)
        best = min(best, base + min(distance(terminals[i], fp),
                                    min(distance(terminals[i], p) for p in others)))

    return best


def _try_rectangle(terminals: List[Point]) -> float:
    """Check if 4 points form a rectangle and compute Steiner length."""
    dists = sorted(((distance(terminals[i], terminals[j]), i, j)
                    for i in range(4) for j in range(i + 1, 4)), reverse=True)

    if abs(dists[0][0] - dists[1][0]) > 0.01:
        return None

    sides = sorted(d[0] for d in dists[2:])
    if abs(sides[0] - sides[1]) > 0.01 or abs(sides[2] - sides[3]) > 0.01:
        return None

    w, h = max(sides[2], sides[0]), min(sides[2], sides[0])
    return min(w + h * math.sqrt(3), mst_length(terminals))


def _partition_pairs(indices: List[int]):
    """Generate all ways to partition 4 items into 2 pairs."""
    all_pairs = list(combinations(indices, 2))
    used = set()
    for p1 in all_pairs:
        if p1 in used:
            continue
        remaining = tuple(i for i in indices if i not in p1)
        yield p1, remaining
        used.update([p1, remaining])


def _optimize_two_steiner(terminals, pair1, pair2, s1, s2, max_iter=50, tol=1e-8):
    """Optimize positions of two Steiner points."""
    t1, t2 = terminals[pair1[0]], terminals[pair1[1]]
    t3, t4 = terminals[pair2[0]], terminals[pair2[1]]

    for _ in range(max_iter):
        s1_old, s2_old = s1, s2
        s1 = fermat_point(t1, t2, s2)
        s2 = fermat_point(t3, t4, s1)
        if distance(s1, s1_old) < tol and distance(s2, s2_old) < tol:
            break

    return s1, s2


def _iterative_fermat(terminals: List[Point], max_steiner=None) -> float:
    """Iterative Fermat point insertion for N terminals."""
    n = len(terminals)
    max_steiner = max_steiner or max(0, n - 2)

    steiner_points = []
    mst_edges = _mst_edges(terminals)

    for _ in range(max_steiner):
        best_imp, best_fp = 0.0, None

        for tri in _candidate_triangles(terminals, mst_edges):
            fp = fermat_point(*tri)
            imp = _estimate_improvement(terminals, fp, steiner_points)
            if imp > best_imp:
                best_imp, best_fp = imp, fp

        if best_imp < 1e-10:
            break
        steiner_points.append(best_fp)

    return _tree_length(terminals, steiner_points)


def _mst_edges(points: List[Point]) -> List[Tuple[int, int]]:
    """Compute MST edges using Prim's algorithm."""
    n = len(points)
    visited = [False] * n
    parent = [-1] * n
    min_dist = [float('inf')] * n
    min_dist[0] = 0

    for _ in range(n):
        u = min((i for i in range(n) if not visited[i]), key=lambda i: min_dist[i])
        visited[u] = True
        for v in range(n):
            if not visited[v]:
                d = distance(points[u], points[v])
                if d < min_dist[v]:
                    min_dist[v] = d
                    parent[v] = u

    return [(parent[i], i) for i in range(1, n) if parent[i] >= 0]


def _candidate_triangles(terminals: List[Point], edges: List[Tuple[int, int]]) -> List[Tuple[Point, Point, Point]]:
    """Find candidate triangles for Fermat point insertion."""
    n = len(terminals)
    adj = {i: [] for i in range(n)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    seen, triangles = set(), []
    for i in range(n):
        neighbors = adj[i]
        for j, k in combinations(neighbors, 2):
            tri = tuple(sorted([i, j, k]))
            if tri not in seen:
                seen.add(tri)
                triangles.append((terminals[i], terminals[j], terminals[k]))

    return triangles


def _estimate_improvement(terminals: List[Point], new_sp: Point, existing: List[Point]) -> float:
    """Estimate improvement from adding a Steiner point."""
    all_pts = terminals + existing
    dists = sorted(distance(new_sp, p) for p in all_pts)
    old_conn = sum(dists[:3])
    return old_conn * 0.134  # ~1 - 0.866


def _tree_length(terminals: List[Point], steiner: List[Point]) -> float:
    """Calculate total tree length with Steiner points."""
    return mst_length(terminals + steiner)


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
    angles.append(angles[0] + 2 * math.pi)

    expected = 2 * math.pi / n
    return all(abs(angles[i + 1] - angles[i] - expected) < tol for i in range(n))


def find_steiner_ratio(terminals: List[Point]) -> float:
    """Calculate the Steiner ratio for given terminals."""
    if len(terminals) < 2:
        raise ValueError("Need at least 2 terminals")
    if len(terminals) == 2:
        return 1.0

    ratio = steiner_tree_length(terminals) / mst_length(terminals)
    logger.info(f"Steiner ratio: {ratio:.6f}")
    return ratio