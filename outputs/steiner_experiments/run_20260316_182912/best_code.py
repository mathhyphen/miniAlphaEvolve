"""Fermat Point Baseline for Steiner Tree Problem.

This module provides baseline seed code for evolving Steiner Tree solutions.
It implements the classic Fermat point solution for 3-terminal cases.

The Fermat point (also called Torricelli point) is the point that minimizes
the total distance to the three vertices of a triangle. For triangles with
all angles < 120 degrees, it is the point where each side subtends an angle
of 120 degrees.

TODO: Evolution directions for this baseline:
1. Extend to N-terminal cases using iterative Fermat point insertion
2. Add genetic operators for evolving Steiner point positions
3. Implement local search refinement around Fermat points
4. Add topology optimization (which terminals connect to which Steiner points)
5. Implement full Steiner tree evaluation with arbitrary topologies
6. Add gradient-based refinement for Steiner point positions
7. Support degenerate cases (collinear points, coincident points)
"""

import math
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


def distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        p1: First point as (x, y) tuple.
        p2: Second point as (x, y) tuple.

    Returns:
        Euclidean distance between p1 and p2.

    Examples:
        >>> distance((0, 0), (3, 4))
        5.0
        >>> distance((0, 0), (1, 1))
        1.4142135623730951
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def mst_length(points: List[Point]) -> float:
    """Calculate Minimum Spanning Tree length using Prim's algorithm.

    This provides an upper bound on the Steiner tree length.
    The Steiner ratio is defined as: optimal_steiner_length / mst_length

    For 3 points forming an equilateral triangle:
    - MST length = 2 * side_length
    - Steiner length = sqrt(3) * side_length
    - Steiner ratio = sqrt(3) / 2 ≈ 0.866

    Args:
        points: List of terminal points as (x, y) tuples.

    Returns:
        Total length of the minimum spanning tree.

    Raises:
        ValueError: If fewer than 2 points provided.

    Examples:
        >>> import math
        >>> # Equilateral triangle with side length 1
        >>> pts = [(0, 0), (1, 0), (0.5, math.sqrt(3)/2)]
        >>> mst_length(pts)
        2.0
    """
    if len(points) < 2:
        raise ValueError("MST requires at least 2 points")

    n = len(points)
    if n == 2:
        return distance(points[0], points[1])

    # Prim's algorithm
    visited = [False] * n
    min_dist = [float('inf')] * n
    min_dist[0] = 0  # Start from first point

    total_length = 0.0
    visited_count = 0

    while visited_count < n:
        # Find minimum distance vertex among unvisited
        u = -1
        for i in range(n):
            if not visited[i] and (u == -1 or min_dist[i] < min_dist[u]):
                u = i

        visited[u] = True
        visited_count += 1
        total_length += min_dist[u]

        # Update distances to neighbors
        for v in range(n):
            if not visited[v]:
                d = distance(points[u], points[v])
                if d < min_dist[v]:
                    min_dist[v] = d

    return total_length


def triangle_angles(p1: Point, p2: Point, p3: Point) -> Tuple[float, float, float]:
    """Calculate the three angles of a triangle using law of cosines.

    Uses the law of cosines to compute angles:
    cos(A) = (b² + c² - a²) / (2bc)

    where a, b, c are side lengths opposite to angles A, B, C respectively.

    Args:
        p1: First vertex.
        p2: Second vertex.
        p3: Third vertex.

    Returns:
        Tuple of three angles (angle_at_p1, angle_at_p2, angle_at_p3) in radians.
        Each angle is in range [0, pi].

    Examples:
        >>> import math
        >>> # Equilateral triangle
        >>> angles = triangle_angles((0, 0), (1, 0), (0.5, math.sqrt(3)/2))
        >>> all(abs(a - math.pi/3) < 0.001 for a in angles)
        True

        >>> # Right triangle
        >>> angles = triangle_angles((0, 0), (1, 0), (0, 1))
        >>> abs(angles[0] - math.pi/2) < 0.001  # 90 degrees at origin
        True
    """
    # Calculate side lengths
    a = distance(p2, p3)  # Side opposite to p1
    b = distance(p1, p3)  # Side opposite to p2
    c = distance(p1, p2)  # Side opposite to p3

    def safe_acos(x: float) -> float:
        """Safely compute arccos with clamping to [-1, 1]."""
        return math.acos(max(-1.0, min(1.0, x)))

    # Law of cosines
    # Avoid division by zero for degenerate triangles
    if b * c < 1e-10:
        angle1 = 0.0
    else:
        angle1 = safe_acos((b * b + c * c - a * a) / (2 * b * c))

    if a * c < 1e-10:
        angle2 = 0.0
    else:
        angle2 = safe_acos((a * a + c * c - b * b) / (2 * a * c))

    if a * b < 1e-10:
        angle3 = 0.0
    else:
        angle3 = safe_acos((a * a + b * b - c * c) / (2 * a * b))

    return (angle1, angle2, angle3)


def fermat_point(p1: Point, p2: Point, p3: Point) -> Point:
    """Calculate the Fermat point of a triangle.

    The Fermat point minimizes the sum of distances to the three vertices.

    For triangles with all angles < 120 degrees:
    - The Fermat point is inside the triangle
    - Each side subtends 120 degrees from the Fermat point
    - Can be found as intersection of lines from vertices

    For triangles with an angle >= 120 degrees:
    - The Fermat point is the obtuse vertex

    Algorithm for < 120 degrees case:
    1. Construct equilateral triangle on one side (outward)
    2. Line from opposite vertex to new vertex passes through Fermat point
    3. Repeat for another side
    4. Intersection is the Fermat point

    Args:
        p1: First vertex.
        p2: Second vertex.
        p3: Third vertex.

    Returns:
        The Fermat point as (x, y) tuple.

    Examples:
        >>> import math
        >>> # Equilateral triangle - Fermat point is centroid
        >>> fp = fermat_point((0, 0), (1, 0), (0.5, math.sqrt(3)/2))
        >>> abs(fp[0] - 0.5) < 0.01 and abs(fp[1] - math.sqrt(3)/6) < 0.01
        True

        >>> # Right triangle with angle < 120 at all vertices
        >>> fp = fermat_point((0, 0), (1, 0), (0, 1))
        >>> fp  # Should be inside the triangle
        (0.21..., 0.21...)
    """
    angles = triangle_angles(p1, p2, p3)

    # Check if any angle >= 120 degrees (2*pi/3 radians)
    threshold = 2 * math.pi / 3

    if angles[0] >= threshold:
        return p1
    if angles[1] >= threshold:
        return p2
    if angles[2] >= threshold:
        return p3

    # All angles < 120 degrees - use geometric construction
    # Construct equilateral triangle on side p2-p3 (outward from p1)
    # Then find intersection of line p1->new_vertex with similar line from another side

    def rotate_60_degrees(a: Point, b: Point) -> Point:
        """Rotate point b around point a by 60 degrees counterclockwise."""
        cos60 = 0.5
        sin60 = math.sqrt(3) / 2
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return (
            a[0] + dx * cos60 - dy * sin60,
            a[1] + dx * sin60 + dy * cos60
        )

    # Construct equilateral triangle on side p2-p3
    p4 = rotate_60_degrees(p2, p3)

    # Construct equilateral triangle on side p1-p3
    p5 = rotate_60_degrees(p1, p3)

    # Line p1->p4 and line p2->p5 intersect at Fermat point
    # Use line intersection formula
    # Line 1: p1 + t * (p4 - p1)
    # Line 2: p2 + s * (p5 - p2)

    x1, y1 = p1
    x2, y2 = p4
    x3, y3 = p2
    x4, y4 = p5

    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3

    denom = dx1 * dy2 - dy1 * dx2

    if abs(denom) < 1e-10:
        # Lines are nearly parallel - use centroid as fallback
        logger.warning("Lines nearly parallel, using centroid")
        return ((p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3)

    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / denom

    fermat_x = x1 + t * dx1
    fermat_y = y1 + t * dy1

    return (fermat_x, fermat_y)


def steiner_tree_length(terminals: List[Point]) -> float:
    """Calculate Steiner tree length for 3 terminals using Fermat point.

    For 3 terminals, the optimal Steiner tree consists of:
    - One Steiner point (the Fermat point)
    - Three edges connecting the Fermat point to each terminal

    This is ONLY correct for 3 terminals. For N > 3 terminals, the optimal
    topology may require multiple Steiner points.

    TODO: Extend to N terminals:
    - Option 1: Iterative Fermat point insertion
    - Option 2: Full topology search with dynamic programming
    - Option 3: Use GeoSteiner solver for exact solutions

    Args:
        terminals: List of terminal points (must be exactly 3 for this implementation).

    Returns:
        Total length of the Steiner tree.

    Raises:
        ValueError: If not exactly 3 terminals.

    Examples:
        >>> import math
        >>> # Equilateral triangle with side length 1
        >>> pts = [(0, 0), (1, 0), (0.5, math.sqrt(3)/2)]
        >>> steiner_tree_length(pts)
        1.7320508075688772  # sqrt(3)
    """
    if len(terminals) != 3:
        raise ValueError(
            f"Currently only supports exactly 3 terminals, got {len(terminals)}. "
            "TODO: Extend to N terminals using iterative methods."
        )

    p1, p2, p3 = terminals
    fp = fermat_point(p1, p2, p3)

    total = distance(fp, p1) + distance(fp, p2) + distance(fp, p3)

    logger.debug(f"Steiner tree: Fermat point at {fp}, total length = {total}")

    return total


def find_steiner_ratio(terminals: List[Point]) -> float:
    """Calculate the Steiner ratio for given terminals.

    The Steiner ratio is defined as:
        rho = L_optimal_steiner / L_MST

    For 3 points forming an equilateral triangle:
        rho = sqrt(3) / 2 ≈ 0.866025...

    This is the minimum possible Steiner ratio (Gilbert-Pollak theorem).

    The Steiner ratio measures the potential improvement from using Steiner
    points. A ratio of 1.0 means Steiner points don't help (e.g., 2 points,
    or collinear points). A ratio closer to sqrt(3)/2 means significant
    improvement is possible.

    Args:
        terminals: List of terminal points.

    Returns:
        Steiner ratio (value in range [sqrt(3)/2, 1.0]).

    Raises:
        ValueError: If fewer than 2 terminals.

    Examples:
        >>> import math
        >>> # Equilateral triangle - minimum Steiner ratio
        >>> pts = [(0, 0), (1, 0), (0.5, math.sqrt(3)/2)]
        >>> ratio = find_steiner_ratio(pts)
        >>> abs(ratio - math.sqrt(3)/2) < 0.01
        True

        >>> # Collinear points - Steiner ratio = 1.0
        >>> pts = [(0, 0), (1, 0), (2, 0)]
        >>> abs(find_steiner_ratio(pts) - 1.0) < 0.01
        True
    """
    if len(terminals) < 2:
        raise ValueError("Need at least 2 terminals")

    if len(terminals) == 2:
        # For 2 points, Steiner tree = MST (no Steiner point needed)
        return 1.0

    # For 3 points, use Fermat point
    steiner_length = steiner_tree_length(terminals)
    mst_len = mst_length(terminals)

    if mst_len < 1e-10:
        # All points coincident
        return 1.0

    ratio = steiner_length / mst_len
    logger.info(f"Steiner ratio: {ratio:.6f} (Steiner={steiner_length:.4f}, MST={mst_len:.4f})")

    return ratio


# =============================================================================
# TODO: Evolution extension points
# =============================================================================
#
# The following functions are placeholders for future evolution:
#
# 1. N-terminal Steiner tree:
#    def steiner_tree_n(terminals: List[Point], max_steiner_points: int) -> float:
#        """Compute Steiner tree for N terminals with up to K Steiner points."""
#        # TODO: Implement iterative Fermat point insertion
#        # TODO: Explore different topologies
#        pass
#
# 2. Gradient-based refinement:
#    def refine_steiner_points(terminals: List[Point],
#                              steiner_points: List[Point]) -> List[Point]:
#        """Refine Steiner point positions using gradient descent."""
#        # TODO: Implement Weiszfeld's algorithm or similar
#        pass
#
# 3. Topology optimization:
#    def optimize_topology(terminals: List[Point]) -> List[Tuple[int, int]]:
#        """Find optimal connection topology for Steiner tree."""
#        # TODO: Enumerate full Steiner topologies
#        # TODO: Use dynamic programming for optimal topology
#        pass
#
# 4. Genetic operators:
#    def mutate_steiner_points(points: List[Point],
#                              mutation_rate: float) -> List[Point]:
#        """Apply mutation to Steiner point positions."""
#        # TODO: Add Gaussian perturbation
#        # TODO: Add topology-changing mutations
#        pass
#
# 5. Local search:
#    def local_search(terminals: List[Point],
#                     initial_steiner_points: List[Point]) -> List[Point]:
#        """Perform local search to improve Steiner point positions."""
#        # TODO: Implement hill climbing with adaptive step size
#        # TODO: Add simulated annealing for escaping local optima
#        pass
#
# =============================================================================