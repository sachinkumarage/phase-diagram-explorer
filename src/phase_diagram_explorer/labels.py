"""Textbook phase names and label placement on phase diagrams.

Display names
-------------
A system may give textbook names for its CALPHAD phases in its .meta.json
("display_names"), e.g. {"LIQUID": "L", "FCC_A1": "(Pb)"}. A phase with a
miscibility gap whose two sides are different terminal solutions gets one
name per side: {"FCC_A1": {"base_side": "(Ag)", "dependent_side": "(Cu)"}}.
A composition set is named by the side of the gap it lies on (its
composition against the gap's midpoint at that temperature, or 0.5 if there
is no gap there), not by its #1/#2 number. A phase that forms separate
fields at low and high temperature gets one name per temperature region:
{"BCC_A2": {"below_k": 1400, "below": "alpha (ferrite)", "above": "delta
(ferrite)"}}. Phases without a display name keep their CALPHAD label.

Label placement
---------------
Labels are laid out in a nominal pixel model of the plot area (PlotBox):
text boxes are CHAR_WIDTH_PX per character by LINE_HEIGHT_PX. A field
label sits at the centroid of the field (clipped to the displayed
composition range) if its box fits inside the field, else at
the nearest position inside the field where it fits; otherwise it is placed
outside with a leader line to the field, avoiding boundary lines where
possible. An invariant
temperature label sits just above its line, moved along it or further up
(with a leader) until its box crosses no boundary line. No box leaves the
plot area. Labels of marked lines (A1, A3, Acm) are placed like field
labels that do not fit, with a leader to the line; labels beside a vertical
composition line ("Steels", "Cast irons") sit at the bottom of the plot.
"""
from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as PolygonPath

from phase_diagram_explorer.equilibrium.equilibrium import base_phase_name
from phase_diagram_explorer.tracing import TracedDiagram, TracedField

BASE_SIDE = "base_side"
BELOW_K = "below_k"
DEPENDENT_SIDE = "dependent_side"
CHAR_WIDTH_PX = 7.0
LINE_HEIGHT_PX = 16.0
PADDING_PX = 2.0
# Distances (px) tried for labels placed outside their field, and directions.
LEADER_DISTANCES_PX = (34.0, 52.0, 74.0, 100.0, 130.0)
LEADER_DIRECTIONS = ((0, -1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (1, 1), (-1, 1))
# Spacing (px) of the candidate positions tried inside a field.
INTERIOR_STEP_PX = 6.0
INVARIANT_GAP_PX = 6.0
INVARIANT_STEPS_PX = (0.0, 25.0, -25.0, 50.0, -50.0, 80.0, -80.0, 120.0, -120.0, 170.0, -170.0)
INVARIANT_RAISES_PX = (0.0, 14.0, 28.0, 45.0)


# --- display names -------------------------------------------------------------


class PhaseNames:
    """Textbook names for a system's phases (see module docstring)."""

    def __init__(self, display_names: dict | None = None):
        self.display_names = dict(display_names or {})

    def __bool__(self) -> bool:
        return bool(self.display_names)

    def name(self, label: str, x: float | None = None, split: float = 0.5, T: float | None = None) -> str:
        """Display name of a phase (or composition-set) label at composition x
        and temperature T (K)."""
        entry = self.display_names.get(base_phase_name(label))
        if entry is None:
            return label
        if isinstance(entry, str):
            return entry
        if BELOW_K in entry:
            return entry["below"] if T is None or T < entry[BELOW_K] else entry["above"]
        side = BASE_SIDE if x is not None and x < split else DEPENDENT_SIDE
        return entry[side]

    def names(self, labels_and_x, split: float = 0.5, T: float | None = None) -> list[str]:
        """Display names for the phases of one equilibrium at T: composition
        sets of one phase are named by side, the lower-x set on the base side."""
        items = list(labels_and_x)
        result = []
        for label, x in items:
            base = base_phase_name(label)
            entry = self.display_names.get(base)
            siblings = [other_x for other, other_x in items if base_phase_name(other) == base]
            if len(siblings) > 1 and isinstance(entry, dict) and BASE_SIDE in entry:
                side = BASE_SIDE if x == min(siblings) else DEPENDENT_SIDE
                result.append(entry[side])
            else:
                result.append(self.name(label, x, split, T))
        return result

    def legend(self, display: str, calphad: str) -> str:
        """Legend entry with both names: "(Ag) FCC_A1" for a single phase,
        "L + (Ag) (LIQUID + FCC_A1)" for a two-phase field."""
        if display == calphad:
            return calphad
        return f"{display} ({calphad})" if " + " in calphad else f"{display} {calphad}"

    def reaction(self, reaction_type: str, names: list[str]) -> str:
        """e.g. "L -> (Ag) + (Cu)" (eutectic-type) or "L + (Cu) -> (Ag)" (peritectic-type)."""
        p1, p2, p3 = names
        if reaction_type in ("eutectic", "eutectoid"):
            return f"{p2} -> {p1} + {p3}"
        return f"{p1} + {p3} -> {p2}"


def gap_split(traced: TracedDiagram | None, base: str, T: float) -> float:
    """Midpoint of `base`'s miscibility gap at T (0.5 if there is none)."""
    if traced is not None:
        for strip in traced.fields:
            if strip.phases == (base, base) and strip.T[0] <= T <= strip.T[-1]:
                lo = float(np.interp(T, strip.T, strip.x_min))
                hi = float(np.interp(T, strip.T, strip.x_max))
                return (lo + hi) / 2.0
    return 0.5


def field_names(names: PhaseNames, strip: TracedField, traced: TracedDiagram | None = None) -> tuple[str, str]:
    """(display label, CALPHAD label) of a traced field, e.g. ("L + (Ag)",
    "LIQUID + FCC_A1"). The side of each phase is taken at the field's
    widest level."""
    calphad = strip.label
    if not names:
        return calphad, calphad
    widths = np.asarray(strip.x_max) - np.asarray(strip.x_min)
    k = int(np.argmax(widths)) if len(widths) else 0
    T = strip.T[k]
    if not strip.is_two_phase:
        x = (strip.x_min[k] + strip.x_max[k]) / 2.0
        phase = strip.phases[0]
        return names.name(phase, x, gap_split(traced, phase, T), T), calphad
    left, right = strip.phases
    if left == right:
        shown = names.names([(left, strip.x_min[k]), (right, strip.x_max[k])], T=T)
    else:
        shown = [
            names.name(left, strip.x_min[k], gap_split(traced, left, T), T),
            names.name(right, strip.x_max[k], gap_split(traced, right, T), T),
        ]
    ordered = sorted(zip(strip.phases, shown), key=lambda item: "liquid" not in item[0].lower())
    return " + ".join(name for _, name in ordered), calphad


# --- composition range ----------------------------------------------------------


def _clip_half(points, x_limit: float, keep_below: bool):
    """Sutherland-Hodgman: the part of a closed polygon on one side of x = x_limit."""
    def inside(p):
        return p[0] <= x_limit if keep_below else p[0] >= x_limit

    result = []
    for k, current in enumerate(points):
        previous = points[k - 1]
        if inside(current):
            if not inside(previous):
                result.append(_crossing(previous, current, x_limit))
            result.append(current)
        elif inside(previous):
            result.append(_crossing(previous, current, x_limit))
    return result


def _crossing(p, q, x_limit):
    t = (x_limit - p[0]) / (q[0] - p[0])
    return (x_limit, p[1] + t * (q[1] - p[1]))


def clip_polygon(x, y, x_range):
    """A polygon (x, y) clipped to x_range[0] <= x <= x_range[1]; empty arrays
    if nothing is left."""
    points = list(zip(np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
    points = _clip_half(points, x_range[1], keep_below=True)
    if points:
        points = _clip_half(points, x_range[0], keep_below=False)
    if len(points) < 3:
        return np.array([]), np.array([])
    xs, ys = zip(*points)
    return np.array(xs), np.array(ys)


def polygon_area(x, y) -> float:
    x, y = np.asarray(x), np.asarray(y)
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def polygon_anchor(x, y, grid: int = 41) -> tuple[float, float]:
    """Area centroid of a polygon or, if it falls outside a non-convex
    outline, the grid point inside the polygon nearest to it."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2.0
    path = PolygonPath(np.column_stack([x, y]))
    if abs(area) < 1e-15:
        return float(x.mean()), float(y.mean())
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6.0 * area)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6.0 * area)
    if path.contains_point((cx, cy)):
        return float(cx), float(cy)
    gx, gy = np.meshgrid(np.linspace(x.min(), x.max(), grid), np.linspace(y.min(), y.max(), grid))
    points = np.column_stack([gx.ravel(), gy.ravel()])
    inside = points[path.contains_points(points)]
    if not len(inside):
        return float(x.mean()), float(y.mean())
    # nearest in units of the polygon's extent
    sx, sy = max(np.ptp(x), 1e-12), max(np.ptp(y), 1e-12)
    k = int(np.argmin(((inside[:, 0] - cx) / sx) ** 2 + ((inside[:, 1] - cy) / sy) ** 2))
    return float(inside[k, 0]), float(inside[k, 1])


# --- placement --------------------------------------------------------------------


@dataclass(frozen=True)
class PlotBox:
    """Plot area in data units and its nominal size in pixels."""

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    width_px: float = 760.0
    height_px: float = 470.0

    def to_px(self, x, y):
        (x0, x1), (y0, y1) = self.x_range, self.y_range
        return (np.asarray(x) - x0) / (x1 - x0) * self.width_px, (np.asarray(y) - y0) / (y1 - y0) * self.height_px

    def to_data(self, px, py):
        (x0, x1), (y0, y1) = self.x_range, self.y_range
        return x0 + px / self.width_px * (x1 - x0), y0 + py / self.height_px * (y1 - y0)


@dataclass
class Label:
    """A placed label: `anchor` is the point it labels, `position` the
    centre of its text (data units); `leader` draws a line between them."""

    text: str
    anchor: tuple[float, float]
    position: tuple[float, float]
    leader: bool = False
    kind: str = "field"
    hover: str = ""
    extra: dict = field(default_factory=dict)


def text_box_px(text: str, centre_px: tuple[float, float]) -> tuple[float, float, float, float]:
    """(left, bottom, right, top) of a text box in pixels (y up)."""
    lines = text.split("<br>")
    width = CHAR_WIDTH_PX * max(len(line) for line in lines) + 2 * PADDING_PX
    height = LINE_HEIGHT_PX * len(lines) + 2 * PADDING_PX
    cx, cy = centre_px
    return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2


def label_box_px(label: Label, plot: PlotBox):
    px, py = plot.to_px(*label.position)
    return text_box_px(label.text, (float(px), float(py)))


def _inside_plot(box, plot: PlotBox) -> bool:
    left, bottom, right, top = box
    return left >= 0 and bottom >= 0 and right <= plot.width_px and top <= plot.height_px


def _overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _box_points(box, n: int = 5):
    left, bottom, right, top = box
    xs, ys = np.linspace(left, right, n), np.linspace(bottom, top, 3)
    return [(x, y) for x in xs for y in ys]


def _segment_hits_box(p, q, box) -> bool:
    """Whether segment pq intersects the axis-aligned box (Liang-Barsky)."""
    left, bottom, right, top = box
    (x0, y0), (x1, y1) = p, q
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-dx, x0 - left), (dx, right - x0), (-dy, y0 - bottom), (dy, top - y0)):
        if pk == 0:
            if qk < 0:
                return False
            continue
        r = qk / pk
        if pk < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return False
    return True


def _hits_lines(box, lines_px) -> bool:
    for xs, ys in lines_px:
        for k in range(len(xs) - 1):
            if _segment_hits_box((xs[k], ys[k]), (xs[k + 1], ys[k + 1]), box):
                return True
    return False


def _clamped(box, plot: PlotBox):
    """Shift a box (keeping its size) to lie inside the plot area."""
    left, bottom, right, top = box
    dx = max(0.0, -left) - max(0.0, right - plot.width_px)
    dy = max(0.0, -bottom) - max(0.0, top - plot.height_px)
    return left + dx, bottom + dy, right + dx, top + dy


def _interior_candidates(polygon_px: PolygonPath, anchor_px):
    """The anchor, then grid points inside the polygon, nearest first."""
    (left, bottom), (right, top) = polygon_px.vertices.min(axis=0), polygon_px.vertices.max(axis=0)
    xs = np.arange(left, right + INTERIOR_STEP_PX, INTERIOR_STEP_PX)
    ys = np.arange(bottom, top + INTERIOR_STEP_PX, INTERIOR_STEP_PX)
    grid = np.array([(x, y) for x in xs for y in ys])
    if len(grid):
        grid = grid[polygon_px.contains_points(grid)]
        grid = grid[np.argsort(np.hypot(grid[:, 0] - anchor_px[0], grid[:, 1] - anchor_px[1]))]
    return [tuple(anchor_px)] + [tuple(point) for point in grid]


def place_field_label(text: str, polygon_px: PolygonPath, anchor_px, plot: PlotBox, placed: list, lines_px=()):
    """Centre (px) of a field label, whether it needs a leader line, and its box."""
    for centre in _interior_candidates(polygon_px, anchor_px):
        box = text_box_px(text, centre)
        if (
            _inside_plot(box, plot)
            and all(polygon_px.contains_point(p) for p in _box_points(box))
            and not any(_overlaps(box, other) for other in placed)
            and not _hits_lines(box, lines_px)
        ):
            return centre, False, box
    ax, ay = anchor_px
    for avoid_lines in (True, False):
        for distance in LEADER_DISTANCES_PX:
            for ux, uy in LEADER_DIRECTIONS:
                norm = (ux * ux + uy * uy) ** 0.5
                centre = (ax + ux / norm * distance, ay - uy / norm * distance)
                box = text_box_px(text, centre)
                if not _inside_plot(box, plot) or any(_overlaps(box, other) for other in placed):
                    continue
                if avoid_lines and _hits_lines(box, lines_px):
                    continue
                return centre, True, box
    box = _clamped(text_box_px(text, (ax, ay + LEADER_DISTANCES_PX[0])), plot)
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), True, box


def place_invariant_label(text: str, x_range_px, y_px, plot: PlotBox, lines_px, placed: list):
    """Centre (px) of an invariant label just above its line, avoiding
    boundary lines; and whether it needs a leader line to the line."""
    x_lo, x_hi = x_range_px
    centre_x = (x_lo + x_hi) / 2.0
    fallback = None
    for raise_px in INVARIANT_RAISES_PX:
        for step in INVARIANT_STEPS_PX:
            box = text_box_px(text, (centre_x + step, 0.0))
            height = box[3] - box[1]
            centre = (centre_x + step, y_px + INVARIANT_GAP_PX + raise_px + height / 2.0)
            box = text_box_px(text, centre)
            if not _inside_plot(box, plot) or any(_overlaps(box, other) for other in placed):
                continue
            if fallback is None:
                fallback = (centre, box)
            if not _hits_lines(box, lines_px):
                return centre, raise_px > 0 or not (x_lo <= centre[0] <= x_hi), box
    if fallback is not None:
        return fallback[0], True, fallback[1]
    box = _clamped(text_box_px(text, (centre_x, y_px + 20.0)), plot)
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), True, box


COMPOSITION_LABEL_GAP_PX = 6.0
COMPOSITION_LABEL_BOTTOM_PX = 14.0


def place_composition_labels(text_left: str, text_right: str, x_line: float, plot: PlotBox):
    """Centres (data units) of the labels either side of a vertical line, at
    the bottom of the plot, kept inside it."""
    x_px = float(plot.to_px(x_line, plot.y_range[0])[0])
    result = []
    for text, sign in ((text_left, -1.0), (text_right, 1.0)):
        box = text_box_px(text, (0.0, 0.0))
        half_width, half_height = (box[2] - box[0]) / 2.0, (box[3] - box[1]) / 2.0
        centre = (x_px + sign * (COMPOSITION_LABEL_GAP_PX + half_width), COMPOSITION_LABEL_BOTTOM_PX + half_height)
        box = _clamped(text_box_px(text, centre), plot)
        result.append((text, plot.to_data((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0), box))
    return result


def layout_labels(
    field_items, invariant_items, boundary_lines, plot: PlotBox, line_items=(), composition_items=()
) -> list[Label]:
    """Place field, invariant and marked-line labels (see module docstring).

    field_items: (text, hover, polygon (x, T) arrays, anchor (x, T)) in data units;
    invariant_items: (text, hover, x_min, x_max, T, marker x) in data units;
    boundary_lines: (x, T) arrays in data units (boundaries and invariant lines);
    line_items: (text, hover, anchor (x, T)) for labels of marked lines;
    composition_items: (left text, right text, x) for vertical composition lines.
    """
    lines_px = [plot.to_px(x, y) for x, y in boundary_lines]
    placed_boxes: list = []
    labels: list[Label] = []
    for left, right, x_line in composition_items:
        for text, position, box in place_composition_labels(left, right, x_line, plot):
            placed_boxes.append(box)
            labels.append(Label(text, position, position, False, "composition", text))
    for text, hover, x_min, x_max, T, marker_x in invariant_items:
        (x_lo, x_hi), (y_px, _) = plot.to_px([x_min, x_max], [T, T])
        centre, leader, box = place_invariant_label(text, (float(x_lo), float(x_hi)), float(y_px), plot, lines_px, placed_boxes)
        placed_boxes.append(box)
        anchor_x = min(max(plot.to_data(centre[0], 0.0)[0], x_min), x_max)
        labels.append(
            Label(text, (anchor_x, T), plot.to_data(*centre), leader, "invariant", hover, {"marker_x": marker_x})
        )
    for text, hover, (poly_x, poly_y), anchor in field_items:
        polygon_px = PolygonPath(np.column_stack(plot.to_px(poly_x, poly_y)))
        anchor_px = tuple(float(v) for v in plot.to_px(*anchor))
        centre, leader, box = place_field_label(text, polygon_px, anchor_px, plot, placed_boxes, lines_px)
        placed_boxes.append(box)
        labels.append(Label(text, anchor, plot.to_data(*centre), leader, "field", hover))
    for text, hover, anchor in line_items:
        anchor_px = tuple(float(v) for v in plot.to_px(*anchor))
        point = PolygonPath(np.array([anchor_px, anchor_px, anchor_px]))
        centre, _, box = place_field_label(text, point, anchor_px, plot, placed_boxes, lines_px)
        placed_boxes.append(box)
        labels.append(Label(text, anchor, plot.to_data(*centre), True, "line", hover))
    return labels
