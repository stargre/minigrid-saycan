"""Shared, observation-grounded placement-cell selection."""
from perception.belief_state import DIRECTIONS
from skills.target import relation_offset


_NEAR_RELATIONS = {"near", "adjacent", "nearby", "beside", "next_to"}


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _has_reachable_approach(belief, navigator, target_pos):
    """Return whether the agent can reach a walkable cell beside target_pos."""
    start = tuple(belief.agent_pos)
    for dx, dy in DIRECTIONS:
        approach = (target_pos[0] + dx, target_pos[1] + dy)
        if not navigator._safe_walkable(belief.cell_at(approach)):
            continue
        if navigator._find_path(start, approach) is not None:
            return True
    return False


def select_placement(belief, navigator, destination_positions, relation):
    """Select a known-empty, reachable placement cell around a destination.

    Returns (destination_position, placement_position, error_status).
    A None error_status means both the destination and a reachable placement
    cell have been found. The function is shared by feasibility and execution
    so the planner cannot approve a different placement cell from the one the
    skill subsequently chooses.
    """
    relation_name = str(relation or "near").strip().lower().replace("-", "_")
    offset = relation_offset(relation_name)
    if offset is None and relation_name not in _NEAR_RELATIONS:
        return None, None, "unsupported_relation"

    start = tuple(belief.agent_pos)
    ordered_destinations = sorted(
        {tuple(pos) for pos in destination_positions},
        key=lambda pos: (_distance(pos, start), pos),
    )
    saw_occupied = False
    saw_unreachable = False

    for dest_pos in ordered_destinations:
        if offset is not None:
            raw_candidates = [(dest_pos[0] + offset[0], dest_pos[1] + offset[1])]
        else:
            raw_candidates = [
                (dest_pos[0] + dx, dest_pos[1] + dy)
                for dx, dy in DIRECTIONS
            ]

        known_empty = []
        for pos in raw_candidates:
            cell = belief.cell_at(pos)
            if cell is None:
                continue
            if cell.get("object") not in {"empty", "floor"}:
                saw_occupied = True
                continue
            known_empty.append(pos)

        if not known_empty:
            continue

        reachable = [
            pos for pos in known_empty
            if _has_reachable_approach(belief, navigator, pos)
        ]
        if not reachable:
            saw_unreachable = True
            continue

        placement_pos = min(
            reachable,
            key=lambda pos: (_distance(pos, start), pos),
        )
        return dest_pos, placement_pos, None

    if saw_unreachable:
        return None, None, "no_known_path"
    if offset is not None and saw_occupied:
        return None, None, "placement_cell_occupied"
    return None, None, "placement_cell_unknown"
