from minigrid.core.actions import Actions
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

OBJECT_NAMES = {v: k for k, v in OBJECT_TO_IDX.items()}
COLOR_NAMES = {v: k for k, v in COLOR_TO_IDX.items()}
STATE_NAMES = {v: k for k, v in STATE_TO_IDX.items()}

# MiniGrid world coordinates: 0=east, 1=south, 2=west, 3=north.
DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
PICKABLE = {"ball", "key", "box"}


def decode_cell(encoded):
    obj_id, color_id, state_id = (int(v) for v in encoded)
    return {
        "object": OBJECT_NAMES.get(obj_id, "unknown"),
        "color": COLOR_NAMES.get(color_id, "unknown"),
        "state": STATE_NAMES.get(state_id, "unknown"),
    }


def is_walkable(cell):
    if not cell:
        return False
    obj = cell.get("object")
    if obj in {"empty", "floor", "goal"}:
        return True
    return obj == "door" and cell.get("state") == "open"


class BeliefState:
    """Agent-side belief reconstructed only from observations and actions.

    Coordinates use an arbitrary origin chosen at reset. They are not MiniGrid
    global coordinates. This class deliberately never reads environment fields.
    """

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.agent_pos = (0, 0)
        self.agent_dir = 0
        self.known_map = {}
        self.inventory_status = "unknown"
        self.held_object = None
        self.completed_subgoals = []
        self.observation_count = 0
        self.last_obs = None

    def reset(self, obs, initial_inventory="empty"):
        self.reset_state()
        self.agent_dir = int(obs["direction"])
        self.inventory_status = initial_inventory
        self.last_obs = obs
        self._integrate_observation(obs)

    @staticmethod
    def _front_cell(obs):
        image = obs["image"]
        width, height, _ = image.shape
        return decode_cell(image[width // 2, height - 2])

    @staticmethod
    def _world_offset(forward, lateral, direction):
        hx, hy = DIRECTIONS[int(direction)]
        rx, ry = -hy, hx
        return (forward * hx + lateral * rx, forward * hy + lateral * ry)

    def _integrate_observation(self, obs):
        image = obs["image"]
        width, height, _ = image.shape
        center = width // 2
        direction = int(obs["direction"])
        self.observation_count += 1

        for x in range(width):
            for y in range(height):
                cell = decode_cell(image[x, y])
                if cell["object"] == "unseen":
                    continue
                forward = height - 1 - y
                lateral = x - center
                dx, dy = self._world_offset(forward, lateral, direction)
                pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

                if cell["object"] == "agent":
                    # The agent occludes the underlying tile. Preserve the
                    # last observation of that tile (e.g. an open door or
                    # green goal) instead of rewriting it as empty floor.
                    previous_cell = self.known_map.get(pos)
                    if previous_cell is None:
                        previous_cell = {
                            "object": "empty",
                            "color": "unknown",
                            "state": "unknown",
                            "last_seen": self.observation_count,
                        }
                    self.known_map[pos] = dict(previous_cell)
                    continue

                self.known_map[pos] = {
                    **cell,
                    "last_seen": self.observation_count,
                }

        # Normally the agent marker above has already populated this cell.
        # Keep a safe fallback for observations without an explicit marker,
        # but never overwrite previously observed terrain/object identity.
        if self.agent_pos not in self.known_map:
            self.known_map[self.agent_pos] = {
                "object": "empty",
                "color": "unknown",
                "state": "unknown",
                "last_seen": self.observation_count,
            }

    def update(self, previous_obs, action, next_obs):
        action_id = int(action)
        before_front = self._front_cell(previous_obs)
        before_inventory = self.inventory_status
        before_held = self.held_object
        before_dir = int(previous_obs["direction"])
        before_pos = tuple(self.agent_pos)
        front_delta = DIRECTIONS[before_dir]
        front_pos = (before_pos[0] + front_delta[0], before_pos[1] + front_delta[1])

        if action_id == int(Actions.forward) and is_walkable(before_front):
            dx, dy = DIRECTIONS[before_dir]
            self.agent_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        after_front = self._front_cell(next_obs)

        if (
            action_id == int(Actions.pickup)
            and before_inventory == "empty"
            and before_front.get("object") in PICKABLE
            and not (
                after_front.get("object") == before_front.get("object")
                and after_front.get("color") == before_front.get("color")
            )
        ):
            self.held_object = {
                "object": before_front["object"],
                "color": before_front.get("color", "unknown"),
                "state": "held",
            }
            self.inventory_status = (
                f"{before_front.get('color', 'unknown')}_{before_front['object']}"
            )
            self._mark_completed("pick")
            self._mark_completed("pickup")

        # Record every state transition supported by the new observation.
        # In MiniGrid DoorKey, toggling a locked door with the matching key can
        # transition directly from locked to open, satisfying both subgoals.
        if action_id == int(Actions.toggle) and before_front.get("object") == "door":
            before_state = before_front.get("state")
            after_state = after_front.get("state")
            if before_state == "closed" and after_state == "open":
                self._mark_completed("open_door")
            elif before_state == "locked" and after_state in {"closed", "open"}:
                self._mark_completed("unlock_door")
                if after_state == "open":
                    self._mark_completed("open_door")

        # Drop puts the held object into the cell directly in front.
        if action_id == int(Actions.drop) and before_held is not None:
            if (
                after_front.get("object") == before_held.get("object")
                and after_front.get("color") == before_held.get("color")
            ):
                self.held_object = None
                self.inventory_status = "empty"
                self._mark_completed("drop")

        self.agent_dir = int(next_obs["direction"])
        self._integrate_observation(next_obs)
        self.last_obs = next_obs

    def _mark_completed(self, name):
        if name not in self.completed_subgoals:
            self.completed_subgoals.append(name)

    def find_objects(self, object_name, color=None, state=None):
        matches = []
        for pos, cell in self.known_map.items():
            if cell.get("object") != object_name:
                continue
            if color is not None and cell.get("color") != color:
                continue
            if state is not None and cell.get("state") != state:
                continue
            matches.append((pos, cell))
        return matches

    def is_known(self, pos):
        return tuple(pos) in self.known_map

    def cell_at(self, pos):
        return self.known_map.get(tuple(pos))

    def summary(self):
        return {
            "estimated_position": list(self.agent_pos),
            "direction": self.agent_dir,
            "known_cell_count": len(self.known_map),
            "inventory_status": self.inventory_status,
            "held_object": dict(self.held_object) if self.held_object else None,
            "completed_subgoals": list(self.completed_subgoals),
            "observation_count": self.observation_count,
        }
