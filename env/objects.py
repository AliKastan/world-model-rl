"""Game objects for the Sokoban-style puzzle environment.

Every object lives on a grid cell (x, y). Objects define their physics
(solid, pushable), visual properties (color, symbol), and affordances
(what interactions are possible).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Tuple, Type


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class GameObject:
    """Base class for all grid objects."""

    pos: Tuple[int, int]
    obj_type: str = "generic"
    color: Tuple[int, int, int] = (200, 200, 200)
    pushable: bool = False
    solid: bool = True
    symbol: str = "?"
    affordances: List[str] = field(default_factory=list)

    # Registry for from_dict deserialization
    _registry: ClassVar[Dict[str, Type[GameObject]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Auto-register every concrete subclass
        GameObject._registry[cls.__name__] = cls

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the object to a plain dictionary."""
        return {
            "class": type(self).__name__,
            "pos": list(self.pos),
            "obj_type": self.obj_type,
            "color": list(self.color),
            "pushable": self.pushable,
            "solid": self.solid,
            "symbol": self.symbol,
            "affordances": list(self.affordances),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GameObject:
        """Deserialize an object from a dictionary.

        Dispatches to the correct subclass based on the ``class`` key.
        """
        class_name = data["class"]
        subcls = GameObject._registry.get(class_name)
        if subcls is None:
            raise ValueError(f"Unknown object class: {class_name}")
        return subcls._from_dict_impl(data)

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> GameObject:
        """Override in subclasses that have extra fields."""
        return cls(pos=tuple(data["pos"]))  # type: ignore[arg-type]

    # -- rendering ----------------------------------------------------------

    def render_info(self) -> Dict[str, Any]:
        """Return visual metadata used by the renderer.

        Returns:
            Dictionary with ``color``, ``shape``, and ``icon`` keys.
        """
        return {
            "color": self.color,
            "shape": "square",
            "icon": self.symbol,
        }


# ---------------------------------------------------------------------------
# Concrete objects
# ---------------------------------------------------------------------------

@dataclass
class Floor(GameObject):
    """Empty walkable space."""

    pos: Tuple[int, int] = (0, 0)
    obj_type: str = "floor"
    color: Tuple[int, int, int] = (45, 45, 58)
    pushable: bool = False
    solid: bool = False
    symbol: str = " "
    affordances: List[str] = field(default_factory=lambda: ["walkable"])

    def render_info(self) -> Dict[str, Any]:
        return {"color": self.color, "shape": "square", "icon": " "}


@dataclass
class Wall(GameObject):
    """Impassable obstacle."""

    pos: Tuple[int, int] = (0, 0)
    obj_type: str = "wall"
    color: Tuple[int, int, int] = (75, 85, 99)
    pushable: bool = False
    solid: bool = True
    symbol: str = "#"
    affordances: List[str] = field(default_factory=lambda: ["blocking"])

    def render_info(self) -> Dict[str, Any]:
        return {"color": self.color, "shape": "square", "icon": "#"}


@dataclass
class Box(GameObject):
    """Pushable crate -- core Sokoban mechanic.

    Boxes can only be **pushed**, never pulled. If pushed against a wall
    or another box the push is rejected.
    """

    pos: Tuple[int, int] = (0, 0)
    obj_type: str = "box"
    color: Tuple[int, int, int] = (251, 191, 36)
    pushable: bool = True
    solid: bool = True
    symbol: str = "$"
    affordances: List[str] = field(
        default_factory=lambda: ["pushable", "blocking", "placeable"]
    )
    on_target: bool = False

    @property
    def display_symbol(self) -> str:
        """``*`` when sitting on a target, ``$`` otherwise."""
        return "*" if self.on_target else "$"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["on_target"] = self.on_target
        return d

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> Box:
        box = cls(pos=tuple(data["pos"]))  # type: ignore[arg-type]
        box.on_target = data.get("on_target", False)
        return box

    def render_info(self) -> Dict[str, Any]:
        icon = "*" if self.on_target else "$"
        color = (52, 211, 153) if self.on_target else self.color
        return {"color": color, "shape": "square", "icon": icon}


@dataclass
class Target(GameObject):
    """Goal position where a box must be placed."""

    pos: Tuple[int, int] = (0, 0)
    obj_type: str = "target"
    color: Tuple[int, int, int] = (52, 211, 153)
    pushable: bool = False
    solid: bool = False
    symbol: str = "."
    affordances: List[str] = field(default_factory=lambda: ["goal_position"])

    def render_info(self) -> Dict[str, Any]:
        return {"color": self.color, "shape": "diamond", "icon": "."}


@dataclass
class Key(GameObject):
    """Collectible item -- picked up by walking over it."""

    pos: Tuple[int, int] = (0, 0)
    obj_type: str = "key"
    color: Tuple[int, int, int] = (251, 146, 60)
    pushable: bool = False
    solid: bool = False
    symbol: str = "k"
    affordances: List[str] = field(
        default_factory=lambda: ["collectible", "unlocks_door"]
    )
    key_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["key_id"] = self.key_id
        return d

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> Key:
        return cls(pos=tuple(data["pos"]), key_id=data.get("key_id", 0))  # type: ignore[arg-type]

    def render_info(self) -> Dict[str, Any]:
        return {"color": self.color, "shape": "circle", "icon": "k"}


@dataclass
class Door(GameObject):
    """Blocks passage until unlocked with a matching :class:`Key`."""

    pos: Tuple[int, int] = (0, 0)
    obj_type: str = "door"
    color: Tuple[int, int, int] = (239, 68, 68)  # locked color
    pushable: bool = False
    solid: bool = True
    symbol: str = "D"
    affordances: List[str] = field(
        default_factory=lambda: [
            "blocking_when_locked",
            "passable_when_unlocked",
            "requires_key",
        ]
    )
    door_id: int = 0
    locked: bool = True
    color_locked: Tuple[int, int, int] = (239, 68, 68)
    color_unlocked: Tuple[int, int, int] = (74, 222, 128)

    @property
    def display_symbol(self) -> str:
        return "D" if self.locked else "d"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["door_id"] = self.door_id
        d["locked"] = self.locked
        return d

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> Door:
        door = cls(
            pos=tuple(data["pos"]),  # type: ignore[arg-type]
            door_id=data.get("door_id", 0),
        )
        door.locked = data.get("locked", True)
        door.solid = door.locked
        door.color = door.color_locked if door.locked else door.color_unlocked
        door.symbol = door.display_symbol
        return door

    def unlock(self) -> None:
        """Unlock the door, making it passable."""
        self.locked = False
        self.solid = False
        self.color = self.color_unlocked
        self.symbol = "d"

    def render_info(self) -> Dict[str, Any]:
        color = self.color_locked if self.locked else self.color_unlocked
        icon = "D" if self.locked else "d"
        return {"color": color, "shape": "square", "icon": icon}


@dataclass
class IceTile(GameObject):
    """Slippery floor -- agent/box slides until hitting a wall or solid object."""

    pos: Tuple[int, int] = (0, 0)
    obj_type: str = "ice"
    color: Tuple[int, int, int] = (147, 197, 253)
    pushable: bool = False
    solid: bool = False
    symbol: str = "~"
    affordances: List[str] = field(
        default_factory=lambda: ["slippery", "momentum_transfer"]
    )

    def render_info(self) -> Dict[str, Any]:
        return {"color": self.color, "shape": "square", "icon": "~"}


@dataclass
class PressureSwitch(GameObject):
    """Activated when agent or box stands on it -- toggles a linked :class:`SwitchWall`."""

    pos: Tuple[int, int] = (0, 0)
    obj_type: str = "switch"
    color: Tuple[int, int, int] = (168, 85, 247)  # inactive color
    pushable: bool = False
    solid: bool = False
    symbol: str = "^"
    affordances: List[str] = field(
        default_factory=lambda: ["activatable", "toggles_wall"]
    )
    switch_id: int = 0
    activated: bool = False
    color_inactive: Tuple[int, int, int] = (168, 85, 247)
    color_active: Tuple[int, int, int] = (236, 72, 153)

    def activate(self) -> None:
        """Activate the switch."""
        self.activated = True
        self.color = self.color_active

    def deactivate(self) -> None:
        """Deactivate the switch."""
        self.activated = False
        self.color = self.color_inactive

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["switch_id"] = self.switch_id
        d["activated"] = self.activated
        return d

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> PressureSwitch:
        sw = cls(
            pos=tuple(data["pos"]),  # type: ignore[arg-type]
            switch_id=data.get("switch_id", 0),
        )
        if data.get("activated", False):
            sw.activate()
        return sw

    def render_info(self) -> Dict[str, Any]:
        color = self.color_active if self.activated else self.color_inactive
        return {"color": color, "shape": "circle", "icon": "^"}


@dataclass
class SwitchWall(GameObject):
    """Wall that appears/disappears when a linked switch is toggled."""

    pos: Tuple[int, int] = (0, 0)
    obj_type: str = "switch_wall"
    color: Tuple[int, int, int] = (168, 85, 247)
    pushable: bool = False
    solid: bool = True
    symbol: str = "%"
    affordances: List[str] = field(
        default_factory=lambda: ["conditional_blocking", "toggled_by_switch"]
    )
    linked_switch_id: int = 0
    open: bool = False

    def toggle(self) -> None:
        """Toggle between open (passable) and closed (solid)."""
        self.open = not self.open
        self.solid = not self.open
        self.symbol = " " if self.open else "%"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["linked_switch_id"] = self.linked_switch_id
        d["open"] = self.open
        return d

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> SwitchWall:
        sw = cls(
            pos=tuple(data["pos"]),  # type: ignore[arg-type]
            linked_switch_id=data.get("linked_switch_id", 0),
        )
        if data.get("open", False):
            sw.toggle()
        return sw

    def render_info(self) -> Dict[str, Any]:
        if self.open:
            return {"color": (45, 45, 58), "shape": "square", "icon": " "}
        return {"color": self.color, "shape": "square", "icon": "%"}


_DIRECTION_SYMBOLS = {"up": "^", "down": "v", "left": "<", "right": ">"}


@dataclass
class OneWayTile(GameObject):
    """Floor that only allows passage in one direction."""

    pos: Tuple[int, int] = (0, 0)
    obj_type: str = "oneway"
    color: Tuple[int, int, int] = (56, 189, 248)
    pushable: bool = False
    solid: bool = False  # base solid is False; logic handled in PuzzleWorld.step
    symbol: str = ">"
    affordances: List[str] = field(
        default_factory=lambda: ["directional_passage"]
    )
    direction: str = "right"

    def __post_init__(self) -> None:
        self.symbol = _DIRECTION_SYMBOLS.get(self.direction, ">")

    def allows(self, move_direction: str) -> bool:
        """Return True if *move_direction* is permitted through this tile."""
        return move_direction == self.direction

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["direction"] = self.direction
        return d

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> OneWayTile:
        return cls(
            pos=tuple(data["pos"]),  # type: ignore[arg-type]
            direction=data.get("direction", "right"),
        )

    def render_info(self) -> Dict[str, Any]:
        return {
            "color": self.color,
            "shape": "arrow",
            "icon": _DIRECTION_SYMBOLS.get(self.direction, ">"),
        }
