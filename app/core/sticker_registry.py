"""Central repository for active sticker instances and their adjustments.

Replaces the bare ``self.active_stickers`` and ``self.adjustments`` dicts
that were shared across ``ConsumerProcessor`` and its mixins.  Provides
typed access via ``StickerInstance`` / ``Adjustment`` dataclasses and an
event hook for cross-domain notifications (e.g. when a sticker is removed,
the animation processor must clean up its state).
"""

import copy
import logging
from typing import Callable

from app.core.protocol import Adjustment, StickerInstance

log = logging.getLogger(__name__)


class StickerRegistry:
    """Manages the lifecycle of active sticker instances.

    Events
    ------
    ``"removed"`` — called with ``(instance_id: str)`` after an instance is
    removed.  Used by ``AnimationProcessor`` to clean up evaluation state,
    delta-mode tracking, and texture animator entries.
    """

    def __init__(self, max_stickers: int = 20):
        self._instances: dict[str, StickerInstance] = {}
        self._adjustments: dict[str, Adjustment] = {}
        self._callbacks: dict[str, list[Callable]] = {}
        self._max_stickers = max_stickers

    # ── properties ──

    @property
    def count(self) -> int:
        return len(self._instances)

    @property
    def max_stickers(self) -> int:
        return self._max_stickers

    @property
    def all(self) -> list[StickerInstance]:
        """Return all instances in insertion order (index 0 = bottom)."""
        # Python 3.7+ dicts preserve insertion order
        return list(self._instances.values())

    @property
    def is_full(self) -> bool:
        return self.count >= self._max_stickers

    # ── instance CRUD ──

    def add(self, inst: StickerInstance, adj: Adjustment | None = None) -> str:
        """Register *inst* and its optional *adj*.  Returns ``instance_id``."""
        self._instances[inst.instance_id] = inst
        self._adjustments[inst.instance_id] = (
            copy.deepcopy(adj) if adj is not None else Adjustment()
        )
        return inst.instance_id

    def remove(self, instance_id: str) -> StickerInstance | None:
        """Remove *instance_id* and fire ``"removed"``.  Returns the removed instance or ``None``."""
        inst = self._instances.pop(instance_id, None)
        self._adjustments.pop(instance_id, None)
        if inst is not None:
            self._fire("removed", instance_id)
        return inst

    def get(self, instance_id: str) -> StickerInstance | None:
        return self._instances.get(instance_id)

    def has(self, instance_id: str) -> bool:
        return instance_id in self._instances

    def clear(self) -> None:
        """Remove all instances, firing ``"removed"`` for each."""
        for iid in list(self._instances):
            self.remove(iid)

    # ── adjustment access ──

    def get_adj(self, instance_id: str) -> Adjustment:
        """Return the adjustment for *instance_id* (always a valid object)."""
        adj = self._adjustments.get(instance_id)
        if adj is None:
            adj = Adjustment()
            self._adjustments[instance_id] = adj
        return adj

    def update_adj(self, instance_id: str, **kwargs) -> None:
        """Update specific fields on the adjustment (e.g. ``offset_x=5.0``)."""
        adj = self.get_adj(instance_id)
        for key, value in kwargs.items():
            if hasattr(adj, key):
                setattr(adj, key, value)

    # ── event hooks ──

    def on(self, event: str, cb: Callable) -> None:
        """Register *cb* to be called when *event* fires."""
        self._callbacks.setdefault(event, []).append(cb)

    def _fire(self, event: str, *args, **kwargs) -> None:
        for cb in self._callbacks.get(event, []):
            try:
                cb(*args, **kwargs)
            except Exception:
                log.exception("StickerRegistry event handler failed: %s", event)
