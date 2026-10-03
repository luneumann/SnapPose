"""Tiny component registry with license tags (enforces `license_mode: commercial`)."""
from __future__ import annotations

NON_COMMERCIAL = {"non-commercial", "unknown"}


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, tuple[object, str]] = {}

    def register(self, name: str, license: str = "own"):
        def deco(obj):
            self._items[name] = (obj, license)
            return obj
        return deco

    def get(self, name: str, license_mode: str = "commercial"):
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {sorted(self._items)}")
        obj, lic = self._items[name]
        if license_mode == "commercial" and lic in NON_COMMERCIAL:
            raise PermissionError(
                f"{self.kind} {name!r} has license {lic!r} and is blocked by license_mode: commercial "
                "(benchmark-only component)")
        return obj


REFINERS = Registry("refiner")
