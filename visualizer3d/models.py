from dataclasses import dataclass


@dataclass
class AssetPart:
    path: str
    geometry: object
    preferred_mode: str
    category: str
