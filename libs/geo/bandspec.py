"""BandSpec: which bands a raster holds, in which order, and whether indices were appended."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import rasterio

TAG_KEY = "BANDSPEC"


@dataclass(frozen=True)
class BandSpec:
    names: tuple[str, ...]
    reflectance_scale: float = 1.0
    indices: tuple[str, ...] = ()

    @property
    def n_channels(self) -> int:
        return len(self.names) + len(self.indices)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> BandSpec:
        d = json.loads(text)
        return cls(
            names=tuple(d["names"]),
            reflectance_scale=float(d.get("reflectance_scale", 1.0)),
            indices=tuple(d.get("indices", ())),
        )


def write_bandspec(dst: rasterio.io.DatasetWriter, spec: BandSpec) -> None:
    dst.update_tags(**{TAG_KEY: spec.to_json()})


def read_bandspec(src: rasterio.io.DatasetReader) -> BandSpec:
    tags = src.tags()
    if TAG_KEY not in tags:
        raise ValueError(f"{src.name} has no {TAG_KEY} tag; was it written by this pipeline?")
    return BandSpec.from_json(tags[TAG_KEY])
