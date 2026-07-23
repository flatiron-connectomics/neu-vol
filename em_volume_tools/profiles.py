"""Storage target profiles: destination-aware chunk/shard/compression defaults.

Decouples the viewer-facing *read chunk* from the on-disk *file/object* size and
picks sensible defaults per destination (docs/DESIGN.md §5). All values are
overridable at call sites.

  - ``local``           : zarr v3, 128^3 chunks, unsharded  (dev / smoke test)
  - ``ceph``            : zarr v3, 128^3 read chunks packed into ~1024^3 shards
                          (sharding codec) -> few inodes, quota-safe intermediates
  - ``s3-neuroglancer`` : precomputed, small unsharded chunks (web viewing)
                          [create not implemented yet; zarr3 is the v1 target]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class StorageProfile:
    """Destination-aware storage defaults for the spatial axes."""

    format: str                                  # 'zarr3' | 'neuroglancer_precomputed'
    chunk: tuple[int, ...]                        # inner read chunk (spatial)
    shard: tuple[int, ...] | None = None          # array-level shard (spatial); None = unsharded
    compressor: str | None = "zstd"               # 'zstd' | 'gzip' | 'blosc' | None
    compressor_level: int | None = None


PROFILES: dict[str, StorageProfile] = {
    "local": StorageProfile("zarr3", chunk=(128, 128, 128), shard=None),
    "ceph": StorageProfile("zarr3", chunk=(128, 128, 128), shard=(1024, 1024, 1024)),
    "s3-neuroglancer": StorageProfile(
        "neuroglancer_precomputed", chunk=(128, 128, 128), shard=None, compressor="gzip"
    ),
}


def get_profile(profile: str | StorageProfile) -> StorageProfile:
    if isinstance(profile, StorageProfile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError as e:
        raise ValueError(f"unknown profile {profile!r}; known: {sorted(PROFILES)}") from e


def _with_channel(spatial: Sequence[int], num_channels: int, has_channels: bool) -> list[int]:
    """Prepend a full-extent channel axis when the volume is multi-channel."""
    return ([num_channels] + list(spatial)) if has_channels else list(spatial)


def zarr3_create_spec(
    profile: str | StorageProfile,
    path: str,
    shape: Sequence[int],
    dtype: str,
    *,
    has_channels: bool = False,
    num_channels: int = 1,
    dimension_names: Sequence[str] | None = None,
    chunk: Sequence[int] | None = None,
    shard: Sequence[int] | None = None,
) -> dict:
    """Build a normalized zarr3 create spec (for ``TensorStoreBackend.create``).

    ``chunk``/``shard`` override the profile's spatial values. A leading channel
    axis (full extent) is added automatically when ``has_channels``.
    """
    p = get_profile(profile)
    if p.format != "zarr3":
        raise NotImplementedError(f"create spec for format {p.format!r} not implemented (zarr3 only)")

    spatial_chunk = tuple(chunk) if chunk is not None else p.chunk
    spatial_shard = tuple(shard) if shard is not None else p.shard

    chunk_full = _with_channel(spatial_chunk, num_channels, has_channels)
    spec: dict = {
        "backend": "zarr3",
        "path": path,
        "shape": list(shape),
        "dtype": dtype,
        "compressor": p.compressor,
        "compressor_level": p.compressor_level,
    }
    if dimension_names is not None:
        spec["dimension_names"] = list(dimension_names)

    if spatial_shard is not None:
        # Sharded: array-level chunk = shard; inner (read) chunk = chunk.
        for s, c in zip(spatial_shard, spatial_chunk):
            if s % c != 0:
                raise ValueError(f"shard {spatial_shard} must be a multiple of chunk {spatial_chunk}")
        spec["chunk_shape"] = _with_channel(spatial_shard, num_channels, has_channels)
        spec["inner_chunk_shape"] = chunk_full
    else:
        spec["chunk_shape"] = chunk_full
        spec["inner_chunk_shape"] = None
    return spec
