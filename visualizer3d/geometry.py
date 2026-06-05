import copy
import re
from typing import Optional

import numpy as np
import open3d as o3d


def clone_geometry(geometry):
    point_cloud_type = getattr(o3d.geometry, "PointCloud", None)
    triangle_mesh_type = getattr(o3d.geometry, "TriangleMesh", None)
    if point_cloud_type is not None and isinstance(geometry, point_cloud_type):
        return point_cloud_type(geometry)
    if triangle_mesh_type is not None and isinstance(geometry, triangle_mesh_type):
        return triangle_mesh_type(geometry)
    return copy.deepcopy(geometry)


def geometry_point_count(geometry) -> int:
    point_cloud_type = getattr(o3d.geometry, "PointCloud", None)
    triangle_mesh_type = getattr(o3d.geometry, "TriangleMesh", None)
    voxel_grid_type = getattr(o3d.geometry, "VoxelGrid", None)
    if point_cloud_type is not None and isinstance(geometry, point_cloud_type):
        return len(geometry.points)
    if triangle_mesh_type is not None and isinstance(geometry, triangle_mesh_type):
        return len(geometry.vertices)
    if voxel_grid_type is not None and isinstance(geometry, voxel_grid_type):
        return len(geometry.get_voxels())
    return 0


def parse_matrix(text: str) -> np.ndarray:
    tokens = re.split(r"[,\s]+", text.strip())
    tokens = [t for t in tokens if t]
    if len(tokens) != 16:
        raise ValueError("Matrix input must have 16 numbers.")
    vals = np.array([float(t) for t in tokens], dtype=float)
    return vals.reshape(4, 4)


def layout_clouds(clouds, gap: float, preserve_coordinates: bool = False) -> list:
    if not clouds:
        return []
    if preserve_coordinates:
        return [clone_geometry(cloud) for cloud in clouds]
    extents = [c.get_axis_aligned_bounding_box().get_extent() for c in clouds]
    max_x = max(e[0] for e in extents) if extents else 1.0
    max_y = max(e[1] for e in extents) if extents else 1.0
    step_x = max_x + gap
    step_y = max_y + gap

    n = len(clouds)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    positioned = []
    for idx, cloud in enumerate(clouds):
        row = idx // cols
        col = idx % cols
        offset_x = (col - (cols - 1) / 2.0) * step_x
        offset_y = ((rows - 1) / 2.0 - row) * step_y
        copy_cloud = clone_geometry(cloud)
        copy_cloud.translate(-copy_cloud.get_center())
        copy_cloud.translate([offset_x, offset_y, 0.0])
        positioned.append(copy_cloud)
    return positioned


def combined_bbox(clouds) -> Optional[o3d.geometry.AxisAlignedBoundingBox]:
    if not clouds:
        return None
    mins = []
    maxs = []
    for cloud in clouds:
        bbox = cloud.get_axis_aligned_bounding_box()
        mins.append(bbox.get_min_bound())
        maxs.append(bbox.get_max_bound())
    min_bound = np.min(np.asarray(mins), axis=0)
    max_bound = np.max(np.asarray(maxs), axis=0)
    return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)


def create_camera_axis_marker(size: float = 1.0):
    depth = float(size)
    half_width = depth * 0.35
    half_height = depth * 0.25
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [depth, -half_width, -half_height],
            [depth, half_width, -half_height],
            [depth, half_width, half_height],
            [depth, -half_width, half_height],
        ],
        dtype=float,
    )
    lines = np.asarray(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ],
        dtype=int,
    )
    marker = o3d.geometry.LineSet()
    marker.points = o3d.utility.Vector3dVector(points)
    marker.lines = o3d.utility.Vector2iVector(lines)
    marker.colors = o3d.utility.Vector3dVector(
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.45, 1.0],
                [1.0, 0.85, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.55, 0.0],
                [0.8, 0.8, 0.8],
            ],
            dtype=float,
        )
    )
    return marker


def create_axis_marker(style: str, size: float = 1.0, transform=None):
    if style == "Camera":
        marker = create_camera_axis_marker(size)
    else:
        marker = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size, origin=[0, 0, 0]
        )
    if transform is not None:
        marker.transform(transform)
    return marker


def clamp_camera_line_width(value: float) -> float:
    return min(12.0, max(1.0, float(value)))


def configure_axis_marker_material(material, style: str, line_width: float = 5.0):
    if style == "Camera":
        material.shader = "unlitLine"
        material.line_width = clamp_camera_line_width(line_width)
    else:
        material.shader = "defaultUnlit"
    return material


def split_clouds(clouds, pane_count: int) -> list:
    groups = [[] for _ in range(max(pane_count, 0))]
    if pane_count <= 0 or not clouds:
        return groups
    group_size = int(np.ceil(len(clouds) / pane_count))
    for idx, cloud in enumerate(clouds):
        group_idx = min(idx // group_size, pane_count - 1)
        groups[group_idx].append(cloud)
    return groups
