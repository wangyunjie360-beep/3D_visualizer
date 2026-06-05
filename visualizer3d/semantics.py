import os
import re

import numpy as np
import open3d as o3d

from .models import AssetPart


def parse_ply_semantic_color_map(lines) -> dict[tuple[int, int, int], str]:
    mapping = {}
    pattern = re.compile(
        r"comment\s+class\s+\d+\s*:\s*(.*?)\s+rgb=\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
        re.IGNORECASE,
    )
    for line in lines:
        match = pattern.search(line.strip())
        if not match:
            continue
        name = match.group(1).strip()
        color = tuple(int(match.group(idx)) for idx in (2, 3, 4))
        mapping[color] = name
    return mapping


def find_semantic_color_map(path: str) -> dict[tuple[int, int, int], str]:
    directory = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    candidates = []
    if "_rendered_voxel_mesh" in stem:
        candidates.append(
            os.path.join(directory, stem.split("_rendered_voxel_mesh", 1)[0] + ".ply")
        )
    candidates.append(os.path.join(directory, stem + ".ply"))

    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        lines = []
        try:
            with open(candidate, "r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    lines.append(line)
                    if line.strip() == "end_header":
                        break
        except OSError:
            continue
        mapping = parse_ply_semantic_color_map(lines)
        if mapping:
            return mapping
    return {}


def _mesh_vertex_colors(mesh):
    if hasattr(mesh, "vertex_colors") and len(mesh.vertex_colors) > 0:
        return np.asarray(mesh.vertex_colors)
    if hasattr(mesh, "has_vertex_colors") and mesh.has_vertex_colors():
        return np.asarray(mesh.vertex_colors)
    return None


def _make_triangle_mesh(vertices, triangles, colors=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    if len(triangles) > 0:
        mesh.compute_vertex_normals()
    return mesh


def split_mesh_by_vertex_color(
    path: str, mesh, color_name_map: dict[tuple[int, int, int], str]
) -> list[AssetPart]:
    colors = _mesh_vertex_colors(mesh)
    if colors is None or len(colors) != len(mesh.vertices):
        return []

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        return []

    color_bytes = np.rint(np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    parts = []
    seen_colors = []
    for triangle in triangles:
        triangle_colors = color_bytes[triangle]
        if not np.all(triangle_colors == triangle_colors[0]):
            continue
        color = tuple(int(value) for value in triangle_colors[0])
        if color not in color_name_map:
            continue
        if color not in seen_colors:
            seen_colors.append(color)

    for color in seen_colors:
        category = color_name_map[color]
        triangle_mask = np.all(color_bytes[triangles] == np.asarray(color, dtype=np.uint8), axis=(1, 2))
        selected_triangles = triangles[triangle_mask]
        if len(selected_triangles) == 0:
            continue
        unique_vertices, inverse = np.unique(selected_triangles.reshape(-1), return_inverse=True)
        part_vertices = vertices[unique_vertices]
        part_triangles = inverse.reshape(-1, 3)
        part_colors = colors[unique_vertices]
        part_mesh = _make_triangle_mesh(part_vertices, part_triangles, part_colors)
        parts.append(AssetPart(path, part_mesh, "Mesh", category))
    return parts
