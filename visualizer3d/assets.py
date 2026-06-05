import os

import numpy as np
import open3d as o3d

from .models import AssetPart
from .semantics import find_semantic_color_map, split_mesh_by_vertex_color


def load_bin_as_pcd(path: str) -> o3d.geometry.PointCloud:
    raw = np.fromfile(path, dtype=np.float32)
    cols = None
    for k in (5, 4, 3):
        if raw.size % k == 0:
            cols = k
            break
    if cols is None:
        raise ValueError(f"{os.path.basename(path)}: unexpected float count {raw.size}.")
    pts_all = raw.reshape(-1, cols)
    pts = pts_all[:, :3]
    colors = None
    if cols >= 4:
        intensities = pts_all[:, 3]
        if intensities.max() > intensities.min():
            norm = (intensities - intensities.min()) / np.ptp(intensities)
            colors = np.stack([norm, norm, norm], axis=1).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def preferred_render_mode_for_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".glb", ".gltf"}:
        return "Mesh"
    if ext in {".bin", ".pcd", ".ply"}:
        return "Points"
    raise ValueError(f"Unsupported file type: {ext}")


def load_asset(path: str, use_cuda: bool):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        return load_bin_as_pcd(path), preferred_render_mode_for_path(path)
    if ext in {".pcd", ".ply"}:
        if use_cuda:
            try:
                device = o3d.core.Device("CUDA:0")
                tpcd = o3d.t.io.read_point_cloud(path, device=device)
                pcd = tpcd.to_legacy()
            except Exception:
                pcd = o3d.io.read_point_cloud(path)
        else:
            pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise ValueError(f"{os.path.basename(path)}: loaded empty point cloud.")
        return pcd, preferred_render_mode_for_path(path)
    if ext in {".glb", ".gltf"}:
        mesh = None
        tensor_reader = getattr(getattr(o3d, "t", None), "io", None)
        if tensor_reader is not None and hasattr(tensor_reader, "read_triangle_mesh"):
            mesh = tensor_reader.read_triangle_mesh(path)
            if hasattr(mesh, "to_legacy"):
                mesh = mesh.to_legacy()
        if mesh is None:
            mesh = o3d.io.read_triangle_mesh(path)
        if mesh.is_empty():
            raise ValueError(f"{os.path.basename(path)}: loaded empty triangle mesh.")
        if len(mesh.triangles) > 0 and not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        return mesh, preferred_render_mode_for_path(path)
    raise ValueError(f"Unsupported file type: {ext}")


def load_asset_parts(path: str, use_cuda: bool) -> list[AssetPart]:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".glb", ".gltf"} and hasattr(o3d.io, "read_triangle_model"):
        model = o3d.io.read_triangle_model(path)
        mesh_infos = list(getattr(model, "meshes", []))
        if len(mesh_infos) == 1:
            mesh_info = mesh_infos[0]
            mesh, preferred_mode = load_asset(path, use_cuda)
            semantic_parts = split_mesh_by_vertex_color(
                path, mesh, find_semantic_color_map(path)
            )
            if semantic_parts:
                return semantic_parts
            name = getattr(mesh_info, "mesh_name", "") or os.path.splitext(
                os.path.basename(path)
            )[0]
            return [AssetPart(path, mesh, preferred_mode, name)]
        parts = []
        for idx, mesh_info in enumerate(mesh_infos, start=1):
            mesh = getattr(mesh_info, "mesh", None)
            if mesh is None or mesh.is_empty():
                continue
            if len(mesh.triangles) > 0 and not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            name = getattr(mesh_info, "mesh_name", "") or f"mesh_{idx}"
            parts.append(AssetPart(path, mesh, "Mesh", name))
        if parts:
            return parts

    geometry, preferred_mode = load_asset(path, use_cuda)
    category = preferred_mode if preferred_mode != "Mesh" else os.path.splitext(os.path.basename(path))[0]
    return [AssetPart(path, geometry, preferred_mode, category)]
