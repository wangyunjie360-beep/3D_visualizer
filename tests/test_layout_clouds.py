import importlib
import sys
import types
import unittest

import numpy as np


class _FakeBoundingBox:
    def __init__(self, points):
        self._points = np.asarray(points, dtype=float)

    def get_extent(self):
        return self.get_max_bound() - self.get_min_bound()

    def get_min_bound(self):
        return self._points.min(axis=0)

    def get_max_bound(self):
        return self._points.max(axis=0)


class _FakePointCloud:
    def __init__(self, other=None):
        if other is None:
            self.points = np.zeros((0, 3), dtype=float)
        else:
            self.points = np.array(other.points, dtype=float, copy=True)

    def translate(self, offset):
        self.points = self.points + np.asarray(offset, dtype=float)

    def get_center(self):
        return self.points.mean(axis=0)

    def get_axis_aligned_bounding_box(self):
        return _FakeBoundingBox(self.points)


class _FakeTriangleMesh:
    def __init__(self, other=None):
        if other is None:
            self.vertices = np.zeros((0, 3), dtype=float)
            self.triangles = np.zeros((0, 3), dtype=int)
        else:
            self.vertices = np.array(other.vertices, dtype=float, copy=True)
            self.triangles = np.array(other.triangles, dtype=int, copy=True)

    def is_empty(self):
        return len(self.vertices) == 0

    def has_vertex_normals(self):
        return True

    def compute_vertex_normals(self):
        return self

    def get_axis_aligned_bounding_box(self):
        return _FakeBoundingBox(self.vertices)

    def translate(self, offset):
        self.vertices = self.vertices + np.asarray(offset, dtype=float)

    def get_center(self):
        return self.vertices.mean(axis=0)

    def sample_points_uniformly(self, number_of_points):
        cloud = _FakePointCloud()
        cloud.points = self.vertices[: min(number_of_points, len(self.vertices))]
        return cloud


def _fake_mesh(vertices):
    mesh = _FakeTriangleMesh()
    mesh.vertices = np.asarray(vertices, dtype=float)
    mesh.triangles = np.array([[0, 1, 2]], dtype=int)
    return mesh


def _install_fake_open3d():
    compact_mesh = _fake_mesh([[9, 0, 0], [9, 1, 0], [9, 0, 1]])
    fake_mesh_model = types.SimpleNamespace(
        meshes=[
            types.SimpleNamespace(mesh=_fake_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), mesh_name="car"),
            types.SimpleNamespace(mesh=_fake_mesh([[0, 0, 1], [1, 0, 1], [0, 1, 1]]), mesh_name="road"),
        ]
    )
    fake_module = types.ModuleType("open3d")
    fake_module.geometry = types.SimpleNamespace(
        PointCloud=_FakePointCloud,
        TriangleMesh=_FakeTriangleMesh,
        VoxelGrid=type("_FakeVoxelGrid", (), {"get_voxels": lambda self: []}),
        AxisAlignedBoundingBox=_FakeBoundingBox,
    )
    fake_module.utility = types.SimpleNamespace(
        Vector3dVector=lambda values: np.asarray(values, dtype=float)
    )
    fake_module.io = types.SimpleNamespace(
        read_triangle_model=lambda path: types.SimpleNamespace(
            meshes=[types.SimpleNamespace(mesh=_fake_mesh([[0, 0, 0]] * 6), mesh_name="single")]
        )
        if "single" in path
        else fake_mesh_model,
        read_triangle_mesh=lambda path: compact_mesh,
    )
    fake_module.t = types.SimpleNamespace(io=types.SimpleNamespace(read_triangle_mesh=lambda path: compact_mesh))
    sys.modules["open3d"] = fake_module


_install_fake_open3d()
app_module = importlib.import_module("pcd_viewer_app")
layout_clouds = app_module.layout_clouds
preferred_render_mode_for_path = app_module.preferred_render_mode_for_path


class LayoutCloudsTests(unittest.TestCase):
    def test_preserve_coordinates_keeps_original_origin(self):
        cloud = _FakePointCloud()
        points = np.array(
            [
                [10.0, 0.0, 0.0],
                [11.0, 1.0, 0.0],
                [12.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        cloud.points = points

        positioned = layout_clouds([cloud], gap=1.0, preserve_coordinates=True)

        np.testing.assert_allclose(positioned[0].points, points)

    def test_preferred_mode_uses_mesh_for_glb(self):
        self.assertEqual(preferred_render_mode_for_path("foo.glb"), "Mesh")
        self.assertEqual(preferred_render_mode_for_path("foo.gltf"), "Mesh")
        self.assertEqual(preferred_render_mode_for_path("foo.ply"), "Points")

    def test_load_glb_returns_named_category_parts(self):
        parts = app_module.load_asset_parts("scene.glb", use_cuda=False)

        self.assertEqual([part.category for part in parts], ["car", "road"])
        self.assertTrue(all(part.preferred_mode == "Mesh" for part in parts))

    def test_single_mesh_glb_uses_compact_mesh_reader(self):
        parts = app_module.load_asset_parts("single.glb", use_cuda=False)

        self.assertEqual([part.category for part in parts], ["single"])
        self.assertEqual(len(parts[0].geometry.vertices), 3)

    def test_screenshot_path_uses_png_extension(self):
        path = app_module.default_screenshot_path("/tmp/out", "20260503_120000")

        self.assertTrue(path.endswith("render_20260503_120000.png"))

    def test_missing_category_widgets_only_returns_new_categories(self):
        missing = app_module.missing_category_widgets(["car", "road", "tree"], {"road"})

        self.assertEqual(missing, ["car", "tree"])


if __name__ == "__main__":
    unittest.main()
