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


def _install_fake_open3d():
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
    fake_module.t = types.SimpleNamespace(io=types.SimpleNamespace(read_triangle_mesh=lambda path: _FakeTriangleMesh()))
    sys.modules["open3d"] = fake_module


_install_fake_open3d()
layout_clouds = importlib.import_module("pcd_viewer_app").layout_clouds
preferred_render_mode_for_path = importlib.import_module("pcd_viewer_app").preferred_render_mode_for_path


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


if __name__ == "__main__":
    unittest.main()
