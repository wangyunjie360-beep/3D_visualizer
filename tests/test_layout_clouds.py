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

    @classmethod
    def create_coordinate_frame(cls, size=1.0, origin=None):
        origin = np.zeros(3, dtype=float) if origin is None else np.asarray(origin, dtype=float)
        mesh = cls()
        mesh.vertices = np.asarray(
            [origin, origin + [size, 0, 0], origin + [0, size, 0], origin + [0, 0, size]],
            dtype=float,
        )
        mesh.triangles = np.zeros((0, 3), dtype=int)
        return mesh

    def transform(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        points = np.c_[self.vertices, np.ones(len(self.vertices), dtype=float)]
        self.vertices = (matrix @ points.T).T[:, :3]
        return self


class _FakeLineSet:
    def __init__(self):
        self.points = np.zeros((0, 3), dtype=float)
        self.lines = np.zeros((0, 2), dtype=int)
        self.colors = np.zeros((0, 3), dtype=float)

    def transform(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        points = np.c_[np.asarray(self.points, dtype=float), np.ones(len(self.points), dtype=float)]
        self.points = (matrix @ points.T).T[:, :3]
        return self


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
        LineSet=_FakeLineSet,
    )
    fake_module.utility = types.SimpleNamespace(
        Vector3dVector=lambda values: np.asarray(values, dtype=float),
        Vector2iVector=lambda values: np.asarray(values, dtype=int),
        Vector3iVector=lambda values: np.asarray(values, dtype=int),
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

    def test_render_scene_image_uses_open3d_019_signature(self):
        class FakeApp:
            def __init__(self):
                self.calls = []

            def render_to_image(self, scene, width, height):
                self.calls.append((scene, width, height))
                return "image"

        app = FakeApp()

        image = app_module.render_scene_image(app, "scene", 320, 240)

        self.assertEqual(image, "image")
        self.assertEqual(app.calls, [("scene", 320, 240)])

    def test_render_scene_image_applies_screenshot_scale(self):
        class FakeApp:
            def __init__(self):
                self.calls = []

            def render_to_image(self, scene, width, height):
                self.calls.append((scene, width, height))
                return "image"

        app = FakeApp()

        image = app_module.render_scene_image(app, "scene", 320, 240, scale=2.0)

        self.assertEqual(image, "image")
        self.assertEqual(app.calls, [("scene", 640, 480)])

    def test_screenshot_render_size_clamps_invalid_dimensions(self):
        self.assertEqual(app_module.screenshot_render_size(0, -2, 2.0), (1, 1))
        self.assertEqual(app_module.screenshot_render_size(320, 240, 1.5), (480, 360))

    def test_stitch_screenshot_arrays_keeps_layout_and_white_empty_cells(self):
        red = np.full((2, 3, 3), [255, 0, 0], dtype=np.uint8)
        green = np.full((2, 3, 3), [0, 255, 0], dtype=np.uint8)
        blue = np.full((2, 3, 3), [0, 0, 255], dtype=np.uint8)

        stitched = app_module.stitch_screenshot_arrays([red, green, blue], rows=2, cols=2)

        self.assertEqual(stitched.shape, (4, 6, 3))
        np.testing.assert_array_equal(stitched[0:2, 0:3], red)
        np.testing.assert_array_equal(stitched[0:2, 3:6], green)
        np.testing.assert_array_equal(stitched[2:4, 0:3], blue)
        np.testing.assert_array_equal(stitched[2:4, 3:6], np.full((2, 3, 3), 255, dtype=np.uint8))

    def test_camera_axis_marker_points_along_positive_x(self):
        marker = app_module.create_axis_marker("Camera", size=2.0)

        points = np.asarray(marker.points)
        lines = np.asarray(marker.lines)

        np.testing.assert_allclose(points[0], [0, 0, 0])
        self.assertTrue(np.all(points[1:, 0] > 0))
        self.assertEqual(lines.shape, (8, 2))

    def test_camera_axis_marker_uses_multiple_line_colors(self):
        marker = app_module.create_axis_marker("Camera", size=1.0)

        colors = np.asarray(marker.colors)

        self.assertEqual(colors.shape, (8, 3))
        self.assertGreater(len(np.unique(colors, axis=0)), 1)

    def test_camera_axis_marker_transform_moves_origin(self):
        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 3]

        marker = app_module.create_axis_marker("Camera", size=1.0, transform=transform)

        np.testing.assert_allclose(np.asarray(marker.points)[0], [1, 2, 3])

    def test_camera_axis_marker_material_is_thick_line_shader(self):
        material = types.SimpleNamespace(shader="", line_width=1.0)

        app_module.configure_axis_marker_material(material, "Camera")

        self.assertEqual(material.shader, "unlitLine")
        self.assertGreater(material.line_width, 1.0)

    def test_camera_axis_marker_material_uses_requested_line_width(self):
        material = types.SimpleNamespace(shader="", line_width=1.0)

        app_module.configure_axis_marker_material(material, "Camera", line_width=8.5)

        self.assertEqual(material.shader, "unlitLine")
        self.assertEqual(material.line_width, 8.5)

    def test_camera_line_width_is_clamped_to_slider_range(self):
        self.assertEqual(app_module.clamp_camera_line_width(0.1), 1.0)
        self.assertEqual(app_module.clamp_camera_line_width(99), 12.0)
        self.assertEqual(app_module.clamp_camera_line_width(6.5), 6.5)

    def test_default_axis_marker_material_stays_default_unlit(self):
        material = types.SimpleNamespace(shader="", line_width=1.0)

        app_module.configure_axis_marker_material(material, "Axis")

        self.assertEqual(material.shader, "defaultUnlit")
        self.assertEqual(material.line_width, 1.0)

    def test_missing_category_widgets_only_returns_new_categories(self):
        missing = app_module.missing_category_widgets(["car", "road", "tree"], {"road"})

        self.assertEqual(missing, ["car", "tree"])

    def test_parse_ply_semantic_color_map(self):
        header = [
            "ply\n",
            "comment class 0: floor rgb=[230, 25, 75]\n",
            "comment class 1: wall rgb=[60, 180, 75]\n",
            "end_header\n",
        ]

        mapping = app_module.parse_ply_semantic_color_map(header)

        self.assertEqual(mapping[(230, 25, 75)], "floor")
        self.assertEqual(mapping[(60, 180, 75)], "wall")

    def test_triangle_mesh_splits_by_vertex_color_names(self):
        mesh = _FakeTriangleMesh()
        mesh.vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
            ],
            dtype=float,
        )
        mesh.triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
        mesh.vertex_colors = np.array(
            [
                [230 / 255, 25 / 255, 75 / 255],
                [230 / 255, 25 / 255, 75 / 255],
                [230 / 255, 25 / 255, 75 / 255],
                [60 / 255, 180 / 255, 75 / 255],
                [60 / 255, 180 / 255, 75 / 255],
                [60 / 255, 180 / 255, 75 / 255],
            ],
            dtype=float,
        )

        parts = app_module.split_mesh_by_vertex_color(
            "scene.glb", mesh, {(230, 25, 75): "floor", (60, 180, 75): "wall"}
        )

        self.assertEqual([part.category for part in parts], ["floor", "wall"])
        self.assertEqual([len(part.geometry.triangles) for part in parts], [1, 1])


if __name__ == "__main__":
    unittest.main()
