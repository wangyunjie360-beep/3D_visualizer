import importlib
import inspect
from pathlib import Path
import sys
import types
import unittest
from tempfile import NamedTemporaryFile

import numpy as np


class _FakeBoundingBox:
    def __init__(self, points):
        self._points = np.asarray(points, dtype=float)

    def is_empty(self):
        return len(self._points) == 0

    def get_center(self):
        return self._points.mean(axis=0)

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

    def test_load_bin_as_pcd_normalizes_intensity_with_numpy_2(self):
        rows = np.array(
            [
                [1.0, 2.0, 3.0, 0.0],
                [4.0, 5.0, 6.0, 10.0],
            ],
            dtype=np.float32,
        )

        with NamedTemporaryFile(suffix=".bin") as file:
            rows.tofile(file.name)

            cloud = app_module.load_bin_as_pcd(file.name)

        np.testing.assert_allclose(cloud.points, rows[:, :3])
        np.testing.assert_allclose(cloud.colors, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

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

    def test_panel_width_only_reserves_space_when_visible(self):
        self.assertEqual(app_module.panel_width_for_visibility(True, 14), 336)
        self.assertEqual(app_module.panel_width_for_visibility(False, 14), 0)

    def test_restore_tab_sits_on_right_edge_middle_when_panel_hidden(self):
        rect = app_module.panel_restore_tab_frame(
            content_x=10,
            content_y=20,
            content_width=800,
            content_height=600,
            font_size=14,
            panel_visible=False,
        )

        self.assertEqual(rect, (779, 276, 31, 88))

    def test_hide_tab_sits_on_panel_left_edge_middle_when_panel_visible(self):
        rect = app_module.panel_hide_tab_frame(
            content_x=10,
            content_y=20,
            content_width=800,
            content_height=600,
            panel_width=336,
            font_size=14,
            panel_visible=True,
        )

        self.assertEqual(rect, (443, 276, 31, 88))

    def test_restore_tab_is_hidden_when_panel_visible(self):
        self.assertIsNone(
            app_module.panel_restore_tab_frame(
                content_x=10,
                content_y=20,
                content_width=800,
                content_height=600,
                font_size=14,
                panel_visible=True,
            )
        )

    def test_panel_arrow_labels_match_visibility_direction(self):
        self.assertEqual(app_module.panel_toggle_label(True), ">")
        self.assertEqual(app_module.panel_toggle_label(False), "<")

    def test_set_pane_background_syncs_scene_and_widget_color(self):
        class FakeScene:
            def __init__(self):
                self.backgrounds = []
                self.skybox_states = []

            def set_background(self, color):
                self.backgrounds.append(list(color))

            def show_skybox(self, enabled):
                self.skybox_states.append(enabled)

        class FakeWidget:
            def __init__(self):
                self.background_color = None
                self.redraws = 0

            def force_redraw(self):
                self.redraws += 1

        class FakeColor:
            def __init__(self, r, g, b, a):
                self.values = (r, g, b, a)

        pane = {
            "scene": FakeScene(),
            "widget": FakeWidget(),
            "gui_color": FakeColor,
        }

        app_module.set_pane_background(pane, [0.1, 0.2, 0.3, 1.0])

        self.assertEqual(pane["background"], [0.1, 0.2, 0.3, 1.0])
        self.assertEqual(pane["scene"].backgrounds, [[0.1, 0.2, 0.3, 1.0]])
        self.assertEqual(pane["scene"].skybox_states, [False])
        self.assertEqual(pane["widget"].background_color.values, (0.1, 0.2, 0.3, 1.0))
        self.assertEqual(pane["widget"].redraws, 1)

    def test_panel_overlay_controls_are_installed_after_panes_for_z_order(self):
        source = inspect.getsource(app_module.launch_viewer)

        self.assertLess(
            source.index("self._apply_layout(rebuild=True)"),
            source.index("self._install_panel_overlay_controls()"),
        )

    def test_panel_container_is_installed_after_panes_for_z_order(self):
        source = inspect.getsource(app_module.launch_viewer)

        self.assertLess(
            source.index("self._apply_layout(rebuild=True)"),
            source.index("self._install_panel_container()"),
        )

    def test_scene_widget_interaction_disables_cache_and_redraws_on_drag(self):
        class FakeWidget:
            def __init__(self):
                self.cache_states = []
                self.mouse_callback = None
                self.redraws = 0

            def enable_scene_caching(self, enabled):
                self.cache_states.append(enabled)

            def set_on_mouse(self, callback):
                self.mouse_callback = callback

            def force_redraw(self):
                self.redraws += 1

        event_types = types.SimpleNamespace(DRAG="drag", MOVE="move")
        callback_result = types.SimpleNamespace(IGNORED="ignored")
        widget = FakeWidget()

        app_module.configure_scene_widget_interaction(widget, event_types, callback_result)

        self.assertEqual(widget.cache_states, [False])
        self.assertIsNotNone(widget.mouse_callback)
        self.assertEqual(widget.mouse_callback(types.SimpleNamespace(type="move")), "ignored")
        self.assertEqual(widget.redraws, 0)
        self.assertEqual(widget.mouse_callback(types.SimpleNamespace(type="drag")), "ignored")
        self.assertEqual(widget.redraws, 1)

    def test_legacy_entrypoint_is_thin_wrapper(self):
        source_path = Path(app_module.__file__)
        source = source_path.read_text(encoding="utf-8")

        self.assertEqual(source_path.name, "pcd_viewer_app.py")
        self.assertLessEqual(len(source.splitlines()), 80)
        self.assertIn("from visualizer3d.cli import main", source)

    def test_selected_listbox_paths_returns_selected_files(self):
        class FakeListbox:
            def __init__(self):
                self.items = ["/tmp/a.ply", "/tmp/b.glb", "/tmp/c.bin"]

            def curselection(self):
                return (0, 2)

            def get(self, index):
                return self.items[index]

        paths = app_module.selected_listbox_paths(FakeListbox())

        self.assertEqual(paths, ["/tmp/a.ply", "/tmp/c.bin"])

    def test_render_parameter_rebuilds_preserve_camera(self):
        self.assertFalse(app_module.should_reset_camera_for_rebuild("point_size"))
        self.assertFalse(app_module.should_reset_camera_for_rebuild("voxel_size"))
        self.assertFalse(app_module.should_reset_camera_for_rebuild("axis"))
        self.assertTrue(app_module.should_reset_camera_for_rebuild("scene"))

    def test_maybe_setup_camera_skips_when_preserving_camera(self):
        class FakeWidget:
            def __init__(self):
                self.calls = []

            def setup_camera(self, *args):
                self.calls.append(args)

        widget = FakeWidget()
        bbox = _FakeBoundingBox([[0, 0, 0], [2, 4, 6]])

        did_setup = app_module.maybe_setup_camera(widget, bbox, reset_camera=False)

        self.assertFalse(did_setup)
        self.assertEqual(widget.calls, [])

    def test_maybe_setup_camera_resets_camera_when_requested(self):
        class FakeWidget:
            def __init__(self):
                self.calls = []

            def setup_camera(self, *args):
                self.calls.append(args)

        widget = FakeWidget()
        bbox = _FakeBoundingBox([[0, 0, 0], [2, 4, 6]])

        did_setup = app_module.maybe_setup_camera(widget, bbox, reset_camera=True)

        self.assertTrue(did_setup)
        self.assertEqual(len(widget.calls), 1)
        self.assertEqual(widget.calls[0][0], 60.0)
        self.assertIs(widget.calls[0][1], bbox)
        np.testing.assert_allclose(widget.calls[0][2], [1, 2, 3])

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
