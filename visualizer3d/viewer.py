import os

import numpy as np
import open3d as o3d

from .assets import load_asset_parts
from .device import resolve_device
from .geometry import (
    combined_bbox,
    configure_axis_marker_material,
    create_axis_marker,
    geometry_point_count,
    layout_clouds,
    parse_matrix,
    split_clouds,
)
from .scene import (
    configure_scene_widget_interaction,
    maybe_setup_camera,
    set_pane_background,
    should_reset_camera_for_rebuild,
)
from .screenshots import default_screenshot_path, render_scene_image, stitch_screenshot_arrays
from .ui_layout import (
    missing_category_widgets,
    panel_hide_tab_frame,
    panel_restore_tab_frame,
    panel_toggle_label,
    panel_width_for_visibility,
)


def launch_viewer(file_paths, title: str, device_pref: str) -> None:
    from open3d.visualization import gui, rendering

    class ViewerWindow:
        def __init__(self, init_files):
            self._device, self._use_cuda, self._device_note = resolve_device(device_pref)
            self._parts = []
            self._clouds = []
            self._preferred_modes = []
            self._categories = []
            self._visible_categories = set()
            self._category_checkboxes = {}
            self._screenshot_was_pending = False
            self._screenshot_original_backgrounds = []
            self._paths = []
            self._panes = []
            self._axis_transform = np.eye(4)
            self._updating = False
            self._gap = 1.0
            self._preserve_coordinates = True
            self._screenshot_scale = 2.0
            self._camera_line_width = 5.0

            self._pane_count = max(1, len(init_files) if init_files else 1)
            self._auto_layout = True
            self._layout_rows = 1
            self._layout_cols = 1

            self._app = gui.Application.instance
            self._window = self._app.create_window(f"{title}", 1280, 800)
            self._window.set_on_layout(self._on_layout)
            self._window.set_on_key(self._on_key)

            em = self._window.theme.font_size
            margin = int(0.5 * em)
            self._panel = gui.Vert(0, gui.Margins(margin, margin, margin, margin))
            self._panel_is_installed = False

            self._file_label = gui.Label("No file loaded")
            self._panel.add_child(self._file_label)

            self._points_label = gui.Label("Points: -")
            self._panel.add_child(self._points_label)

            self._status_label = gui.Label(self._device_note)
            self._panel.add_child(self._status_label)

            open_btn = gui.Button("Open...")
            open_btn.set_on_clicked(self._on_open)
            self._panel.add_child(open_btn)

            self._panel_toggle = None
            self._panel_restore_tab = None

            self._panel.add_fixed(0.5 * em)
            self._panel.add_child(gui.Label("Pane Layout"))
            self._auto_layout_cb = gui.Checkbox("Auto layout")
            self._auto_layout_cb.checked = True
            self._auto_layout_cb.set_on_checked(self._on_auto_layout)
            self._panel.add_child(self._auto_layout_cb)

            self._pane_count_edit = gui.NumberEdit(gui.NumberEdit.INT)
            self._pane_count_edit.int_value = self._pane_count
            self._pane_count_edit.set_limits(1, 64)
            self._pane_count_edit.set_on_value_changed(self._on_pane_count_changed)
            self._panel.add_child(gui.Label("Panes"))
            self._panel.add_child(self._pane_count_edit)

            grid = gui.VGrid(2, 0.25 * em)
            self._rows_edit = gui.NumberEdit(gui.NumberEdit.INT)
            self._rows_edit.int_value = 1
            self._rows_edit.set_limits(1, 64)
            self._rows_edit.set_on_value_changed(self._on_rows_changed)
            self._cols_edit = gui.NumberEdit(gui.NumberEdit.INT)
            self._cols_edit.int_value = 1
            self._cols_edit.set_limits(1, 64)
            self._cols_edit.set_on_value_changed(self._on_cols_changed)
            grid.add_child(gui.Label("Rows"))
            grid.add_child(self._rows_edit)
            grid.add_child(gui.Label("Cols"))
            grid.add_child(self._cols_edit)
            self._panel.add_child(grid)

            self._panel.add_fixed(0.5 * em)
            self._panel.add_child(gui.Label("Render Mode"))
            self._mode_combo = gui.Combobox()
            self._mode_combo.add_item("Auto")
            self._mode_combo.add_item("Points")
            self._mode_combo.add_item("Voxel")
            self._mode_combo.add_item("Mesh")
            self._mode_combo.selected_index = 0
            self._mode_combo.set_on_selection_changed(self._on_mode_changed)
            self._panel.add_child(self._mode_combo)

            self._panel.add_fixed(0.5 * em)
            self._panel.add_child(gui.Label("Point Size"))
            self._point_slider = gui.Slider(gui.Slider.DOUBLE)
            self._point_slider.set_on_value_changed(self._on_point_size_slider)
            self._point_slider.double_value = 3.0
            self._point_slider.set_limits(1.0, 10.0)
            self._panel.add_child(self._point_slider)
            self._point_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            self._point_edit.double_value = 3.0
            self._point_edit.set_limits(1.0, 10.0)
            self._point_edit.set_on_value_changed(self._on_point_size_edit)
            self._panel.add_child(self._point_edit)

            self._panel.add_fixed(0.5 * em)
            self._panel.add_child(gui.Label("Voxel Size"))
            self._voxel_slider = gui.Slider(gui.Slider.DOUBLE)
            self._voxel_slider.set_on_value_changed(self._on_voxel_size_slider)
            self._voxel_slider.double_value = 0.05
            self._voxel_slider.set_limits(0.01, 1.0)
            self._panel.add_child(self._voxel_slider)
            self._voxel_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            self._voxel_edit.double_value = 0.05
            self._voxel_edit.set_limits(0.01, 1.0)
            self._voxel_edit.set_on_value_changed(self._on_voxel_size_edit)
            self._panel.add_child(self._voxel_edit)

            self._panel.add_fixed(0.5 * em)
            self._panel.add_child(gui.Label("Categories"))
            self._category_list = gui.Vert(0, gui.Margins(0, 0, 0, 0))
            self._panel.add_child(self._category_list)

            screenshot_btn = gui.Button("Screenshot PNG")
            screenshot_btn.set_on_clicked(self._on_screenshot)
            self._panel.add_child(screenshot_btn)
            self._panel.add_child(gui.Label("Screenshot Scale"))
            self._screenshot_scale_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            self._screenshot_scale_edit.double_value = self._screenshot_scale
            self._screenshot_scale_edit.set_limits(1.0, 4.0)
            self._screenshot_scale_edit.set_on_value_changed(self._on_screenshot_scale)
            self._panel.add_child(self._screenshot_scale_edit)

            self._panel.add_fixed(0.5 * em)
            self._show_axis = gui.Checkbox("Show Axis")
            self._show_axis.checked = True
            self._show_axis.set_on_checked(self._on_axis_toggle)
            self._panel.add_child(self._show_axis)

            self._panel.add_child(gui.Label("Axis Style"))
            self._axis_style = gui.Combobox()
            self._axis_style.add_item("Axis")
            self._axis_style.add_item("Camera")
            self._axis_style.selected_index = 0
            self._axis_style.set_on_selection_changed(self._on_axis_style_changed)
            self._panel.add_child(self._axis_style)

            self._panel.add_child(gui.Label("Camera Line Width"))
            self._camera_line_width_slider = gui.Slider(gui.Slider.DOUBLE)
            self._camera_line_width_slider.double_value = self._camera_line_width
            self._camera_line_width_slider.set_limits(1.0, 12.0)
            self._camera_line_width_slider.set_on_value_changed(
                self._on_camera_line_width
            )
            self._panel.add_child(self._camera_line_width_slider)

            self._preserve_coords_cb = gui.Checkbox("Preserve Coordinates")
            self._preserve_coords_cb.checked = self._preserve_coordinates
            self._preserve_coords_cb.set_on_checked(self._on_preserve_coordinates)
            self._panel.add_child(self._preserve_coords_cb)

            self._panel.add_fixed(0.5 * em)
            self._panel.add_child(gui.Label("Axis Input Mode"))
            self._axis_mode = gui.Combobox()
            self._axis_mode.add_item("Quaternion + Translation")
            self._axis_mode.add_item("Matrix 4x4")
            self._axis_mode.selected_index = 0
            self._axis_mode.set_on_selection_changed(self._on_axis_mode_changed)
            self._panel.add_child(self._axis_mode)

            self._quat_inputs = self._build_quat_inputs()
            self._panel.add_child(self._quat_inputs)

            self._matrix_label = gui.Label("Matrix 4x4 (row-major)")
            self._panel.add_child(self._matrix_label)
            self._matrix_edit = gui.TextEdit()
            self._matrix_edit.text_value = (
                "1 0 0 0\n"
                "0 1 0 0\n"
                "0 0 1 0\n"
                "0 0 0 1\n"
            )
            self._panel.add_child(self._matrix_edit)
            self._matrix_label.visible = False
            self._matrix_edit.visible = False

            apply_btn = gui.Button("Apply Axis Transform")
            apply_btn.set_on_clicked(self._on_apply_axis)
            self._panel.add_child(apply_btn)

            reset_btn = gui.Button("Reset View")
            reset_btn.set_on_clicked(self._on_reset_view)
            self._panel.add_child(reset_btn)

            if init_files:
                for path in init_files:
                    self._add_file(path, rebuild=False)
            self._apply_layout(rebuild=True)
            self._install_panel_container()
            self._install_panel_overlay_controls()

        def _install_panel_container(self):
            if self._panel_is_installed:
                return
            self._window.add_child(self._panel)
            self._panel_is_installed = True
            self._window.set_needs_layout()
            self._window.post_redraw()

        def _install_panel_overlay_controls(self):
            if self._panel_toggle is not None:
                return
            self._panel_toggle = gui.Button(panel_toggle_label(True))
            self._panel_toggle.set_on_clicked(self._on_toggle_panel)
            self._window.add_child(self._panel_toggle)

            self._panel_restore_tab = gui.Button("<")
            self._panel_restore_tab.visible = False
            self._panel_restore_tab.set_on_clicked(lambda: self._set_panel_visible(True))
            self._window.add_child(self._panel_restore_tab)
            self._window.set_needs_layout()
            self._window.post_redraw()

        def _build_quat_inputs(self) -> gui.Widget:
            grid = gui.VGrid(2, 0.25 * self._window.theme.font_size)
            self._qw = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            self._qx = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            self._qy = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            self._qz = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            self._tx = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            self._ty = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            self._tz = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            self._qw.double_value = 1.0
            for w in (self._qx, self._qy, self._qz, self._tx, self._ty, self._tz):
                w.double_value = 0.0
            grid.add_child(gui.Label("qw"))
            grid.add_child(self._qw)
            grid.add_child(gui.Label("qx"))
            grid.add_child(self._qx)
            grid.add_child(gui.Label("qy"))
            grid.add_child(self._qy)
            grid.add_child(gui.Label("qz"))
            grid.add_child(self._qz)
            grid.add_child(gui.Label("tx"))
            grid.add_child(self._tx)
            grid.add_child(gui.Label("ty"))
            grid.add_child(self._ty)
            grid.add_child(gui.Label("tz"))
            grid.add_child(self._tz)
            return grid

        def _ensure_panes(self, count: int):
            while len(self._panes) < count:
                widget = gui.SceneWidget()
                widget.scene = rendering.Open3DScene(self._window.renderer)
                configure_scene_widget_interaction(
                    widget,
                    gui.MouseEvent.Type,
                    gui.Widget.EventCallbackResult,
                )
                self._window.add_child(widget)
                pane = {
                    "widget": widget,
                    "scene": widget.scene,
                    "background": [0, 0, 0, 1],
                    "gui_color": gui.Color,
                    "cloud_names": [],
                    "axis_name": f"axis_{len(self._panes)}",
                }
                set_pane_background(pane, pane["background"])
                self._panes.append(pane)
            for idx, pane in enumerate(self._panes):
                pane["widget"].visible = idx < count

        def _compute_layout(self):
            if self._auto_layout:
                panes = max(1, self._pane_count)
                cols = int(np.ceil(np.sqrt(panes)))
                rows = int(np.ceil(panes / cols))
                self._layout_rows = rows
                self._layout_cols = cols
                self._rows_edit.int_value = rows
                self._cols_edit.int_value = cols
                self._rows_edit.enabled = False
                self._cols_edit.enabled = False
                self._pane_count_edit.enabled = True
            else:
                rows = max(1, int(self._rows_edit.int_value))
                cols = max(1, int(self._cols_edit.int_value))
                if rows * cols < self._pane_count:
                    cols = int(np.ceil(self._pane_count / rows))
                    self._cols_edit.int_value = cols
                self._layout_rows = rows
                self._layout_cols = cols
                self._rows_edit.enabled = True
                self._cols_edit.enabled = True
                self._pane_count_edit.enabled = False

        def _apply_layout(self, rebuild: bool):
            self._compute_layout()
            self._ensure_panes(self._pane_count)
            self._window.set_needs_layout()
            if rebuild:
                self._rebuild_scene()

        def _on_layout(self, _ctx):
            r = self._window.content_rect
            font_size = self._window.theme.font_size
            panel_width = panel_width_for_visibility(self._panel.visible, font_size)
            pane_rect = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
            if self._panel.visible:
                self._panel.frame = gui.Rect(
                    r.get_right() - panel_width, r.y, panel_width, r.height
                )
            else:
                self._panel.frame = gui.Rect(r.get_right(), r.y, 0, r.height)

            hide_tab = panel_hide_tab_frame(
                r.x,
                r.y,
                r.width,
                r.height,
                panel_width,
                font_size,
                self._panel.visible,
            )
            if self._panel_toggle is None:
                pass
            elif hide_tab is None:
                self._panel_toggle.visible = False
                self._panel_toggle.frame = gui.Rect(r.get_right(), r.y, 0, 0)
            else:
                self._panel_toggle.visible = True
                self._panel_toggle.frame = gui.Rect(*hide_tab)

            restore_tab = panel_restore_tab_frame(
                r.x,
                r.y,
                r.width,
                r.height,
                font_size,
                self._panel.visible,
            )
            if self._panel_restore_tab is None:
                pass
            elif restore_tab is None:
                self._panel_restore_tab.visible = False
                self._panel_restore_tab.frame = gui.Rect(r.get_right(), r.y, 0, 0)
            else:
                self._panel_restore_tab.visible = True
                self._panel_restore_tab.frame = gui.Rect(*restore_tab)

            rows = max(1, self._layout_rows)
            cols = max(1, self._layout_cols)
            cell_w = pane_rect.width / cols
            cell_h = pane_rect.height / rows

            for idx, pane in enumerate(self._panes[: self._pane_count]):
                row = idx // cols
                col = idx % cols
                x = int(pane_rect.x + col * cell_w)
                y = int(pane_rect.y + row * cell_h)
                w = int(cell_w)
                h = int(cell_h)
                pane["widget"].frame = gui.Rect(x, y, w, h)

        def _set_status(self, text: str) -> None:
            self._status_label.text = text

        def _set_panel_visible(self, visible: bool) -> None:
            self._panel.visible = visible
            if self._panel_toggle is not None:
                self._panel_toggle.visible = visible
                self._panel_toggle.text = panel_toggle_label(visible)
            if self._panel_restore_tab is not None:
                self._panel_restore_tab.visible = not visible
            self._window.set_needs_layout()
            self._window.post_redraw()

        def _on_toggle_panel(self):
            self._set_panel_visible(not self._panel.visible)

        def _on_key(self, event):
            if event.type == gui.KeyEvent.DOWN and event.key == gui.KeyName.H:
                self._set_panel_visible(not self._panel.visible)
                return True
            return False

        def _update_file_labels(self):
            if not self._paths:
                self._file_label.text = "No file loaded"
                self._points_label.text = "Points: -"
                return
            self._file_label.text = f"Files: {len(self._paths)}"
            lines = []
            for idx, part in enumerate(self._parts, start=1):
                lines.append(
                    f"{idx}. {os.path.basename(part.path)} [{part.category}] - {geometry_point_count(part.geometry)} items"
                )
            self._points_label.text = "\n".join(lines)

        def _refresh_category_list(self):
            for category in missing_category_widgets(
                self._categories, self._category_checkboxes
            ):
                checkbox = gui.Checkbox(category)
                checkbox.checked = category in self._visible_categories
                checkbox.set_on_checked(
                    lambda is_checked, category=category: self._on_category_checked(
                        category, is_checked
                    )
                )
                self._category_checkboxes[category] = checkbox
                self._category_list.add_child(checkbox)
            for category, checkbox in self._category_checkboxes.items():
                checkbox.checked = category in self._visible_categories

        def _on_category_checked(self, category: str, is_checked: bool):
            if is_checked:
                self._visible_categories.add(category)
            else:
                self._visible_categories.discard(category)
            self._rebuild_scene()

        def _on_screenshot(self):
            if not self._panes:
                self._set_status("Screenshot failed: no scene")
                return
            path = default_screenshot_path(os.getcwd())
            try:
                self._screenshot_original_backgrounds = []
                for pane in self._panes[: self._pane_count]:
                    self._screenshot_original_backgrounds.append(pane["background"])
                    set_pane_background(pane, [1, 1, 1, 1])
                self._screenshot_was_pending = True

                images = []
                for pane in self._panes[: self._pane_count]:
                    frame = pane["widget"].frame
                    images.append(
                        render_scene_image(
                            self._app,
                            pane["scene"],
                            frame.width,
                            frame.height,
                            self._screenshot_scale,
                        )
                    )
                stitched = stitch_screenshot_arrays(
                    images, self._layout_rows, self._layout_cols
                )
                if not o3d.io.write_image(path, o3d.geometry.Image(stitched), 9):
                    raise RuntimeError("Open3D failed to write image")
                self._set_status(f"Screenshot saved: {path}")
            except Exception as exc:
                self._set_status(f"Screenshot failed: {exc}")
            finally:
                self._restore_screenshot_backgrounds()

        def _on_screenshot_scale(self, value: float):
            self._screenshot_scale = min(4.0, max(1.0, float(value)))
            self._screenshot_scale_edit.double_value = self._screenshot_scale
            self._set_status(f"Screenshot scale: {self._screenshot_scale:g}x")

        def _restore_screenshot_backgrounds(self):
            if not self._screenshot_was_pending:
                return
            for pane, background in zip(
                self._panes[: self._pane_count], self._screenshot_original_backgrounds
            ):
                set_pane_background(pane, background)
            self._screenshot_was_pending = False
            self._screenshot_original_backgrounds = []
            self._window.post_redraw()

        def _on_open(self):
            dlg = gui.FileDialog(gui.FileDialog.OPEN, "Open 3D File", self._window.theme)
            dlg.add_filter(".ply", "PLY")
            dlg.add_filter(".pcd", "PCD")
            dlg.add_filter(".bin", "BIN")
            dlg.add_filter(".glb", "GLB")
            dlg.add_filter(".gltf", "GLTF")
            dlg.add_filter("", "All files")
            if self._paths:
                dlg.set_path(os.path.dirname(self._paths[-1]))
            else:
                dlg.set_path(os.getcwd())
            dlg.set_on_cancel(self._window.close_dialog)
            dlg.set_on_done(self._on_open_done)
            self._window.show_dialog(dlg)

        def _on_open_done(self, filename):
            self._window.close_dialog()
            if filename:
                self._add_file(filename, rebuild=True)

        def _add_file(self, filename: str, rebuild: bool) -> None:
            try:
                parts = load_asset_parts(filename, self._use_cuda)
            except Exception as exc:
                self._set_status(f"Load failed: {exc}")
                return
            for part in parts:
                self._parts.append(part)
                self._clouds.append(part.geometry)
                self._preferred_modes.append(part.preferred_mode)
                if part.category not in self._categories:
                    self._categories.append(part.category)
                    self._visible_categories.add(part.category)
            self._paths.append(filename)
            self._pane_count = max(1, self._pane_count)
            self._pane_count_edit.int_value = self._pane_count
            self._update_file_labels()
            self._refresh_category_list()
            self._set_status(self._device_note)
            if rebuild:
                self._apply_layout(rebuild=True)

        def _clear_pane(self, pane):
            for name in pane["cloud_names"]:
                if pane["scene"].has_geometry(name):
                    pane["scene"].remove_geometry(name)
            pane["cloud_names"] = []
            axis_name = pane["axis_name"]
            if pane["scene"].has_geometry(axis_name):
                pane["scene"].remove_geometry(axis_name)

        def _rebuild_scene(self, reason: str = "scene"):
            reset_camera = should_reset_camera_for_rebuild(reason)
            for pane in self._panes:
                self._clear_pane(pane)
            visible_indices = [
                idx
                for idx, part in enumerate(self._parts)
                if part.category in self._visible_categories
            ]
            if not visible_indices:
                return

            groups = split_clouds(visible_indices, self._pane_count)
            mode = self._mode_combo.selected_text

            for pane_idx, pane in enumerate(self._panes[: self._pane_count]):
                index_group = groups[pane_idx] if pane_idx < len(groups) else []
                if not index_group:
                    continue
                group = [self._clouds[index_] for index_ in index_group]
                positioned = layout_clouds(
                    group,
                    self._gap,
                    preserve_coordinates=self._preserve_coordinates,
                )
                for idx, cloud in enumerate(positioned):
                    preferred_mode = self._preferred_modes[index_group[idx]]
                    active_mode = preferred_mode if mode == "Auto" else mode
                    if active_mode == "Voxel" and isinstance(cloud, o3d.geometry.PointCloud):
                        voxel_size = float(self._voxel_edit.double_value)
                        if voxel_size <= 0:
                            self._set_status("Voxel size must be > 0")
                            return
                        geom = o3d.geometry.VoxelGrid.create_from_point_cloud(
                            cloud, voxel_size=voxel_size
                        )
                    elif active_mode == "Points" and isinstance(cloud, o3d.geometry.TriangleMesh):
                        geom = cloud.sample_points_uniformly(
                            number_of_points=max(2048, len(cloud.triangles) * 10)
                        )
                    elif active_mode == "Mesh" and isinstance(cloud, o3d.geometry.PointCloud):
                        geom = cloud
                    else:
                        geom = cloud

                    material = rendering.MaterialRecord()
                    material.shader = "defaultLit" if isinstance(geom, o3d.geometry.TriangleMesh) else "defaultUnlit"
                    if active_mode == "Points":
                        material.point_size = float(self._point_edit.double_value)
                    name = f"cloud_{pane_idx}_{idx}"
                    pane["cloud_names"].append(name)
                    pane["scene"].add_geometry(name, geom, material)

                if self._show_axis.checked:
                    axis_style = self._axis_style.selected_text
                    axis = create_axis_marker(
                        axis_style,
                        size=1.0,
                        transform=self._axis_transform,
                    )
                    material = rendering.MaterialRecord()
                    configure_axis_marker_material(
                        material,
                        axis_style,
                        line_width=self._camera_line_width,
                    )
                    pane["scene"].add_geometry(pane["axis_name"], axis, material)

                bbox = combined_bbox(positioned)
                maybe_setup_camera(pane["widget"], bbox, reset_camera=reset_camera)

        def _on_mode_changed(self, _text, _index):
            self._rebuild_scene()

        def _on_point_size_slider(self, value):
            if self._updating:
                return
            self._updating = True
            self._point_edit.double_value = value
            self._updating = False
            if self._mode_combo.selected_text in {"Points", "Auto"}:
                self._rebuild_scene(reason="point_size")

        def _on_point_size_edit(self, value):
            if self._updating:
                return
            self._updating = True
            self._point_slider.double_value = value
            self._updating = False
            if self._mode_combo.selected_text in {"Points", "Auto"}:
                self._rebuild_scene(reason="point_size")

        def _on_voxel_size_slider(self, value):
            if self._updating:
                return
            self._updating = True
            self._voxel_edit.double_value = value
            self._updating = False
            if self._mode_combo.selected_text in {"Voxel", "Auto"}:
                self._rebuild_scene(reason="voxel_size")

        def _on_voxel_size_edit(self, value):
            if self._updating:
                return
            self._updating = True
            self._voxel_slider.double_value = value
            self._updating = False
            if self._mode_combo.selected_text in {"Voxel", "Auto"}:
                self._rebuild_scene(reason="voxel_size")

        def _on_axis_toggle(self, _is_checked):
            self._rebuild_scene(reason="axis")

        def _on_axis_style_changed(self, _text, _index):
            self._rebuild_scene(reason="axis")

        def _on_camera_line_width(self, value):
            self._camera_line_width = clamp_camera_line_width(value)
            self._camera_line_width_slider.double_value = self._camera_line_width
            self._set_status(f"Camera line width: {self._camera_line_width:g}")
            if self._axis_style.selected_text == "Camera" and self._show_axis.checked:
                self._rebuild_scene(reason="axis")

        def _on_preserve_coordinates(self, is_checked):
            self._preserve_coordinates = bool(is_checked)
            self._rebuild_scene()

        def _on_axis_mode_changed(self, text, _index):
            is_matrix = text == "Matrix 4x4"
            self._quat_inputs.visible = not is_matrix
            self._matrix_label.visible = is_matrix
            self._matrix_edit.visible = is_matrix

        def _on_auto_layout(self, is_checked):
            self._auto_layout = bool(is_checked)
            self._apply_layout(rebuild=True)

        def _on_pane_count_changed(self, value):
            if self._auto_layout:
                self._pane_count = max(1, int(value))
                self._apply_layout(rebuild=True)

        def _on_rows_changed(self, value):
            if not self._auto_layout:
                self._layout_rows = max(1, int(value))
                self._pane_count = max(1, self._layout_rows * self._layout_cols)
                self._apply_layout(rebuild=True)

        def _on_cols_changed(self, value):
            if not self._auto_layout:
                self._layout_cols = max(1, int(value))
                self._pane_count = max(1, self._layout_rows * self._layout_cols)
                self._apply_layout(rebuild=True)

        def _on_apply_axis(self):
            try:
                if self._axis_mode.selected_text == "Matrix 4x4":
                    mat = parse_matrix(self._matrix_edit.text_value)
                    self._axis_transform = mat
                else:
                    q = np.array(
                        [
                            self._qw.double_value,
                            self._qx.double_value,
                            self._qy.double_value,
                            self._qz.double_value,
                        ],
                        dtype=float,
                    )
                    t = np.array(
                        [
                            self._tx.double_value,
                            self._ty.double_value,
                            self._tz.double_value,
                        ],
                        dtype=float,
                    )
                    R = o3d.geometry.get_rotation_matrix_from_quaternion(q)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t
                    self._axis_transform = T
                self._rebuild_scene(reason="axis")
                self._set_status("Axis transform applied")
            except Exception as exc:
                self._set_status(f"Axis transform error: {exc}")

        def _on_reset_view(self):
            self._rebuild_scene()

    app = gui.Application.instance
    app.initialize()
    ViewerWindow(file_paths)
    app.run()
