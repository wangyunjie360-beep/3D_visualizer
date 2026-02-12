import argparse
import os
import re
import subprocess
import sys
from typing import Optional, Tuple

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point cloud viewer with launcher")
    parser.add_argument("--viewer", action="store_true", help="Run in viewer mode")
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Point cloud file to open (can be used multiple times)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Preferred device for loading point clouds",
    )
    parser.add_argument("--title", type=str, default="PointCloudViewer", help="Window title prefix")
    return parser.parse_args()


def _cuda_available() -> bool:
    try:
        return bool(o3d.core.cuda.is_available())
    except Exception:
        return False


def resolve_device(requested: str) -> Tuple[str, bool, str]:
    if requested == "cpu":
        return "cpu", False, "Device: CPU"
    if requested == "cuda":
        if _cuda_available():
            return "cuda", True, "Device: CUDA"
        return "cpu", False, "CUDA not available, fallback to CPU"
    if _cuda_available():
        return "cuda", True, "Device: CUDA (auto)"
    return "cpu", False, "CUDA not available, using CPU"


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
            norm = (intensities - intensities.min()) / intensities.ptp()
            colors = np.stack([norm, norm, norm], axis=1).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def load_point_cloud(path: str, use_cuda: bool) -> o3d.geometry.PointCloud:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        return load_bin_as_pcd(path)
    if ext in {".pcd", ".ply"}:
        if use_cuda:
            try:
                device = o3d.core.Device("CUDA:0")
                tpcd = o3d.t.io.read_point_cloud(path, device=device)
                return tpcd.to_legacy()
            except Exception:
                pass
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise ValueError(f"{os.path.basename(path)}: loaded empty point cloud.")
        return pcd
    raise ValueError(f"Unsupported file type: {ext}")


def parse_matrix(text: str) -> np.ndarray:
    tokens = re.split(r"[,\s]+", text.strip())
    tokens = [t for t in tokens if t]
    if len(tokens) != 16:
        raise ValueError("Matrix input must have 16 numbers.")
    vals = np.array([float(t) for t in tokens], dtype=float)
    return vals.reshape(4, 4)


def layout_clouds(clouds, gap: float) -> list:
    if not clouds:
        return []
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
        copy_cloud = o3d.geometry.PointCloud(cloud)
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


def split_clouds(clouds, pane_count: int) -> list:
    groups = [[] for _ in range(max(pane_count, 0))]
    if pane_count <= 0 or not clouds:
        return groups
    group_size = int(np.ceil(len(clouds) / pane_count))
    for idx, cloud in enumerate(clouds):
        group_idx = min(idx // group_size, pane_count - 1)
        groups[group_idx].append(cloud)
    return groups


def launch_viewer(file_paths, title: str, device_pref: str) -> None:
    from open3d.visualization import gui, rendering

    class ViewerWindow:
        def __init__(self, init_files):
            self._device, self._use_cuda, self._device_note = resolve_device(device_pref)
            self._clouds = []
            self._paths = []
            self._panes = []
            self._axis_transform = np.eye(4)
            self._updating = False
            self._gap = 1.0

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
            self._window.add_child(self._panel)

            self._file_label = gui.Label("No file loaded")
            self._panel.add_child(self._file_label)

            self._points_label = gui.Label("Points: -")
            self._panel.add_child(self._points_label)

            self._status_label = gui.Label(self._device_note)
            self._panel.add_child(self._status_label)

            open_btn = gui.Button("Open...")
            open_btn.set_on_clicked(self._on_open)
            self._panel.add_child(open_btn)

            self._panel_toggle = gui.Button("Hide Panel (H)")
            self._panel_toggle.set_on_clicked(self._on_toggle_panel)
            self._panel.add_child(self._panel_toggle)

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
            self._mode_combo.add_item("Points")
            self._mode_combo.add_item("Voxel")
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
            self._show_axis = gui.Checkbox("Show Axis")
            self._show_axis.checked = True
            self._show_axis.set_on_checked(self._on_axis_toggle)
            self._panel.add_child(self._show_axis)

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
                widget.scene.set_background([0, 0, 0, 1])
                self._window.add_child(widget)
                self._panes.append(
                    {
                        "widget": widget,
                        "scene": widget.scene,
                        "cloud_names": [],
                        "axis_name": f"axis_{len(self._panes)}",
                    }
                )
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
            panel_width = int(24 * self._window.theme.font_size) if self._panel.visible else 0
            pane_rect = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
            if self._panel.visible:
                self._panel.frame = gui.Rect(
                    r.get_right() - panel_width, r.y, panel_width, r.height
                )
            else:
                self._panel.frame = gui.Rect(r.get_right(), r.y, 0, r.height)

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
            self._panel_toggle.text = "Hide Panel (H)" if visible else "Show Panel (H)"
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
            for idx, (path, cloud) in enumerate(zip(self._paths, self._clouds), start=1):
                lines.append(f"{idx}. {os.path.basename(path)} - {len(cloud.points)} pts")
            self._points_label.text = "\n".join(lines)

        def _on_open(self):
            dlg = gui.FileDialog(gui.FileDialog.OPEN, "Open Point Cloud", self._window.theme)
            dlg.add_filter(".ply", "PLY")
            dlg.add_filter(".pcd", "PCD")
            dlg.add_filter(".bin", "BIN")
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
                pcd = load_point_cloud(filename, self._use_cuda)
            except Exception as exc:
                self._set_status(f"Load failed: {exc}")
                return
            self._clouds.append(pcd)
            self._paths.append(filename)
            self._pane_count = max(1, self._pane_count)
            self._pane_count_edit.int_value = self._pane_count
            self._update_file_labels()
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

        def _rebuild_scene(self):
            for pane in self._panes:
                self._clear_pane(pane)
            if not self._clouds:
                return

            groups = split_clouds(self._clouds, self._pane_count)
            mode = self._mode_combo.selected_text

            for pane_idx, pane in enumerate(self._panes[: self._pane_count]):
                group = groups[pane_idx] if pane_idx < len(groups) else []
                if not group:
                    continue
                positioned = layout_clouds(group, self._gap)
                for idx, cloud in enumerate(positioned):
                    if mode == "Voxel":
                        voxel_size = float(self._voxel_edit.double_value)
                        if voxel_size <= 0:
                            self._set_status("Voxel size must be > 0")
                            return
                        geom = o3d.geometry.VoxelGrid.create_from_point_cloud(
                            cloud, voxel_size=voxel_size
                        )
                    else:
                        geom = cloud

                    material = rendering.MaterialRecord()
                    material.shader = "defaultUnlit"
                    if mode == "Points":
                        material.point_size = float(self._point_edit.double_value)
                    name = f"cloud_{pane_idx}_{idx}"
                    pane["cloud_names"].append(name)
                    pane["scene"].add_geometry(name, geom, material)

                if self._show_axis.checked:
                    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                    axis.transform(self._axis_transform)
                    material = rendering.MaterialRecord()
                    material.shader = "defaultUnlit"
                    pane["scene"].add_geometry(pane["axis_name"], axis, material)

                bbox = combined_bbox(positioned)
                if bbox is not None and not bbox.is_empty():
                    pane["widget"].setup_camera(60.0, bbox, bbox.get_center())

        def _on_mode_changed(self, _text, _index):
            self._rebuild_scene()

        def _on_point_size_slider(self, value):
            if self._updating:
                return
            self._updating = True
            self._point_edit.double_value = value
            self._updating = False
            if self._mode_combo.selected_text == "Points":
                self._rebuild_scene()

        def _on_point_size_edit(self, value):
            if self._updating:
                return
            self._updating = True
            self._point_slider.double_value = value
            self._updating = False
            if self._mode_combo.selected_text == "Points":
                self._rebuild_scene()

        def _on_voxel_size_slider(self, value):
            if self._updating:
                return
            self._updating = True
            self._voxel_edit.double_value = value
            self._updating = False
            if self._mode_combo.selected_text == "Voxel":
                self._rebuild_scene()

        def _on_voxel_size_edit(self, value):
            if self._updating:
                return
            self._updating = True
            self._voxel_slider.double_value = value
            self._updating = False
            if self._mode_combo.selected_text == "Voxel":
                self._rebuild_scene()

        def _on_axis_toggle(self, _is_checked):
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
                self._rebuild_scene()
                self._set_status("Axis transform applied")
            except Exception as exc:
                self._set_status(f"Axis transform error: {exc}")

        def _on_reset_view(self):
            self._rebuild_scene()

    app = gui.Application.instance
    app.initialize()
    ViewerWindow(file_paths)
    app.run()


def spawn_viewer(paths, device_pref: str, title: str) -> None:
    if isinstance(paths, str):
        paths = [paths]
    cmd = [sys.executable]
    if not getattr(sys, "frozen", False):
        cmd.append(os.path.abspath(__file__))
    cmd += ["--viewer", "--device", device_pref, "--title", title]
    for path in paths:
        cmd += ["--file", path]
    subprocess.Popen(cmd, close_fds=False)


def launch_launcher(device_pref: str, title: str) -> None:
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        raise RuntimeError("tkinterdnd2 not installed. Please install it to use drag-and-drop.")

    root = TkinterDnD.Tk()
    root.title(f"{title} Launcher")
    root.geometry("520x320")

    info = tk.Label(root, text="Drag and drop .ply/.pcd/.bin files here, or click Open Files...")
    info.pack(padx=10, pady=10)

    listbox = tk.Listbox(root, width=80, height=10)
    listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def open_files():
        paths = filedialog.askopenfilenames(
            filetypes=[
                ("Point Clouds", "*.ply *.pcd *.bin"),
                ("All Files", "*.*"),
            ]
        )
        if paths:
            add_files(list(paths))

    def add_files(paths):
        if not paths:
            return
        for path in paths:
            listbox.insert(tk.END, path)
        spawn_viewer(paths, device_pref, title)

    def on_drop(event):
        files = root.tk.splitlist(event.data)
        add_files(list(files))

    btn = tk.Button(root, text="Open Files...", command=open_files)
    btn.pack(padx=10, pady=6)

    listbox.drop_target_register(DND_FILES)
    listbox.dnd_bind("<<Drop>>", on_drop)
    root.mainloop()


def main() -> None:
    args = parse_args()
    if args.viewer or args.file:
        launch_viewer(args.file, args.title, args.device)
        return
    launch_launcher(args.device, args.title)


if __name__ == "__main__":
    main()
