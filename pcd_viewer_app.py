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
    parser.add_argument("--file", type=str, default=None, help="Point cloud file to open")
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


def launch_viewer(file_path: Optional[str], title: str, device_pref: str) -> None:
    from open3d.visualization import gui, rendering

    class ViewerWindow:
        def __init__(self, init_file: Optional[str]):
            self._title = title
            self._device, self._use_cuda, self._device_note = resolve_device(device_pref)
            self._cloud = None
            self._cloud_name = None
            self._axis = None
            self._axis_name = "axis"
            self._axis_transform = np.eye(4)
            self._updating = False

            self._app = gui.Application.instance
            self._window = self._app.create_window(f"{self._title}", 1280, 800)
            self._window.set_on_layout(self._on_layout)

            self._scene = gui.SceneWidget()
            self._scene.scene = rendering.Open3DScene(self._window.renderer)
            self._scene.scene.set_background([0, 0, 0, 1])
            self._window.add_child(self._scene)

            em = self._window.theme.font_size
            margin = int(0.5 * em)
            self._panel = gui.Vert(0, gui.Margins(margin, margin, margin, margin))
            self._window.add_child(self._panel)

            self._file_label = gui.Label("No file loaded")
            self._panel.add_child(self._file_label)

            self._status_label = gui.Label(self._device_note)
            self._panel.add_child(self._status_label)

            open_btn = gui.Button("Open...")
            open_btn.set_on_clicked(self._on_open)
            self._panel.add_child(open_btn)

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
            self._panel.add_child(self._axis_mode)

            self._quat_inputs = self._build_quat_inputs()
            self._panel.add_child(self._quat_inputs)

            self._panel.add_child(gui.Label("Matrix 4x4 (row-major)"))
            self._matrix_edit = gui.TextEdit()
            self._matrix_edit.text_value = (
                "1 0 0 0\n"
                "0 1 0 0\n"
                "0 0 1 0\n"
                "0 0 0 1\n"
            )
            self._panel.add_child(self._matrix_edit)

            apply_btn = gui.Button("Apply Axis Transform")
            apply_btn.set_on_clicked(self._on_apply_axis)
            self._panel.add_child(apply_btn)

            reset_btn = gui.Button("Reset View")
            reset_btn.set_on_clicked(self._on_reset_view)
            self._panel.add_child(reset_btn)

            if init_file:
                self._load_file(init_file)

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

        def _on_layout(self, ctx):
            r = self._window.content_rect
            panel_width = int(24 * self._window.theme.font_size)
            self._scene.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
            self._panel.frame = gui.Rect(
                r.get_right() - panel_width, r.y, panel_width, r.height
            )

        def _set_status(self, text: str) -> None:
            self._status_label.text = text

        def _on_open(self):
            dlg = gui.FileDialog(gui.FileDialog.OPEN, "Open Point Cloud", self._window.theme)
            dlg.add_filter(".ply", "PLY")
            dlg.add_filter(".pcd", "PCD")
            dlg.add_filter(".bin", "BIN")
            dlg.add_filter("", "All files")
            dlg.set_on_cancel(self._window.close_dialog)
            dlg.set_on_done(self._on_open_done)
            self._window.show_dialog(dlg)

        def _on_open_done(self, filename):
            self._window.close_dialog()
            if filename:
                self._load_file(filename)

        def _load_file(self, filename: str) -> None:
            try:
                pcd = load_point_cloud(filename, self._use_cuda)
            except Exception as exc:
                self._set_status(f"Load failed: {exc}")
                return
            self._cloud = pcd
            self._cloud_name = "cloud"
            self._file_label.text = os.path.basename(filename)
            self._set_status(self._device_note)
            self._rebuild_scene()

        def _clear_cloud(self):
            if self._cloud_name and self._scene.scene.has_geometry(self._cloud_name):
                self._scene.scene.remove_geometry(self._cloud_name)

        def _clear_axis(self):
            if self._scene.scene.has_geometry(self._axis_name):
                self._scene.scene.remove_geometry(self._axis_name)

        def _rebuild_scene(self):
            if self._cloud is None:
                return
            self._clear_cloud()
            mode = self._mode_combo.selected_text
            if mode == "Voxel":
                voxel_size = float(self._voxel_edit.double_value)
                if voxel_size <= 0:
                    self._set_status("Voxel size must be > 0")
                    return
                geom = o3d.geometry.VoxelGrid.create_from_point_cloud(
                    self._cloud, voxel_size=voxel_size
                )
                material = rendering.MaterialRecord()
                material.shader = "defaultUnlit"
                self._scene.scene.add_geometry(self._cloud_name, geom, material)
            else:
                material = rendering.MaterialRecord()
                material.shader = "defaultUnlit"
                material.point_size = float(self._point_edit.double_value)
                self._scene.scene.add_geometry(self._cloud_name, self._cloud, material)
            self._update_axis_geometry()
            self._fit_to_scene()

        def _fit_to_scene(self):
            if self._cloud is None:
                return
            bbox = self._cloud.get_axis_aligned_bounding_box()
            if not bbox.is_empty():
                self._scene.setup_camera(60.0, bbox, bbox.get_center())

        def _update_axis_geometry(self):
            self._clear_axis()
            if not self._show_axis.checked:
                return
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            axis.transform(self._axis_transform)
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            self._scene.scene.add_geometry(self._axis_name, axis, material)

        def _on_mode_changed(self, text, _index):
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

        def _on_axis_toggle(self, is_checked):
            self._update_axis_geometry()

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
                self._update_axis_geometry()
                self._set_status("Axis transform applied")
            except Exception as exc:
                self._set_status(f"Axis transform error: {exc}")

        def _on_reset_view(self):
            self._fit_to_scene()

    app = gui.Application.instance
    app.initialize()
    ViewerWindow(file_path)
    app.run()


def spawn_viewer(path: str, device_pref: str, title: str) -> None:
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--viewer", "--file", path, "--device", device_pref, "--title", title]
    else:
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--viewer",
            "--file",
            path,
            "--device",
            device_pref,
            "--title",
            title,
        ]
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
        for p in paths:
            add_file(p)

    def add_file(path: str):
        if not path:
            return
        listbox.insert(tk.END, path)
        spawn_viewer(path, device_pref, title)

    def on_drop(event):
        files = root.tk.splitlist(event.data)
        for f in files:
            add_file(f)

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
