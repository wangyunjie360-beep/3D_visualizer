# 3D_visualizer

Open3D-based 3D point cloud visualizer with a single-window multi-pane layout and drag-and-drop launcher.

## Features
- Open multiple point clouds in **one Open3D window** using auto or manual pane layout (grid).
- Each pane can contain one or multiple point clouds; files are distributed across panes.
- Supports **Points / Voxel** rendering modes with adjustable sizes.
- Auto-detects the best display mode from the file suffix, including **GLB/GLTF mesh** files.
- Category checkboxes let you show/hide GLB/GLTF sub-mesh categories; all categories are visible by default.
- Screenshot button exports the current rendered view as a white-background PNG.
- Optional coordinate axis display with quaternion or 4x4 matrix transform.
- Optional **Preserve Coordinates** mode so the displayed origin stays aligned with the original camera frame.
- Shows **point count** for each loaded file.
- Drag-and-drop launcher (TkinterDnD2) and file dialog with last-used directory.
- Panel can be hidden/shown with side arrow tabs or **H**.

## Requirements
- Python 3.8+ (recommended)
- Open3D
- NumPy
- TkinterDnD2 (for drag-and-drop launcher)

Install:
```
pip install -r requirements.txt
```

## Usage
One-click launcher on macOS/Linux:
```
sh start_3d_visualizer.sh
```

One-click launcher on Windows:
```
start_3d_visualizer.bat
```

Start launcher (drag & drop):
```
python pcd_viewer_app.py
```

Open files directly:
```
python pcd_viewer_app.py --viewer --file A.ply --file B.glb --file C.bin
```

## UI Notes
- **Auto layout**: set number of panes and the grid is chosen automatically.
- **Manual layout**: disable Auto layout and set rows/cols.
- **Categories**: uncheck categories to hide them from the current render.
- **Screenshot PNG**: saves the current view to `render_YYYYMMDD_HHMMSS.png` in the current working directory.
- **Screenshot Scale**: renders screenshots at 1x-4x window resolution; default is 2x for sharper images.
- **Preserve Coordinates**: keep the point cloud in its original frame so the axis origin remains the camera origin.
- **Axis Style**: switch between the default coordinate axis and a thick, multi-color wireframe camera frustum pointing along +X.
- **Camera Line Width**: adjust the wireframe camera line thickness from 1 to 12.
- **Hide panel**: click the `>` tab at the panel edge or press **H**. Click the `<` tab on the right edge to restore it.

## Project Structure
- `pcd_viewer_app.py`: thin compatibility entry point used by the launchers and PyInstaller.
- `visualizer3d/cli.py`: command-line parsing and app mode selection.
- `visualizer3d/viewer.py`: Open3D viewer window and panel callbacks.
- `visualizer3d/launcher.py`: TkinterDnD drag-and-drop launcher.
- `visualizer3d/assets.py` and `visualizer3d/semantics.py`: 3D file loading and semantic mesh splitting.
- `visualizer3d/geometry.py`, `visualizer3d/scene.py`, `visualizer3d/screenshots.py`, `visualizer3d/ui_layout.py`: reusable viewer helpers.
- `start_3d_visualizer.bat`: Windows launcher. The old misspelled `visiual_.bat` redirect was removed.

## Build EXE (Windows)
```
pip install pyinstaller
powershell -ExecutionPolicy Bypass -File build_exe.ps1
```

## License
MIT
