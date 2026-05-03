# 3D_visualizer

Open3D-based 3D point cloud visualizer with a single-window multi-pane layout and drag-and-drop launcher.

## Features
- Open multiple point clouds in **one Open3D window** using auto or manual pane layout (grid).
- Each pane can contain one or multiple point clouds; files are distributed across panes.
- Supports **Points / Voxel** rendering modes with adjustable sizes.
- Auto-detects the best display mode from the file suffix, including **GLB/GLTF mesh** files.
- Optional coordinate axis display with quaternion or 4x4 matrix transform.
- Optional **Preserve Coordinates** mode so the displayed origin stays aligned with the original camera frame.
- Shows **point count** for each loaded file.
- Drag-and-drop launcher (TkinterDnD2) and file dialog with last-used directory.
- Panel can be hidden/shown with **H**.

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
- **Preserve Coordinates**: keep the point cloud in its original frame so the axis origin remains the camera origin.
- **Hide panel**: click "Hide Panel (H)" or press **H**.

## Build EXE (Windows)
```
pip install pyinstaller
powershell -ExecutionPolicy Bypass -File build_exe.ps1
```

## License
MIT
