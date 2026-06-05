import os
import subprocess
import sys
from pathlib import Path

from .ui_layout import selected_listbox_paths

PROJECT_ENTRYPOINT = Path(__file__).resolve().parent.parent / "pcd_viewer_app.py"


def spawn_viewer(paths, device_pref: str, title: str) -> None:
    if isinstance(paths, str):
        paths = [paths]
    cmd = [sys.executable]
    if not getattr(sys, "frozen", False):
        cmd.append(str(PROJECT_ENTRYPOINT))
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

    info = tk.Label(root, text="Drag and drop .ply/.pcd/.bin/.glb/.gltf files here, or click Open Files...")
    info.pack(padx=10, pady=10)

    listbox = tk.Listbox(root, width=80, height=10)
    listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def open_files():
        paths = filedialog.askopenfilenames(
            filetypes=[
                ("3D Files", "*.ply *.pcd *.bin *.glb *.gltf"),
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

    def reopen_selected(_event=None):
        paths = selected_listbox_paths(listbox)
        if paths:
            spawn_viewer(paths, device_pref, title)

    btn = tk.Button(root, text="Open Files...", command=open_files)
    btn.pack(padx=10, pady=6)

    listbox.drop_target_register(DND_FILES)
    listbox.dnd_bind("<<Drop>>", on_drop)
    listbox.bind("<Double-Button-1>", reopen_selected)
    root.mainloop()
