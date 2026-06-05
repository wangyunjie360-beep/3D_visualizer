def should_reset_camera_for_rebuild(reason: str = "scene") -> bool:
    return reason not in {"point_size", "voxel_size", "axis"}


def maybe_setup_camera(widget, bbox, reset_camera: bool = True) -> bool:
    if not reset_camera or bbox is None or bbox.is_empty():
        return False
    widget.setup_camera(60.0, bbox, bbox.get_center())
    return True


def set_pane_background(pane, color) -> None:
    pane["background"] = list(color)
    pane["scene"].set_background(pane["background"])
    if hasattr(pane["scene"], "show_skybox"):
        pane["scene"].show_skybox(False)
    if hasattr(pane["widget"], "background_color"):
        pane["widget"].background_color = pane["gui_color"](
            pane["background"][0],
            pane["background"][1],
            pane["background"][2],
            pane["background"][3],
        )
    if hasattr(pane["widget"], "force_redraw"):
        pane["widget"].force_redraw()


def configure_scene_widget_interaction(widget) -> None:
    if hasattr(widget, "enable_scene_caching"):
        widget.enable_scene_caching(False)
