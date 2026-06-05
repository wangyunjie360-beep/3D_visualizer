from visualizer3d.cli import main, parse_args
from visualizer3d.device import resolve_device
from visualizer3d.assets import load_asset, load_asset_parts, load_bin_as_pcd, preferred_render_mode_for_path
from visualizer3d.geometry import (
    clamp_camera_line_width,
    clone_geometry,
    combined_bbox,
    configure_axis_marker_material,
    create_axis_marker,
    create_camera_axis_marker,
    geometry_point_count,
    layout_clouds,
    parse_matrix,
    split_clouds,
)
from visualizer3d.launcher import launch_launcher, spawn_viewer
from visualizer3d.scene import (
    configure_scene_widget_interaction,
    maybe_setup_camera,
    set_pane_background,
    should_reset_camera_for_rebuild,
)
from visualizer3d.screenshots import (
    default_screenshot_path,
    image_to_uint8_array,
    render_scene_image,
    screenshot_render_size,
    stitch_screenshot_arrays,
)
from visualizer3d.semantics import (
    find_semantic_color_map,
    parse_ply_semantic_color_map,
    split_mesh_by_vertex_color,
)
from visualizer3d.ui_layout import (
    missing_category_widgets,
    panel_hide_tab_frame,
    panel_restore_tab_frame,
    panel_toggle_label,
    panel_width_for_visibility,
    selected_listbox_paths,
)
from visualizer3d.viewer import launch_viewer
from visualizer3d.models import AssetPart


if __name__ == "__main__":
    main()
