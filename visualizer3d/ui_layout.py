def panel_width_for_visibility(panel_visible: bool, font_size: float) -> int:
    return int(24 * font_size) if panel_visible else 0


def panel_toggle_label(panel_visible: bool) -> str:
    return ">" if panel_visible else "<"


def panel_restore_tab_frame(
    content_x: int,
    content_y: int,
    content_width: int,
    content_height: int,
    font_size: float,
    panel_visible: bool,
):
    if panel_visible:
        return None
    width = max(28, int(round(2.2 * font_size)))
    height = max(64, int(round(6.3 * font_size)))
    x = int(content_x + content_width - width)
    y = int(content_y + (content_height - height) / 2)
    return x, y, width, height


def panel_hide_tab_frame(
    content_x: int,
    content_y: int,
    content_width: int,
    content_height: int,
    panel_width: int,
    font_size: float,
    panel_visible: bool,
):
    if not panel_visible or panel_width <= 0:
        return None
    width = max(28, int(round(2.2 * font_size)))
    height = max(64, int(round(6.3 * font_size)))
    x = int(content_x + content_width - panel_width - width)
    y = int(content_y + (content_height - height) / 2)
    return x, y, width, height


def selected_listbox_paths(listbox) -> list[str]:
    return [listbox.get(index) for index in listbox.curselection()]


def missing_category_widgets(categories: list[str], existing_categories) -> list[str]:
    return [category for category in categories if category not in existing_categories]
