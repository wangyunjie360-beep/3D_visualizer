import argparse

from .launcher import launch_launcher
from .viewer import launch_viewer


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


def main() -> None:
    args = parse_args()
    if args.viewer or args.file:
        launch_viewer(args.file, args.title, args.device)
        return
    launch_launcher(args.device, args.title)
