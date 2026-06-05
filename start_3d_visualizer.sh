#!/bin/sh
set -eu

ENV_NAME="${VISUALIZER_CONDA_ENV:-visualizer3d}"
PYTHON_VERSION="${VISUALIZER_PYTHON_VERSION:-3.11}"
SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
APP_PATH="$SCRIPT_DIR/pcd_viewer_app.py"
REQUIREMENTS_PATH="$SCRIPT_DIR/requirements.txt"

cd "$SCRIPT_DIR"

find_conda() {
    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi

    for candidate in \
        "$HOME/anaconda3/bin/conda" \
        "$HOME/miniconda3/bin/conda" \
        "/opt/anaconda3/bin/conda" \
        "/opt/miniconda3/bin/conda" \
        "/opt/homebrew/Caskroom/miniconda/base/bin/conda"
    do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

if ! CONDA_CMD="$(find_conda)"; then
    printf '[ERROR] Conda was not found.\n' >&2
    printf 'Install Miniconda/Anaconda or add conda to PATH, then try again.\n' >&2
    exit 1
fi

env_exists() {
    "$CONDA_CMD" env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"
}

deps_installed() {
    "$CONDA_CMD" run -n "$ENV_NAME" python -c \
        'from importlib.metadata import version; import numpy, open3d; raise SystemExit(version("tkinterdnd2") != "0.4.3")' \
        >/dev/null 2>&1
}

if ! env_exists; then
    printf 'Creating conda environment: %s (Python %s)\n' "$ENV_NAME" "$PYTHON_VERSION"
    "$CONDA_CMD" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

if ! deps_installed; then
    printf 'Installing Python dependencies in conda environment: %s\n' "$ENV_NAME"
    "$CONDA_CMD" run -n "$ENV_NAME" python -m pip install -r "$REQUIREMENTS_PATH"
fi

printf 'Starting 3D Visualizer with conda environment: %s\n' "$ENV_NAME"
exec "$CONDA_CMD" run --no-capture-output -n "$ENV_NAME" \
    python "$APP_PATH" --device auto "$@"
