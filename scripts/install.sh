#!/bin/sh
# Install bioaccx and a launcher for its browser GUI, on macOS or Linux.
#
#   curl -LsSf https://github.com/biodiversica/bioaccx/releases/latest/download/install.sh | sh
#
# Installs uv if it is missing (uv brings its own Python, so none is needed
# beforehand), installs bioaccx as an isolated tool, and adds a launcher that
# opens the GUI in a work folder. Run it again to upgrade.
#
# Environment overrides:
#   BIOACCX_PACKAGE   what to install          (default: bioaccx[cpu,gui,umap])
#   BIOACCX_WORKDIR   folder the GUI starts in (default: ~/bioaccx)
set -eu

PACKAGE="${BIOACCX_PACKAGE:-bioaccx[cpu,gui,umap]}"
WORKDIR="${BIOACCX_WORKDIR:-$HOME/bioaccx}"
# TensorFlow publishes wheels for a narrow range of Pythons; pinning one keeps a
# newer system Python from being picked and failing to resolve.
PYTHON=3.12

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

case "$(uname -s)" in
  Darwin)
    [ "$(uname -m)" = arm64 ] || die "bioaccx needs an Apple Silicon Mac (TensorFlow has no Intel macOS build)."
    OS=macos ;;
  Linux) OS=linux ;;
  *) die "unsupported system $(uname -s); on Windows use install.ps1." ;;
esac

# ── uv ───────────────────────────────────────────────────────────────────────
if command -v uv >/dev/null 2>&1; then
  UV=$(command -v uv)
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV="$HOME/.local/bin/uv"
else
  say "Installing uv"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
  else
    die "neither curl nor wget is available to download uv."
  fi
  UV="$HOME/.local/bin/uv"
fi

# ── bioaccx ──────────────────────────────────────────────────────────────────
say "Installing $PACKAGE (about 2 GB, mostly TensorFlow; this takes a while)"
"$UV" tool install --python "$PYTHON" --upgrade "$PACKAGE"
"$UV" tool update-shell >/dev/null 2>&1 || true
BIOACCX="$("$UV" tool dir --bin)/bioaccx"
[ -x "$BIOACCX" ] || die "bioaccx was not installed where uv said it would be ($BIOACCX)."

mkdir -p "$WORKDIR"

# ── launcher ─────────────────────────────────────────────────────────────────
# Both open a terminal: it shows the address and is where Ctrl+C stops the GUI.
if [ "$OS" = macos ]; then
  LAUNCHER="$HOME/Desktop/bioaccx.command"
  mkdir -p "$HOME/Desktop"
  cat > "$LAUNCHER" <<EOF
#!/bin/sh
cd "$WORKDIR" && exec "$BIOACCX" gui
EOF
  chmod +x "$LAUNCHER"
  WHERE="double-click bioaccx on your Desktop"
else
  APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
  LAUNCHER="$APPS/bioaccx.desktop"
  mkdir -p "$APPS"
  cat > "$LAUNCHER" <<EOF
[Desktop Entry]
Type=Application
Name=bioaccx
Comment=Train bioacoustic classifiers from the browser
Exec="$BIOACCX" gui
Path=$WORKDIR
Terminal=true
Icon=audio-x-generic
Categories=Science;
EOF
  WHERE="open bioaccx from your applications menu"
fi

say "Done. $("$BIOACCX" --version 2>/dev/null || echo bioaccx) is installed."
cat <<EOF

  To start: $WHERE,
  or run 'bioaccx gui' in a new terminal.

  Your configs and trained models go in $WORKDIR
  To upgrade later, run this installer again.
EOF
