#!/bin/sh
# Install bioaccx and a launcher for its browser GUI, on macOS or Linux.
#
#   curl -LsSf https://github.com/biodiversica/bioaccx/releases/latest/download/install.sh | sh
#
# Installs uv if it is missing (uv brings its own Python, so none is needed
# beforehand), installs bioaccx as an isolated tool, and adds a launcher that
# opens the GUI in a work folder. Run it again to upgrade.
#
# To uninstall, run it with BIOACCX_UNINSTALL set:
#
#   curl -LsSf https://github.com/biodiversica/bioaccx/releases/latest/download/install.sh | BIOACCX_UNINSTALL=1 sh
#
# Either way, it says what it is about to do and asks before doing any of it.
#
# Environment overrides:
#   BIOACCX_PACKAGE    what to install          (default: bioaccx[cpu,gui,umap])
#   BIOACCX_WORKDIR    folder the GUI starts in (default: ~/bioaccx)
#   BIOACCX_UNINSTALL  1 removes bioaccx and its launcher; all also removes the
#                      downloaded backbone models. The work folder, with its
#                      configs and trained models, is never removed, nor is uv.
#   BIOACCX_YES=1      go ahead without asking, e.g. from a script or CI
set -eu

PACKAGE="${BIOACCX_PACKAGE:-bioaccx[cpu,gui,umap]}"
WORKDIR="${BIOACCX_WORKDIR:-$HOME/bioaccx}"
UNINSTALL="${BIOACCX_UNINSTALL:-}"
# TensorFlow publishes wheels for a narrow range of Pythons; pinning one keeps a
# newer system Python from being picked and failing to resolve.
PYTHON=3.12
# The backbones' Hugging Face repositories, as the hub cache names their folders.
MODEL_FOLDERS="models--biodiversica--BirdNET-onnx-backbone models--biodiversica--Perch-onnx-backbone"

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# Ask before going on, unless BIOACCX_YES=1. $1 is what to say if the answer is
# no, $2 what is being done, for the message when there is no one to ask.
confirm() {
  if [ "${BIOACCX_YES:-}" = 1 ]; then return 0; fi
  # Asked of the terminal: under "curl ... | sh" standard input is the script.
  if ! (: </dev/tty) 2>/dev/null; then
    die "there is no terminal to ask in; set BIOACCX_YES=1 to $2 without asking."
  fi
  printf 'Continue? [y/N] '
  read -r ANSWER </dev/tty || ANSWER=
  case "$ANSWER" in
    [yY] | [yY][eE][sS]) ;;
    *) echo "$1"; exit 0 ;;
  esac
}

case "$(uname -s)" in
  Darwin) OS=macos ;;
  Linux) OS=linux ;;
  *) die "unsupported system $(uname -s); on Windows use install.ps1." ;;
esac

if command -v uv >/dev/null 2>&1; then
  UV=$(command -v uv)
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV="$HOME/.local/bin/uv"
else
  UV=
fi

if [ "$OS" = macos ]; then
  LAUNCHER="$HOME/Desktop/bioaccx.command"
else
  APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
  LAUNCHER="$APPS/bioaccx.desktop"
fi

# ── uninstall ────────────────────────────────────────────────────────────────
remove() {
  if [ -e "$1" ]; then
    rm -rf "$1"
    say "Removed $2 ($1)"
  fi
}

if [ -n "$UNINSTALL" ]; then
  case "$UNINSTALL" in
    1|all) ;;
    *) die "BIOACCX_UNINSTALL must be 1 or all, not '$UNINSTALL'." ;;
  esac
  # Only bioaccx's own models: the cache is shared with other programs.
  HUB="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
  ALL_TEXT=
  if [ "$UNINSTALL" = all ]; then
    ALL_TEXT="
  - remove the downloaded BirdNET and Perch backbones from $HUB"
  fi
  cat <<EOF
This uninstalls bioaccx. It will:

  - remove bioaccx, with uv (uv itself stays)
  - remove the launcher, $LAUNCHER$ALL_TEXT

The work folder, $WORKDIR, with your configs and trained models, is not removed.

EOF
  confirm "Nothing was removed." uninstall

  if [ -n "$UV" ] && "$UV" tool list 2>/dev/null | grep -q '^bioaccx '; then
    say "Uninstalling bioaccx"
    "$UV" tool uninstall bioaccx
  fi
  remove "$LAUNCHER" launcher
  if [ "$UNINSTALL" = all ]; then
    for name in $MODEL_FOLDERS; do
      remove "$HUB/$name" model
      remove "$HUB/.locks/$name" "model lock"
    done
  fi
  say "Done. bioaccx is uninstalled."
  cat <<EOF

  The work folder was left where it is: $WORKDIR
  Delete it yourself if you no longer want its configs and trained models.
EOF
  exit 0
fi

# ── install ──────────────────────────────────────────────────────────────────
if [ "$OS" = macos ] && [ "$(uname -m)" != arm64 ]; then
  die "bioaccx needs an Apple Silicon Mac (TensorFlow has no Intel macOS build)."
fi

if [ "$OS" = macos ]; then
  LAUNCHER_TEXT="bioaccx.command on your Desktop"
else
  LAUNCHER_TEXT="a bioaccx entry in your applications menu"
fi
if [ -n "$UV" ]; then
  UV_TEXT="use uv, already installed at $UV"
else
  UV_TEXT="install uv, the tool that installs bioaccx, into ~/.local/bin"
fi
cat <<EOF
This installs bioaccx, a tool for training and sharing bioacoustic
classifiers, with its GUI in the browser. It will:

  - $UV_TEXT
  - install $PACKAGE with uv, isolated, along with a Python $PYTHON
    of its own; your other Pythons are not touched (if bioaccx is already
    installed, it is upgraded)
  - add uv's tools folder to your PATH, in your shell's startup file
  - create the folder $WORKDIR, where the GUI starts
  - add $LAUNCHER_TEXT

The download is about 2 GB, mostly TensorFlow, and takes a while.

EOF
confirm "Nothing was installed." install

# ── uv ───────────────────────────────────────────────────────────────────────
if [ -z "$UV" ]; then
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
say "Installing $PACKAGE"
"$UV" tool install --python "$PYTHON" --upgrade "$PACKAGE"
"$UV" tool update-shell >/dev/null 2>&1 || true
BIOACCX="$("$UV" tool dir --bin)/bioaccx"
[ -x "$BIOACCX" ] || die "bioaccx was not installed where uv said it would be ($BIOACCX)."

mkdir -p "$WORKDIR"

# ── launcher ─────────────────────────────────────────────────────────────────
# Both open a terminal: it shows the address and is where Ctrl+C stops the GUI.
if [ "$OS" = macos ]; then
  mkdir -p "$HOME/Desktop"
  cat > "$LAUNCHER" <<EOF
#!/bin/sh
cd "$WORKDIR" && exec "$BIOACCX" gui
EOF
  chmod +x "$LAUNCHER"
  WHERE="double-click bioaccx on your Desktop"
else
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
  To uninstall, run it with BIOACCX_UNINSTALL=1.
EOF
