# Install bioaccx and a shortcut for its browser GUI, on Windows.
#
#   powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/bioaccx/releases/latest/download/install.ps1 | iex"
#
# Installs uv if it is missing (uv brings its own Python, so none is needed
# beforehand), installs bioaccx as an isolated tool, and adds Desktop and Start
# menu shortcuts that open the GUI in a work folder. Run it again to upgrade.
#
# Environment overrides:
#   BIOACCX_PACKAGE   what to install          (default: bioaccx[cpu,gui,umap])
#   BIOACCX_WORKDIR   folder the GUI starts in (default: %USERPROFILE%\bioaccx)
#
# Keep this file ASCII: Windows PowerShell 5.1 reads a BOM-less script as ANSI.

$ErrorActionPreference = "Stop"

$Package = if ($env:BIOACCX_PACKAGE) { $env:BIOACCX_PACKAGE } else { "bioaccx[cpu,gui,umap]" }
$WorkDir = if ($env:BIOACCX_WORKDIR) { $env:BIOACCX_WORKDIR } else { Join-Path $HOME "bioaccx" }
# TensorFlow publishes wheels for a narrow range of Pythons; pinning one keeps a
# newer system Python from being picked and failing to resolve.
$Python = "3.12"

function Say($Message) { Write-Host "==> $Message" }

# --- uv ----------------------------------------------------------------------
$Found = Get-Command uv -ErrorAction SilentlyContinue
$Uv = if ($Found) { $Found.Source } else { Join-Path $HOME ".local\bin\uv.exe" }
if (-not (Test-Path $Uv)) {
    Say "Installing uv"
    $env:UV_INSTALL_DIR = Join-Path $HOME ".local\bin"
    # A separate process, so an exit in uv's installer cannot end this one.
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Uv)) { throw "uv could not be installed" }
}

# --- bioaccx -----------------------------------------------------------------
Say "Installing $Package (about 2 GB, mostly TensorFlow; this takes a while)"
# Native commands do not stop on failure under $ErrorActionPreference, and their
# stderr must not be redirected: 5.1 would turn uv's progress lines into errors.
& $Uv tool install --python $Python --upgrade $Package
if ($LASTEXITCODE -ne 0) { throw "uv tool install failed with exit code $LASTEXITCODE" }
& $Uv tool update-shell
$Bin = (& $Uv tool dir --bin | Out-String).Trim()
$Exe = Join-Path $Bin "bioaccx.exe"
if (-not (Test-Path $Exe)) { throw "bioaccx was not installed where uv said it would be ($Exe)" }

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

# --- shortcuts ---------------------------------------------------------------
# They open a console window: it shows the address and is where Ctrl+C stops
# the GUI.
$Shell = New-Object -ComObject WScript.Shell
$Places = @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))
foreach ($Folder in $Places) {
    $Link = $Shell.CreateShortcut((Join-Path $Folder "bioaccx.lnk"))
    $Link.TargetPath = $Exe
    $Link.Arguments = "gui"
    $Link.WorkingDirectory = $WorkDir
    $Link.Description = "Train bioacoustic classifiers from the browser"
    $Link.Save()
}

$Version = (& $Exe --version | Out-String).Trim()
Say "Done. $Version is installed."
Write-Host ""
Write-Host "  To start: double-click bioaccx on your Desktop or in the Start menu,"
Write-Host "  or run 'bioaccx gui' in a new terminal."
Write-Host ""
Write-Host "  Your configs and trained models go in $WorkDir"
Write-Host "  To upgrade later, run this installer again."
