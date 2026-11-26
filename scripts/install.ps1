# Install bioaccx and a shortcut for its browser GUI, on Windows.
#
#   powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/bioaccx/releases/latest/download/install.ps1 | Out-String | iex"
#
# Installs uv if it is missing (uv brings its own Python, so none is needed
# beforehand), installs bioaccx as an isolated tool, and adds Desktop and Start
# menu shortcuts that open the GUI in a work folder. Run it again to upgrade.
#
# To uninstall, set BIOACCX_UNINSTALL and run it in the PowerShell window itself:
#
#   $env:BIOACCX_UNINSTALL=1; irm https://github.com/biodiversica/bioaccx/releases/latest/download/install.ps1 | Out-String | iex
#
# Not through powershell -c '...', with the variable in single quotes: those
# quotes can be lost on the way in (it happened on Windows 10), and the window's
# PowerShell then expands the variable to nothing and installs instead.
#
# Either way, it says what it is about to do and asks before doing any of it.
#
# Environment overrides:
#   BIOACCX_PACKAGE    what to install          (default: bioaccx[cpu,gui,umap])
#   BIOACCX_WORKDIR    folder the GUI starts in (default: %USERPROFILE%\bioaccx)
#   BIOACCX_UNINSTALL  1 removes bioaccx and its shortcuts; all also removes the
#                      downloaded backbone models. The work folder, with its
#                      configs and trained models, is never removed, nor is uv.
#   BIOACCX_YES=1      go ahead without asking, e.g. from a script or CI
#
# Keep this file ASCII: Windows PowerShell 5.1 reads a BOM-less script as ANSI.

# All of it in a script block: the uninstall command runs this in the reader's
# own window, and its variables ($ErrorActionPreference above all) and
# functions must not stay behind there.
& {

$ErrorActionPreference = "Stop"

$Package = if ($env:BIOACCX_PACKAGE) { $env:BIOACCX_PACKAGE } else { "bioaccx[cpu,gui,umap]" }
$WorkDir = if ($env:BIOACCX_WORKDIR) { $env:BIOACCX_WORKDIR } else { Join-Path $HOME "bioaccx" }
$Uninstall = $env:BIOACCX_UNINSTALL
# Cleared once read, so that the window it was set in does not uninstall the
# next time the installer is run from it.
if (Test-Path Env:BIOACCX_UNINSTALL) { Remove-Item Env:BIOACCX_UNINSTALL }
# TensorFlow publishes wheels for a narrow range of Pythons; pinning one keeps a
# newer system Python from being picked and failing to resolve.
$Python = "3.12"
# Where the shortcuts go, and so where uninstalling looks for them.
$Places = @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))
# Where the shortcuts' icon is kept, out of the tool's environment.
$LocalData = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $HOME "AppData\Local" }
$IconDir = Join-Path $LocalData "bioaccx"
$Icon = Join-Path $IconDir "bioaccx.ico"
# The backbones' Hugging Face repositories, as the hub cache names their folders.
$ModelFolders = @("models--biodiversica--BirdNET-onnx-backbone", "models--biodiversica--Perch-onnx-backbone")

function Say($Message) { Write-Host "==> $Message" }

function Find-Uv {
    $Found = Get-Command uv -ErrorAction SilentlyContinue
    if ($Found) { $Found.Source } else { Join-Path $HOME ".local\bin\uv.exe" }
}

# Whether to go ahead: yes under BIOACCX_YES=1, otherwise whatever is answered.
function Confirm-Plan($Doing) {
    if ($env:BIOACCX_YES -eq "1") { return $true }
    if ([Console]::IsInputRedirected) {
        throw "there is no console to ask in; set BIOACCX_YES=1 to $Doing without asking."
    }
    $Answer = Read-Host "Continue? [y/N]"
    return ($Answer -match '^\s*(y|yes)\s*$')
}

function Remove-IfPresent($Path, $What) {
    if (Test-Path $Path) {
        Remove-Item -Recurse -Force $Path
        Say "Removed $What ($Path)"
    }
}

function Uninstall-Bioaccx {
    if ($Uninstall -notin @("1", "all")) {
        throw "BIOACCX_UNINSTALL must be 1 or all, not '$Uninstall'"
    }
    # Only bioaccx's own models: the cache is shared with other programs.
    $Hub = Join-Path $HOME ".cache\huggingface\hub"
    if ($env:HF_HOME) { $Hub = Join-Path $env:HF_HOME "hub" }
    if ($env:HF_HUB_CACHE) { $Hub = $env:HF_HUB_CACHE }

    $AllText = ""
    if ($Uninstall -eq "all") {
        $AllText = @"

  - remove the downloaded BirdNET and Perch backbones from $Hub
"@
    }
    Write-Host @"
This uninstalls bioaccx. It will:

  - remove bioaccx, with uv (uv itself stays)
  - remove the bioaccx shortcuts from your Desktop and Start menu$AllText

The work folder, $WorkDir, with your configs and trained models, is not removed.

"@
    if (-not (Confirm-Plan "uninstall")) {
        Write-Host "Nothing was removed."
        return
    }
    # Windows will not delete a program that is running.
    if (Get-Process -Name bioaccx -ErrorAction SilentlyContinue) {
        throw "bioaccx is running. Close its window (or press Ctrl+C in it), then run this again."
    }

    $Uv = Find-Uv
    if ((Test-Path $Uv) -and ((& $Uv tool list | Out-String) -match "(?m)^bioaccx ")) {
        Say "Uninstalling bioaccx"
        & $Uv tool uninstall bioaccx
        if ($LASTEXITCODE -ne 0) { throw "uv tool uninstall failed with exit code $LASTEXITCODE" }
    }
    foreach ($Folder in $Places) {
        Remove-IfPresent (Join-Path $Folder "bioaccx.lnk") "shortcut"
    }
    Remove-IfPresent $Icon "shortcut icon"
    # The folder held only the icon; anything else put there is left alone.
    if ((Test-Path $IconDir) -and -not (Get-ChildItem $IconDir -Force)) {
        Remove-Item -Force $IconDir
    }

    if ($Uninstall -eq "all") {
        foreach ($Name in $ModelFolders) {
            Remove-IfPresent (Join-Path $Hub $Name) "model"
            Remove-IfPresent (Join-Path (Join-Path $Hub ".locks") $Name) "model lock"
        }
    }

    Say "Done. bioaccx is uninstalled."
    Write-Host ""
    Write-Host "  The work folder was left where it is: $WorkDir"
    Write-Host "  Delete it yourself if you no longer want its configs and trained models."
}

function Install-Bioaccx {
    $Uv = Find-Uv
    $HaveUv = Test-Path $Uv

    # --- confirmation --------------------------------------------------------
    $UvText = if ($HaveUv) { "use uv, already installed at $Uv" } else { "install uv, the tool that installs bioaccx, into $(Join-Path $HOME '.local\bin')" }
    Write-Host @"
This installs bioaccx, a tool for training and sharing bioacoustic
classifiers, with its GUI in the browser. It will:

  - $UvText
  - install $Package with uv, isolated, along with a Python $Python
    of its own; your other Pythons are not touched (if bioaccx is already
    installed, it is upgraded)
  - add uv's tools folder to your user PATH
  - create the folder $WorkDir, where the GUI starts
  - add bioaccx shortcuts, with its icon, to your Desktop and Start menu

The download is about 2 GB, mostly TensorFlow, and takes a while.

"@
    if (-not (Confirm-Plan "install")) {
        Write-Host "Nothing was installed."
        return
    }

    # --- uv ------------------------------------------------------------------
    if (-not $HaveUv) {
        Say "Installing uv"
        $env:UV_INSTALL_DIR = Join-Path $HOME ".local\bin"
        # A separate process, so an exit in uv's installer cannot end this one.
        powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Uv)) { throw "uv could not be installed" }
    }

    # --- bioaccx -------------------------------------------------------------
    Say "Installing $Package"
    # Native commands do not stop on failure under $ErrorActionPreference, and
    # their stderr must not be redirected: 5.1 would turn uv's progress lines
    # into errors.
    & $Uv tool install --python $Python --upgrade $Package
    if ($LASTEXITCODE -ne 0) { throw "uv tool install failed with exit code $LASTEXITCODE" }
    & $Uv tool update-shell
    $Bin = (& $Uv tool dir --bin | Out-String).Trim()
    $Exe = Join-Path $Bin "bioaccx.exe"
    if (-not (Test-Path $Exe)) { throw "bioaccx was not installed where uv said it would be ($Exe)" }

    New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

    # --- icon ----------------------------------------------------------------
    # Copied out of the tool's environment, which uv may rebuild on an upgrade,
    # to a place that stays put for as long as the shortcuts point at it. A
    # package from before the logo has none, and the shortcuts then keep the
    # default icon.
    $ToolDir = (& $Uv tool dir | Out-String).Trim()
    $PackagedIcon = Join-Path $ToolDir "bioaccx\Lib\site-packages\bioaccx\assets\bioaccx.ico"
    $HaveIcon = Test-Path $PackagedIcon
    if ($HaveIcon) {
        New-Item -ItemType Directory -Force -Path $IconDir | Out-Null
        Copy-Item -Force $PackagedIcon $Icon
    }

    # --- shortcuts -----------------------------------------------------------
    # They open a console window: it shows the address and is where Ctrl+C
    # stops the GUI.
    $Shell = New-Object -ComObject WScript.Shell
    foreach ($Folder in $Places) {
        $Link = $Shell.CreateShortcut((Join-Path $Folder "bioaccx.lnk"))
        $Link.TargetPath = $Exe
        $Link.Arguments = "gui"
        $Link.WorkingDirectory = $WorkDir
        $Link.Description = "Train bioacoustic classifiers from the browser"
        if ($HaveIcon) { $Link.IconLocation = "$Icon,0" }
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
    Write-Host "  To uninstall, run it with BIOACCX_UNINSTALL=1."
}

# Functions rather than an early return: under iex, a return at the top level
# does not reliably end the script.
if ($Uninstall) { Uninstall-Bioaccx } else { Install-Bioaccx }

}
