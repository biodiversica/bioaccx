# Run one of the README's Windows commands as typed, against this checkout's
# install.ps1 served the way a GitHub release serves it.
#
# Release assets are application/octet-stream, and Windows PowerShell 5.1's
# irm hands such a script to iex line by line, which fails on the first block
# that spans several lines unless the command joins it back with Out-String.
# So a local server sends this checkout's script the same way, and the
# command's URL is pointed at it: what runs is the README's command, quoting
# and all, on the script being tested.
#
#   powershell -NoProfile -ExecutionPolicy ByPass -File .github\scripts\readme-command.ps1 [-Uninstall]
#
# Keep this file ASCII: Windows PowerShell 5.1 reads a BOM-less script as ANSI.

param([switch]$Uninstall)

$ErrorActionPreference = "Stop"
$Release = "https://github.com/biodiversica/bioaccx/releases/latest/download/install.ps1"
$Local = "http://localhost:8766/install.ps1"

$Server = Start-Job -ArgumentList (Resolve-Path scripts\install.ps1).Path {
    param($Path)
    $Bytes = [IO.File]::ReadAllBytes($Path)
    $Listener = New-Object System.Net.HttpListener
    $Listener.Prefixes.Add("http://localhost:8766/")
    $Listener.Start()
    while ($true) {
        $Context = $Listener.GetContext()
        $Context.Response.ContentType = "application/octet-stream"
        $Context.Response.OutputStream.Write($Bytes, 0, $Bytes.Length)
        $Context.Response.Close()
    }
}
try {
    $Ready = $false
    foreach ($i in 1..30) {
        try { $null = Invoke-RestMethod $Local; $Ready = $true; break } catch { Start-Sleep 1 }
    }
    if (-not $Ready) { Receive-Job $Server; throw "the local server did not start" }

    # What irm returns for the script, for the log: an array of lines is the
    # case Out-String is there for.
    $Body = Invoke-RestMethod $Local
    Write-Host ("irm {0} -> {1}, {2} item(s)" -f $Local, $Body.GetType().FullName, @($Body).Count)

    # The uninstall command, once the README has one, sets BIOACCX_UNINSTALL.
    $Which = if ($Uninstall) { "uninstall" } else { "install" }
    $Line = Select-String -Path README.md -SimpleMatch "irm $Release" |
        ForEach-Object { $_.Line } |
        Where-Object { ($_ -match "BIOACCX_UNINSTALL") -eq [bool]$Uninstall } |
        Select-Object -First 1
    if (-not $Line) { throw "README.md has no $Which command fetching $Release" }
    $Command = $Line.Trim().Replace($Release, $Local)
    Write-Host "> $Command"
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) { throw "the README's $Which command failed with exit code $LASTEXITCODE" }
    # The uninstall command runs the script in this session, as it does in the
    # reader's window, and must leave nothing of itself there.
    if (Test-Path Env:BIOACCX_UNINSTALL) { throw "BIOACCX_UNINSTALL is still set after the $Which command" }
    if (Get-Command Install-Bioaccx -ErrorAction SilentlyContinue) { throw "the installer's functions were left in the session" }
} finally {
    Stop-Job $Server
    Remove-Job $Server -Force
}
