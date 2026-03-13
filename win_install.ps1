$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSCommandPath
$bat = Join-Path $root "dialog_txt.bat"
$gui = Join-Path $root "launch_gui.vbs"
$icon = Join-Path $root "docs\icon.ico"

if (-not (Test-Path $bat)) {
    Write-Error "dialog_txt.bat not found in $root"
    exit 1
}

if (-not (Test-Path $gui)) {
    Write-Error "launch_gui.vbs not found in $root"
    exit 1
}

$shell = New-Object -ComObject WScript.Shell

function New-Shortcut {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$TargetPath,
        [string]$WorkingDirectory,
        [string]$Description,
        [string]$IconLocation
    )
    if (Test-Path $Path) {
        Remove-Item $Path -Force
    }
    $shortcut = $shell.CreateShortcut($Path)
    $shortcut.TargetPath = $TargetPath
    $shortcut.WorkingDirectory = $WorkingDirectory
    if ($Description) {
        $shortcut.Description = $Description
    }
    if ($IconLocation -and (Test-Path $IconLocation)) {
        $shortcut.IconLocation = "$IconLocation,0"
    }
    $shortcut.Save()
}

$startMenu = [Environment]::GetFolderPath("Programs")
$startMenuLnk = Join-Path $startMenu "Dialog to TXT.lnk"
New-Shortcut -Path $startMenuLnk -TargetPath $gui -WorkingDirectory $root `
    -Description "Launch Dialog to TXT" -IconLocation $icon
Write-Host "Start Menu shortcut created: $startMenuLnk"

$desktop = [Environment]::GetFolderPath("Desktop")
$desktopLnk = Join-Path $desktop "Dialog to TXT.lnk"
New-Shortcut -Path $desktopLnk -TargetPath $gui -WorkingDirectory $root `
    -Description "Launch Dialog to TXT" -IconLocation $icon
Write-Host "Desktop shortcut created: $desktopLnk"

$taskbarDir = Join-Path $env:APPDATA "Microsoft\Internet Explorer\Quick Launch\User Pinned\TaskBar"
New-Item -ItemType Directory -Path $taskbarDir -Force | Out-Null
$taskbarLnk = Join-Path $taskbarDir "Dialog to TXT.lnk"
Copy-Item -Path $startMenuLnk -Destination $taskbarLnk -Force
Write-Host "Taskbar shortcut copied: $taskbarLnk"

$profilePath = $PROFILE
$profileDir = Split-Path -Parent $profilePath
if (-not (Test-Path $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
}
if (-not (Test-Path $profilePath)) {
    New-Item -ItemType File -Path $profilePath -Force | Out-Null
}

$begin = "# >>> dialog_txt >>>"
$end = "# <<< dialog_txt <<<"
$batForSingleQuoted = $bat -replace "'", "''"
$snippet = @"
$begin
function dialogtxt {
    & '$batForSingleQuoted' @Args
}
$end
"@.Trim()

$profileText = Get-Content -Path $profilePath -Raw
$pattern = "(?ms)$([regex]::Escape($begin)).*?$([regex]::Escape($end))"
if ($profileText -match [regex]::Escape($begin)) {
    $updated = [regex]::Replace($profileText, $pattern, $snippet)
} else {
    if ($profileText -and -not $profileText.EndsWith("`r`n")) {
        $profileText += "`r`n"
    }
    $updated = $profileText + "`r`n" + $snippet + "`r`n"
}
Set-Content -Path $profilePath -Value $updated -Encoding UTF8
Write-Host "PowerShell profile updated: $profilePath"

try {
    $ie4uinit = Join-Path $env:WINDIR "System32\ie4uinit.exe"
    if (Test-Path $ie4uinit) {
        Start-Process -FilePath $ie4uinit -ArgumentList "-show" -NoNewWindow -Wait
        Write-Host "Shell icon cache refreshed."
    }
} catch {
    Write-Warning "Could not refresh shell icon cache automatically."
}

Write-Host ""
Write-Host "Installation complete."
Write-Host "Use Start Menu or Desktop shortcut to launch the app."
Write-Host "PowerShell command added: dialogtxt"
