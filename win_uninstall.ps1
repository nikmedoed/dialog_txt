$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSCommandPath
$startMenu = [Environment]::GetFolderPath("Programs")
$desktop = [Environment]::GetFolderPath("Desktop")
$taskbarDir = Join-Path $env:APPDATA "Microsoft\Internet Explorer\Quick Launch\User Pinned\TaskBar"

$links = @(
    (Join-Path $startMenu "Dialog to TXT.lnk"),
    (Join-Path $desktop "Dialog to TXT.lnk"),
    (Join-Path $taskbarDir "Dialog to TXT.lnk")
)

foreach ($path in $links) {
    if (Test-Path $path) {
        Remove-Item $path -Force
        Write-Host "Removed shortcut: $path"
    }
}

$profilePath = $PROFILE
if (Test-Path $profilePath) {
    $text = Get-Content -Path $profilePath -Raw
    $begin = "# >>> dialog_txt >>>"
    $end = "# <<< dialog_txt <<<"
    $pattern = "(?ms)`r?`n?$([regex]::Escape($begin)).*?$([regex]::Escape($end))`r?`n?"
    $updated = [regex]::Replace($text, $pattern, "`r`n")
    Set-Content -Path $profilePath -Value $updated -Encoding UTF8
    Write-Host "PowerShell profile cleaned: $profilePath"
}

Write-Host ""
Write-Host "Uninstall complete."
Write-Host "Project files in $root were not removed."
