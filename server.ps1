param(
    [ValidateSet('start', 'stop', 'restart', 'status', 'logs', 'install', 'uninstall')]
    [string]$Action = 'start',
    [switch]$ForApp,
    [string]$OwnerToken
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$runtime = Join-Path $root '.server'
$statePath = Join-Path $runtime 'process.json'
$configPath = Join-Path $root 'server.json'
$entry = Join-Path $root 'whisper_server.py'
$python = Join-Path $root '.venv\Scripts\python.exe'
$log = Join-Path $runtime 'server.log'

function Get-ServerProcess {
    if (!(Test-Path -LiteralPath $statePath)) { return $null }
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$state.pid)"
    # A recycled PID must never cause us to stop an unrelated process.
    if ($process -and $process.CommandLine -and
        $process.CommandLine.Contains($entry) -and
        $process.CreationDate.ToUniversalTime().Ticks.ToString() -eq $state.created) {
        return $process
    }
    return $null
}

function Stop-Server {
    if ($OwnerToken) {
        if (!(Test-Path -LiteralPath $statePath)) { return }
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        if ($state.owner -ne $OwnerToken) { return }
    }
    $process = Get-ServerProcess
    if ($process) {
        Stop-Process -Id $process.ProcessId -ErrorAction Stop
        Write-Host 'Server stopped. Active transcription, if any, was interrupted.'
    } else { Write-Host 'Server is not running under this launcher.' }
    if (Test-Path -LiteralPath $statePath) { Remove-Item -LiteralPath $statePath }
}

function New-ServerShortcut($path) {
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($path)
    $shortcut.TargetPath = Join-Path $env:WINDIR 'System32\cmd.exe'
    $shortcut.Arguments = '/c ""{0}""' -f (Join-Path $root 'server.cmd')
    $shortcut.WorkingDirectory = $root
    $shortcut.Save()
}

function Get-ServerConfig {
    $path = $configPath
    if (!(Test-Path -LiteralPath $path)) { $path = Join-Path $root 'server.example.json' }
    Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
}

function Start-Server {
    if (Get-ServerProcess) { Write-Host 'Server is already running.'; return }
    if (!(Test-Path -LiteralPath $python)) {
        throw 'Python environment not found. Run uv sync in the project folder first.'
    }
    New-Item -ItemType Directory -Path $runtime -Force | Out-Null
    if (!(Test-Path -LiteralPath $configPath)) {
        Copy-Item -LiteralPath (Join-Path $root 'server.example.json') -Destination $configPath
    }
    $config = Get-ServerConfig
    $port = [int]$config.port
    $limit = [int]$config.max_request_mb
    if ($port -lt 1 -or $port -gt 65535 -or $limit -lt 1) { throw 'Invalid port or max_request_mb in server.json.' }
    if ([string]$config.host -notmatch '^[a-zA-Z0-9.:-]+$') { throw 'Invalid host in server.json.' }
    $arguments = '"{0}" --host {1} --port {2} --max-request-mb {3} --log-file "{4}" --no-console-log' -f $entry, $config.host, $port, $limit, $log
    $child = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $runtime 'stdout.log') -RedirectStandardError (Join-Path $runtime 'stderr.log')
    try {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($child.Id)"
        if (!$process) { throw 'Server exited during startup.' }
        @{ pid = $child.Id; created = $process.CreationDate.ToUniversalTime().Ticks.ToString(); port = $port; host = [string]$config.host; owner = $OwnerToken } |
            ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
        $probeHost = [string]$config.host
        if ($probeHost -eq '0.0.0.0') { $probeHost = '127.0.0.1' }
        if ($probeHost.Contains(':')) { $probeHost = "[$probeHost]" }
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            $child.Refresh()
            if ($child.HasExited) { throw 'Server exited during startup.' }
            try {
                $health = Invoke-RestMethod "http://${probeHost}:$port/health" -TimeoutSec 1
                if ($health.status -eq 'ok') {
                    $child.Refresh()
                    if ($child.HasExited) { throw 'Server exited during startup.' }
                    Write-Host "Server ready on port $port. Logs: $log"
                    return
                }
            } catch { }
            Start-Sleep -Milliseconds 500
        }
        throw 'Server did not become ready in time.'
    } catch {
        $child.Refresh()
        if (!$child.HasExited) { Stop-Process -Id $child.Id }
        Remove-Item -LiteralPath $statePath -ErrorAction SilentlyContinue
        if (Test-Path (Join-Path $runtime 'stderr.log')) { Get-Content (Join-Path $runtime 'stderr.log') -Tail 15 }
        throw "$_ See $log and .server\stderr.log."
    }
}

try {
    # Wait briefly for concurrent status checks or launcher commands.
    New-Item -ItemType Directory -Path $runtime -Force | Out-Null
    $lockPath = Join-Path $runtime 'launcher.lock'
    for ($attempt = 0; ; $attempt++) {
        try {
            $guard = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None')
            break
        } catch {
            if ($_.Exception.InnerException -isnot [IO.IOException] -or $attempt -ge 25) { throw }
            Start-Sleep -Milliseconds 200
        }
    }
    switch ($Action) {
        'start' { Start-Server }
        'stop' { Stop-Server }
        'restart' { Stop-Server; Start-Server }
        'status' {
            $process = Get-ServerProcess
            $state = $null
            $health = $null
            if ($process) {
                $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
                $probeHost = $state.host
                if ($probeHost -eq '0.0.0.0') { $probeHost = '127.0.0.1' }
                if ($probeHost.Contains(':')) { $probeHost = "[$probeHost]" }
                try { $health = Invoke-RestMethod "http://${probeHost}:$($state.port)/health" -TimeoutSec 3 }
                catch { if (!$ForApp) { throw } }
            }
            if ($ForApp) {
                @{
                    running = [bool]$process; ready = $health.status -eq 'ok'
                    port = $state.port
                } | ConvertTo-Json -Compress
            } elseif ($process) {
                Write-Host "PID $($process.ProcessId), port $($state.port), health: $($health.status)"
            } else { Write-Host 'Stopped (or started outside this launcher).'; exit 1 }
        }
        'logs' {
            $guard.Dispose(); $guard = $null
            Get-Content -LiteralPath $log -Tail 50 -Wait
        }
        'install' {
            $destinations = @([Environment]::GetFolderPath('Programs'), [Environment]::GetFolderPath('Desktop'))
            foreach ($folder in $destinations) {
                New-ServerShortcut (Join-Path $folder 'Dialog TXT Server.lnk')
            }
            Write-Host 'Server shortcuts installed. Run server.cmd start to start now.'
        }
        'uninstall' {
            foreach ($name in @('Programs', 'Desktop', 'Startup')) {
                $path = Join-Path ([Environment]::GetFolderPath($name)) 'Dialog TXT Server.lnk'
                if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
            }
            Write-Host 'Server shortcuts removed.'
        }
    }
} catch { Write-Error $_; exit 1 }
finally { if ($guard) { $guard.Dispose() } }
