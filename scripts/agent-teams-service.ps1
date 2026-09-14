<#
.SYNOPSIS
  Manual start / stop / restart / status for Agent Teams (Hermes swarm) on :8000.

.DESCRIPTION
  Windows-hosted process (not systemd). `agent-teams up --detach` needs POSIX
  fork, so on Windows we Start-Process the CLI in the background instead.
  Safe to call from PowerShell or from AICC via:
    powershell.exe -File ...\agent-teams-service.ps1 -Action restart

.EXAMPLE
  .\scripts\agent-teams-service.ps1 -Action status
  .\scripts\agent-teams-service.ps1 -Action start
  .\scripts\agent-teams-service.ps1 -Action stop
  .\scripts\agent-teams-service.ps1 -Action restart
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'stop', 'restart', 'status')]
    [string]$Action = 'status'
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$VenvCli = Join-Path $RepoRoot '.venv\Scripts\agent-teams.exe'
$LogDir = Join-Path $RepoRoot 'data\logs'
$PidDir = Join-Path $RepoRoot 'data'
$PidFile = Join-Path $PidDir 'teams.pid'
$Port = 8000

function Ensure-Dirs {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    New-Item -ItemType Directory -Force -Path $PidDir | Out-Null
}

function Get-PortPids([int]$ListenPort) {
    @(
        Get-NetTCPConnection -LocalPort $ListenPort -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    ) | Where-Object { $_ -and $_ -gt 0 }
}

function Test-PortUp([int]$ListenPort) {
    return @(Get-PortPids $ListenPort).Count -gt 0
}

function Resolve-Launcher {
    if (Test-Path -LiteralPath $VenvCli) {
        return [pscustomobject]@{
            FilePath = $VenvCli
            Args     = @('up')
        }
    }
    if (Test-Path -LiteralPath $VenvPython) {
        return [pscustomobject]@{
            FilePath = $VenvPython
            Args     = @('-m', 'teams_server.cli', 'up')
        }
    }
    throw "No Agent Teams launcher found. Expected $VenvCli or $VenvPython — run install in $RepoRoot first."
}

function New-LogPaths {
    Ensure-Dirs
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    return [pscustomobject]@{
        Out = Join-Path $LogDir "teams.$stamp.out.log"
        Err = Join-Path $LogDir "teams.$stamp.err.log"
    }
}

function Prune-Logs([int]$Keep = 12) {
    Get-ChildItem -Path $LogDir -Filter 'teams.*.log' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -Skip $Keep |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue }
}

function Stop-Port([int]$ListenPort) {
    $pids = @(Get-PortPids $ListenPort)
    if ($pids.Count -eq 0) {
        Write-Host "Agent Teams :$ListenPort already stopped" -ForegroundColor DarkGray
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        return
    }
    foreach ($procId in $pids) {
        try {
            $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
            $name = if ($p) { $p.ProcessName } else { '?' }
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Write-Host "Stopped Agent Teams :$ListenPort (pid $procId, $name)" -ForegroundColor Yellow
        } catch {
            Write-Warning "Could not stop pid $procId on :$ListenPort - $($_.Exception.Message)"
        }
    }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 400
}

function Start-Teams {
    if (Test-PortUp $Port) {
        Write-Host "Agent Teams already listening on :$Port" -ForegroundColor DarkGray
        return
    }

    $launcher = Resolve-Launcher
    Ensure-Dirs
    Prune-Logs
    $logs = New-LogPaths

    $common = @{
        FilePath         = $launcher.FilePath
        ArgumentList     = $launcher.Args
        WorkingDirectory = $RepoRoot
        WindowStyle      = 'Hidden'
        PassThru         = $true
    }

    try {
        $proc = Start-Process @common `
            -RedirectStandardOutput $logs.Out `
            -RedirectStandardError $logs.Err
    } catch {
        Write-Warning "Log redirect failed ($($_.Exception.Message)); starting without file logs."
        $proc = Start-Process @common
    }

    Set-Content -Path $PidFile -Value $proc.Id -Encoding ascii
    Write-Host "Started Agent Teams pid $($proc.Id) -> http://127.0.0.1:$Port/" -ForegroundColor Green
    if ($logs.Out) {
        Write-Host "  logs: $($logs.Out)" -ForegroundColor DarkGray
    }
}

function Show-Status {
    $up = Test-PortUp $Port
    Write-Host "Agent Teams (Hermes swarm) status" -ForegroundColor Cyan
    Write-Host ("  API/UI :{0}  {1}" -f $Port, $(if ($up) { 'UP  http://127.0.0.1:{0}/' -f $Port } else { 'DOWN' }))
    if ($up) {
        try {
            $h = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 -UseBasicParsing
            Write-Host "         /health HTTP $($h.StatusCode)" -ForegroundColor DarkGray
        } catch {
            Write-Host "         listening but /health failed" -ForegroundColor Yellow
        }
    }
    if (Test-Path -LiteralPath $PidFile) {
        $pidText = (Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
        Write-Host "         pidfile=$pidText" -ForegroundColor DarkGray
    }
    return $up
}

function Wait-Port([int]$ListenPort, [int]$TimeoutSec = 25) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-PortUp $ListenPort) { return $true }
        Start-Sleep -Milliseconds 400
    }
    return (Test-PortUp $ListenPort)
}

function Wait-PortClear([int]$ListenPort, [int]$TimeoutSec = 10) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (-not (Test-PortUp $ListenPort)) { return $true }
        Start-Sleep -Milliseconds 300
    }
    return (-not (Test-PortUp $ListenPort))
}

$ok = $true
switch ($Action) {
    'status' {
        $ok = [bool](Show-Status)
    }
    'start' {
        Start-Teams
        [void](Wait-Port $Port 30)
        $ok = [bool](Show-Status)
    }
    'stop' {
        Stop-Port $Port
        $ok = -not (Test-PortUp $Port)
        [void](Show-Status)
    }
    'restart' {
        Stop-Port $Port
        [void](Wait-PortClear $Port 10)
        Start-Teams
        [void](Wait-Port $Port 30)
        $ok = [bool](Show-Status)
    }
}
if (-not $ok) { exit 1 }
exit 0
