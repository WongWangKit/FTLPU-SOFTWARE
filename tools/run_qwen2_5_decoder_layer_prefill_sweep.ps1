param(
    [int]$MaxParallel = 3,
    [switch]$Resume,
    [switch]$NoPipeline
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $repoRoot
$resultRoot = Join-Path $repoRoot "results"
$sourceStream = Join-Path $workspaceRoot "projection_rope_run\softmax_closed3d\decoder_layer.stream.mlir"
$loweringDir = Join-Path $workspaceRoot "projection_rope_run\ttft_sweep_direct"
$sourceSchedule = Join-Path $loweringDir "decoder_layer.schedule.mlir"
$sourceCommand = Join-Path $loweringDir "decoder_layer.command.mlir"
$baseConfig = Join-Path $workspaceRoot "FTLPU-CMODEL\config\ftlpu-lpu32.json"
$optimizer = Join-Path $repoRoot "build-ftlpu-vs2026-direct\compiler\ftlpu_opt.exe"
$compiler = Join-Path $repoRoot "build-ftlpu-vs2026-direct\compiler\ftlpu-compile.exe"
$runner = Join-Path $PSScriptRoot "run_qwen2_5_decoder_layer_prefill.ps1"
$summaryPath = Join-Path $resultRoot "qwen2_5_decoder_layer_prefill_sweep.csv"

foreach ($path in @($sourceStream, $baseConfig, $optimizer, $compiler, $runner)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required sweep input does not exist: $path"
    }
}

$combinations = foreach ($clock in @(500, 750, 1000)) {
    foreach ($bandwidth in @(25600, 51200, 68000, 85600)) {
        [pscustomobject]@{
            ClockMHz = $clock
            BandwidthMBps = $bandwidth
            Name = "qwen2_5_decoder_layer_prefill_f${clock}_bw${bandwidth}"
        }
    }
}

# Create all target overlays first. The hardware command file must be regenerated
# from Stream IR so current direct lowering participates in the sweep; re-encoding
# an old Command IR would preserve stale cross-page ICU domains.
$base = Get-Content -LiteralPath $baseConfig -Raw | ConvertFrom-Json
foreach ($combination in $combinations) {
    $directory = Join-Path $resultRoot $combination.Name
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $configPath = Join-Path $directory "target.json"
    if (-not $Resume -or -not (Test-Path -LiteralPath $configPath)) {
        $config = $base | ConvertTo-Json -Depth 100 | ConvertFrom-Json
        $config.external_memory.lpu_clock_mhz = $combination.ClockMHz
        $config.external_memory.ddr_peak_bandwidth_mbytes_per_second = $combination.BandwidthMBps
        $configText = $config | ConvertTo-Json -Depth 100
        [System.IO.File]::WriteAllText(
            $configPath,
            $configText,
            [System.Text.UTF8Encoding]::new($false))

    }
}

New-Item -ItemType Directory -Path $loweringDir -Force | Out-Null
$loweringCombination = $combinations |
    Where-Object { $_.ClockMHz -eq 500 -and $_.BandwidthMBps -eq 51200 } |
    Select-Object -First 1
$loweringTarget = Join-Path `
    (Join-Path $resultRoot $loweringCombination.Name) "target.json"
$loweringLog = Join-Path $loweringDir "lowering.log"
$loweringOutput = & $optimizer `
    --input $sourceStream `
    --output $sourceSchedule `
    --pipeline ftlpu-stream-to-schedule `
    --mxm-execution vector `
    --ffn-schedule fused `
    --projection-rope-overlap off `
    --target-config $loweringTarget `
    --weight-bank 1 `
    --kv-cache-capacity 256 2>&1
$loweringExitCode = $LASTEXITCODE
if ($loweringOutput) {
    $loweringOutput | Set-Content -LiteralPath $loweringLog -Encoding utf8
}
if ($loweringExitCode -ne 0) {
    throw "Stream-to-schedule direct lowering failed; see $loweringLog"
}
$loweringOutput = & $optimizer `
    --input $sourceSchedule `
    --output $sourceCommand `
    --pipeline ftlpu-schedule-to-commands `
    --mxm-execution vector `
    --ffn-schedule fused `
    --projection-rope-overlap off `
    --target-config $loweringTarget `
    --weight-bank 1 `
    --kv-cache-capacity 256 2>&1
$loweringExitCode = $LASTEXITCODE
if ($loweringOutput) {
    $loweringOutput | Add-Content -LiteralPath $loweringLog -Encoding utf8
}
if ($loweringExitCode -ne 0) {
    throw "Schedule-to-command direct lowering failed; see $loweringLog"
}

# Compile every target. This ensures the frequency in the executable matches
# the directory name; changing only the runtime DDR variable is not enough.
foreach ($combination in $combinations) {
    $directory = Join-Path $resultRoot $combination.Name
    $configPath = Join-Path $directory "target.json"
    $programPath = Join-Path $directory "compiled.ftlpu"
    $compileLog = Join-Path $directory "compile.log"
    $programIsStale = -not (Test-Path -LiteralPath $programPath) -or
        (Get-Item -LiteralPath $sourceCommand).LastWriteTimeUtc -gt
            (Get-Item -LiteralPath $programPath).LastWriteTimeUtc
    if (-not $Resume -or $programIsStale) {
        $compileOutput = & $compiler `
            --input $sourceCommand `
            --output $programPath `
            --input-stage command `
            --target-config $configPath `
            --mxm-execution vector `
            --weight-bank 1 `
            --kv-cache-capacity 256 2>&1
        $compileExitCode = $LASTEXITCODE
        if ($compileOutput) {
            $compileOutput | Set-Content -LiteralPath $compileLog -Encoding utf8
        }
        if ($compileExitCode -ne 0) {
            throw "Compilation failed for $($combination.Name); see $compileLog"
        }
    }
}

$pending = [System.Collections.Generic.Queue[object]]::new()
foreach ($combination in $combinations) {
    $manifestPath = Join-Path (Join-Path $resultRoot $combination.Name) "run.json"
    $passed = $false
    if (Test-Path -LiteralPath $manifestPath) {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw |
            ConvertFrom-Json
        $passed = $manifest.status -eq "passed"
    }
    if (-not $Resume -or -not $passed) {
        $pending.Enqueue($combination)
    }
}
$running = @()
$failed = @()

while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    while ($pending.Count -gt 0 -and $running.Count -lt $MaxParallel) {
        $combination = $pending.Dequeue()
        $directory = Join-Path $resultRoot $combination.Name
        $programPath = Join-Path $directory "compiled.ftlpu"
        $stdoutPath = Join-Path $directory "runner.stdout.log"
        $stderrPath = Join-Path $directory "runner.stderr.log"
        $arguments = @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runner,
            "-Program", $programPath,
            "-DdrBandwidthMBps", [string]$combination.BandwidthMBps,
            "-ClockMHz", [string]$combination.ClockMHz,
            "-ResultDir", $directory
        )
        if ($NoPipeline) {
            $arguments += "-NoPipeline"
        }
        $process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath `
            -WindowStyle Hidden -PassThru
        $running += [pscustomobject]@{
            Combination = $combination
            Process = $process
            StdoutPath = $stdoutPath
            StderrPath = $stderrPath
        }
        Write-Host "Started $($combination.Name) pid=$($process.Id)"
    }

    if ($running.Count -eq 0) {
        break
    }
    Start-Sleep -Seconds 5
    $stillRunning = @()
    foreach ($entry in $running) {
        if (-not $entry.Process.HasExited) {
            $stillRunning += $entry
            continue
        }
        $manifestPath = Join-Path `
            (Join-Path $resultRoot $entry.Combination.Name) "run.json"
        $passed = $false
        if (Test-Path -LiteralPath $manifestPath) {
            $manifest = Get-Content -LiteralPath $manifestPath -Raw |
                ConvertFrom-Json
            $passed = $manifest.status -eq "passed"
        }
        if (-not $passed) {
            $failed += $entry.Combination.Name
            Write-Host "Failed $($entry.Combination.Name) exit=$($entry.Process.ExitCode)"
        } else {
            Write-Host "Finished $($entry.Combination.Name)"
        }
    }
    $running = $stillRunning
}

$summary = foreach ($combination in $combinations) {
    $directory = Join-Path $resultRoot $combination.Name
    $manifestPath = Join-Path $directory "run.json"
    if (Test-Path -LiteralPath $manifestPath) {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $maxCycle = ""
        $initialWaitCycles = ""
        $runtimeWaitCycles = ""
        $failure = if ($manifest.PSObject.Properties.Name -contains "failure") {
            [string]$manifest.failure
        } else {
            ""
        }
        $testLog = Join-Path $directory "test.log"
        if (Test-Path -LiteralPath $testLog) {
            $text = Get-Content -LiteralPath $testLog -Raw
            if ($text -match "max_cycle=(\d+)") { $maxCycle = $Matches[1] }
            if ($text -match "initial_wait_cycles=(\d+)") { $initialWaitCycles = $Matches[1] }
            if ($text -match "runtime_wait_cycles=(\d+)") { $runtimeWaitCycles = $Matches[1] }
        }
        [pscustomobject]@{
            clock_mhz = $combination.ClockMHz
            bandwidth_mbytes_per_second = $combination.BandwidthMBps
            status = $manifest.status
            max_cycle = $maxCycle
            initial_wait_cycles = $initialWaitCycles
            runtime_wait_cycles = $runtimeWaitCycles
            wall_time_seconds = $manifest.elapsed_seconds
            failure = $failure
            result_directory = $combination.Name
        }
    } else {
        [pscustomobject]@{
            clock_mhz = $combination.ClockMHz
            bandwidth_mbytes_per_second = $combination.BandwidthMBps
            status = "failed"
            max_cycle = ""
            initial_wait_cycles = ""
            runtime_wait_cycles = ""
            wall_time_seconds = ""
            failure = "run.json was not produced"
            result_directory = $combination.Name
        }
    }
}
$summary | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding utf8

if ($failed.Count -gt 0) {
    throw "Sweep failed: $($failed -join ', ')"
}
Write-Host "Qwen2.5 prefill sweep complete: $summaryPath"
