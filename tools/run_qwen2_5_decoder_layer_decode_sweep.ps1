param(
    [int]$MaxParallel = 3,
    [switch]$Resume,
    [switch]$SkipNumericChecks,
    [ValidateRange(1, 255)]
    [int]$PastLen = 32,
    [string]$ResultRoot = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$resultRoot = if ($ResultRoot) {
    [System.IO.Path]::GetFullPath($ResultRoot)
} else {
    Join-Path $repoRoot "results\qwen2_5_decoder_layer_decode"
}
$sweepRoot = Join-Path $resultRoot "sweep"
$sourceStablehlo = Join-Path $resultRoot "decoder.stablehlo.mlir"
$fixture = Join-Path $resultRoot "fixture"
$baseConfig = Join-Path (Split-Path -Parent $repoRoot) "FTLPU-CMODEL\config\ftlpu-lpu32.json"
$optimizer = Join-Path $repoRoot "build-ftlpu-vs2026-direct\compiler\ftlpu_opt.exe"
$compiler = Join-Path $repoRoot "build-ftlpu-vs2026-direct\compiler\ftlpu-compile.exe"
$runner = Join-Path $PSScriptRoot "run_qwen2_5_decoder_layer_decode.ps1"
$summaryPath = Join-Path $resultRoot "sweep.csv"
$reportPath = Join-Path $resultRoot "sweep.md"

if ($MaxParallel -le 0) {
    throw "MaxParallel must be positive"
}
if ($PastLen % 32 -ne 0) {
    throw "PastLen must be a positive multiple of 32"
}
foreach ($path in @($sourceStablehlo, $fixture, $baseConfig, $optimizer, $compiler, $runner)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required sweep input does not exist: $path"
    }
}

$combinations = foreach ($clock in @(500, 750, 1000)) {
    foreach ($bandwidth in @(25600, 51200, 68000, 85600)) {
        [pscustomobject]@{
            ClockMHz = $clock
            BandwidthMBps = $bandwidth
            Name = "f${clock}_bw${bandwidth}"
        }
    }
}

New-Item -ItemType Directory -Path $sweepRoot -Force | Out-Null
$base = Get-Content -LiteralPath $baseConfig -Raw | ConvertFrom-Json
foreach ($combination in $combinations) {
    $directory = Join-Path $sweepRoot $combination.Name
    if (-not $Resume -and (Test-Path -LiteralPath $directory)) {
        Remove-Item -LiteralPath $directory -Recurse -Force
    }
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $configPath = Join-Path $directory "target.json"
    $config = $base | ConvertTo-Json -Depth 100 | ConvertFrom-Json
    $config.external_memory.lpu_clock_mhz = $combination.ClockMHz
    $config.external_memory.ddr_peak_bandwidth_mbytes_per_second = $combination.BandwidthMBps
    [System.IO.File]::WriteAllText(
        $configPath,
        ($config | ConvertTo-Json -Depth 100),
        [System.Text.UTF8Encoding]::new($false))
}

foreach ($combination in $combinations) {
    $directory = Join-Path $sweepRoot $combination.Name
    $manifestPath = Join-Path $directory "run.json"
    if ($Resume -and (Test-Path -LiteralPath $manifestPath)) {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        if ($manifest.status -eq "passed") {
            continue
        }
    }
    $configPath = Join-Path $directory "target.json"
    $streamPath = Join-Path $directory "decoder.stream.mlir"
    $schedulePath = Join-Path $directory "decoder.schedule.mlir"
    $commandPath = Join-Path $directory "decoder.command.mlir"
    $programPath = Join-Path $directory "compiled.ftlpu"
    $compileLog = Join-Path $directory "compile.log"
    $common = @(
        "--mxm-execution", "native4",
        "--ffn-schedule", "fused",
        "--projection-rope-overlap", "off",
        "--target-config", $configPath,
        "--weight-bank", "1",
        "--kv-cache-capacity", "256",
        "--decode-past-len", [string]$PastLen,
        "--decode-current-len", "1"
    )
    $log = @()
    $steps = @()
    $steps += ,(@($optimizer, "--input", $sourceStablehlo, "--output", $streamPath,
        "--pipeline", "ftlpu-stablehlo-to-stream") + $common)
    $steps += ,(@($optimizer, "--input", $streamPath, "--output", $schedulePath,
        "--pipeline", "ftlpu-stream-to-schedule") + $common)
    $steps += ,(@($optimizer, "--input", $schedulePath, "--output", $commandPath,
        "--pipeline", "ftlpu-schedule-to-commands") + $common)
    $steps += ,@($compiler, "--input", $commandPath, "--output", $programPath,
        "--input-stage", "command", "--target-config", $configPath,
        "--mxm-execution", "native4", "--weight-bank", "1",
        "--kv-cache-capacity", "256")
    foreach ($step in $steps) {
        $stepOutput = & $step[0] $step[1..($step.Count - 1)] 2>&1
        $stepExitCode = $LASTEXITCODE
        if ($stepOutput) { $log += $stepOutput }
        if ($stepExitCode -ne 0) {
            $log | Set-Content -LiteralPath $compileLog -Encoding utf8
            throw "Compilation failed for $($combination.Name); see $compileLog"
        }
    }
    $log | Set-Content -LiteralPath $compileLog -Encoding utf8
}

$pending = [System.Collections.Generic.Queue[object]]::new()
foreach ($combination in $combinations) {
    $manifestPath = Join-Path (Join-Path $sweepRoot $combination.Name) "run.json"
    $passed = $false
    if (Test-Path -LiteralPath $manifestPath) {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
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
        $directory = Join-Path $sweepRoot $combination.Name
        $arguments = @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runner,
            "-Program", (Join-Path $directory "compiled.ftlpu"),
            "-Fixture", $fixture,
            "-DdrBandwidthMBps", [string]$combination.BandwidthMBps,
            "-ClockMHz", [string]$combination.ClockMHz,
            "-ResultDir", $directory
        )
        if ($SkipNumericChecks) {
            $arguments += "-SkipNumericChecks"
        }
        $process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments `
            -RedirectStandardOutput (Join-Path $directory "runner.stdout.log") `
            -RedirectStandardError (Join-Path $directory "runner.stderr.log") `
            -WindowStyle Hidden -PassThru
        $running += [pscustomobject]@{
            Combination = $combination
            Process = $process
        }
        Write-Host "Started $($combination.Name) pid=$($process.Id)"
    }

    if ($running.Count -eq 0) { break }
    Start-Sleep -Seconds 5
    $stillRunning = @()
    foreach ($entry in $running) {
        if (-not $entry.Process.HasExited) {
            $stillRunning += $entry
            continue
        }
        $directory = Join-Path $sweepRoot $entry.Combination.Name
        $manifestPath = Join-Path $directory "run.json"
        $passed = $false
        if (Test-Path -LiteralPath $manifestPath) {
            $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
            $passed = $manifest.status -eq "passed"
        }
        if ($passed) {
            Write-Host "Finished $($entry.Combination.Name)"
        } else {
            $failed += $entry.Combination.Name
            Write-Host "Failed $($entry.Combination.Name) exit=$($entry.Process.ExitCode)"
        }
    }
    $running = $stillRunning
}

$summary = foreach ($combination in $combinations) {
    $directory = Join-Path $sweepRoot $combination.Name
    $manifestPath = Join-Path $directory "run.json"
    $status = "failed"
    $wallTime = ""
    $failure = "run.json was not produced"
    if (Test-Path -LiteralPath $manifestPath) {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $status = [string]$manifest.status
        $wallTime = $manifest.elapsed_seconds
        $failure = if ($null -ne $manifest.failure) { [string]$manifest.failure } else { "" }
    }
    $values = @{}
    $testLog = Join-Path $directory "test.log"
    if (Test-Path -LiteralPath $testLog) {
        $text = Get-Content -LiteralPath $testLog -Raw
        foreach ($name in @(
            "cycles", "initial_wait_cycles", "runtime_page_wait_cycles",
            "weight_wait_cycles", "state_page_in_cycles",
            "state_page_out_cycles", "c2c_ingress_cycles",
            "c2c_egress_cycles")) {
            if ($text -match "(?:^|\s)$name=(\d+)") {
                $values[$name] = [int64]$Matches[1]
            }
        }
    }
    $computeCycles = if ($values.ContainsKey("cycles")) { $values.cycles } else { "" }
    $initialWait = if ($values.ContainsKey("initial_wait_cycles")) { $values.initial_wait_cycles } else { "" }
    $runtimeWait = if ($values.ContainsKey("runtime_page_wait_cycles")) { $values.runtime_page_wait_cycles } else { "" }
    $coldLayerMs = ""
    $ingressCycles = if ($values.ContainsKey("c2c_ingress_cycles")) { $values.c2c_ingress_cycles } else { "" }
    $egressCycles = if ($values.ContainsKey("c2c_egress_cycles")) { $values.c2c_egress_cycles } else { "" }
    if ($computeCycles -ne "" -and $initialWait -ne "" -and
        $runtimeWait -ne "" -and $ingressCycles -ne "" -and
        $egressCycles -ne "") {
        $coldLayerMs = [math]::Round(
            ($computeCycles + $initialWait + $runtimeWait +
                $ingressCycles + $egressCycles) /
                ($combination.ClockMHz * 1000.0), 6)
    }
    [pscustomobject]@{
        clock_mhz = $combination.ClockMHz
        bandwidth_mbytes_per_second = $combination.BandwidthMBps
        status = $status
        compute_cycles = $computeCycles
        initial_wait_cycles = $initialWait
        runtime_wait_cycles = $runtimeWait
        weight_wait_cycles = if ($values.ContainsKey("weight_wait_cycles")) { $values.weight_wait_cycles } else { "" }
        state_page_in_cycles = if ($values.ContainsKey("state_page_in_cycles")) { $values.state_page_in_cycles } else { "" }
        state_page_out_cycles = if ($values.ContainsKey("state_page_out_cycles")) { $values.state_page_out_cycles } else { "" }
        c2c_ingress_cycles = $ingressCycles
        c2c_egress_cycles = $egressCycles
        cold_layer_ms = $coldLayerMs
        wall_time_seconds = $wallTime
        failure = $failure
        result_directory = "sweep/$($combination.Name)"
    }
}
$summary | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding utf8

$validationText = if ($SkipNumericChecks) {
    "每组均使用独立 target.json 编译，并运行完整 ICU/CModel decode；本轮跳过数值门限。"
} else {
    "每组均使用独立 target.json 编译，并运行完整 ICU/CModel decode 数值检查。"
}
$report = @(
    "# Qwen2.5-1.5B 单层 decode DDR/时钟 sweep（past_len=$PastLen）",
    "",
    $validationText,
    "cold_layer_ms 与 prefill sweep 保持相同口径：",
    "(compute_cycles + initial_wait_cycles + runtime_wait_cycles + c2c_ingress_cycles + c2c_egress_cycles) / clock。",
    "",
    "| LPU 时钟 | DDR 峰值 | 状态 | compute cycles | initial wait | runtime wait | cold layer |",
    "| ---: | ---: | :---: | ---: | ---: | ---: | ---: |"
)
foreach ($row in $summary) {
    $bandwidthGb = [math]::Round($row.bandwidth_mbytes_per_second / 1000.0, 1)
    $resultText = if ($row.status -eq "passed") { "通过" } else { "失败" }
    $coldText = if ($row.cold_layer_ms -eq "") { "-" } else { "$($row.cold_layer_ms) ms" }
    $report += "| $($row.clock_mhz) MHz | $bandwidthGb GB/s | $resultText | $($row.compute_cycles) | $($row.initial_wait_cycles) | $($row.runtime_wait_cycles) | $coldText |"
}
$report += @(
    "",
    "完整真实逐周期 pipeline 保留在同目录的 pipeline.csv。为避免重复产生 12 份大型 trace，",
    "sweep 子目录仅保存各组合的 runtime-linked program、真实 pre-execution programs、",
    "KV 写回、按物理 ICU 导出的实际运行程序以及日志。"
)
$report | Set-Content -LiteralPath $reportPath -Encoding utf8

if ($failed.Count -gt 0) {
    throw "Decode sweep failed: $($failed -join ', ')"
}
Write-Host "Qwen2.5 decode past_len=$PastLen sweep complete: $summaryPath"
