param(
    [string]$Program = "E:\timesintelli\workspace2\projection_rope_run\ttft_sweep\f500_bw25600.pagesync.ftlpu",
    [uint32]$DdrBandwidthMBps = 25600,
    [uint32]$ClockMHz = 500,
    [string]$ResultDir = "",
    [switch]$NoPipeline
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ResultDir)) {
    $resultDir = Join-Path $repoRoot "results\qwen2_5_decoder_layer_prefill"
} elseif ([System.IO.Path]::IsPathRooted($ResultDir)) {
    $resultDir = $ResultDir
} else {
    $resultDir = Join-Path $repoRoot $ResultDir
}
$testExe = Join-Path $repoRoot "build-ftlpu-vs2026-direct\runtime\compiled_qwen2_5_1_5b_decoder_layer_seq32_runtime_test.exe"
$icuExporter = Join-Path $repoRoot "build-ftlpu-vs2026-direct\runtime\ftlpu_icu_program_export.exe"
$programCopy = Join-Path $resultDir "program.ftlpu"
$pipelineCsv = Join-Path $resultDir "runtime.pipeline.csv"
$linkedProgram = Join-Path $resultDir "linked.ftlpu"
$preExecutionDir = Join-Path $resultDir "pre_execution_programs"
$icuProgramDir = Join-Path $resultDir "icu_programs"
$testLog = Join-Path $resultDir "test.log"
$icuExportLog = Join-Path $resultDir "icu_export.log"
$manifest = Join-Path $resultDir "run.json"

if (-not (Test-Path -LiteralPath $testExe)) {
    throw "Qwen2.5 prefill runtime test is not built: $testExe"
}
if (-not (Test-Path -LiteralPath $icuExporter)) {
    throw "ICU program exporter is not built: $icuExporter"
}
if (-not (Test-Path -LiteralPath $Program)) {
    throw "Qwen2.5 prefill program does not exist: $Program"
}

New-Item -ItemType Directory -Path $resultDir -Force | Out-Null
foreach ($directory in @($preExecutionDir, $icuProgramDir)) {
    if (Test-Path -LiteralPath $directory) {
        Remove-Item -LiteralPath $directory -Recurse -Force
    }
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}
Copy-Item -LiteralPath $Program -Destination $programCopy -Force
foreach ($path in @($pipelineCsv, $linkedProgram, $testLog, $icuExportLog, $manifest)) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Force
    }
}

$oldBandwidth = $env:FTLPU_DDR_BANDWIDTH_MBPS
$oldPipeline = $env:FTLPU_QWEN_C2C_PIPELINE_CSV
$oldLinked = $env:FTLPU_QWEN_C2C_LINKED_BINARY
$oldPreExecution = $env:FTLPU_QWEN_C2C_PRE_EXECUTION_DIR
try {
    $env:FTLPU_DDR_BANDWIDTH_MBPS = [string]$DdrBandwidthMBps
    if ($NoPipeline) {
        Remove-Item Env:FTLPU_QWEN_C2C_PIPELINE_CSV -ErrorAction SilentlyContinue
    } else {
        $env:FTLPU_QWEN_C2C_PIPELINE_CSV = $pipelineCsv
    }
    $env:FTLPU_QWEN_C2C_LINKED_BINARY = $linkedProgram
    $env:FTLPU_QWEN_C2C_PRE_EXECUTION_DIR = $preExecutionDir

    $started = Get-Date
    $savedErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & $testExe $programCopy 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedErrorActionPreference
    }
    $output | Tee-Object -FilePath $testLog
    if ($exitCode -ne 0) {
        $icuExitCode = $null
        if (Test-Path -LiteralPath $linkedProgram) {
            try {
                $ErrorActionPreference = "Continue"
                $icuOutput = & $icuExporter $linkedProgram $icuProgramDir `
                    --pre-execution-dir $preExecutionDir 2>&1
                $icuExitCode = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $savedErrorActionPreference
            }
            $icuOutput | Set-Content -LiteralPath $icuExportLog -Encoding utf8
        }
        $finished = Get-Date
        $failureText = (($output | Out-String).Trim() -split "`r?`n" |
            Select-Object -First 3) -join " "
        [ordered]@{
            test = "qwen2_5_decoder_layer_prefill"
            status = "failed"
            exit_code = $exitCode
            icu_export_exit_code = $icuExitCode
            started_at = $started.ToString("o")
            finished_at = $finished.ToString("o")
            elapsed_seconds = ($finished - $started).TotalSeconds
            lpu_clock_mhz = $ClockMHz
            ddr_bandwidth_mbytes_per_second = $DdrBandwidthMBps
            failure = $failureText
            program = "program.ftlpu"
            runtime_pipeline = if ($NoPipeline) { $null } else { "runtime.pipeline.csv" }
            linked_program = "linked.ftlpu"
            pre_execution_programs = "pre_execution_programs"
            icu_programs = "icu_programs"
            log = "test.log"
            icu_export_log = if ($icuExitCode -ne $null) { "icu_export.log" } else { $null }
        } | ConvertTo-Json | Set-Content -LiteralPath $manifest -Encoding utf8
        throw "Qwen2.5 decoder-layer prefill test failed with exit code $exitCode"
    }

    try {
        $ErrorActionPreference = "Continue"
        $icuOutput = & $icuExporter $linkedProgram $icuProgramDir `
            --pre-execution-dir $preExecutionDir 2>&1
        $icuExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedErrorActionPreference
    }
    $icuOutput | Tee-Object -FilePath $icuExportLog
    if ($icuExitCode -ne 0) {
        throw "ICU program export failed with exit code $icuExitCode"
    }

    $finished = Get-Date
    [ordered]@{
        test = "qwen2_5_decoder_layer_prefill"
        status = "passed"
        exit_code = $exitCode
        icu_export_exit_code = $icuExitCode
        started_at = $started.ToString("o")
        finished_at = $finished.ToString("o")
        elapsed_seconds = ($finished - $started).TotalSeconds
        lpu_clock_mhz = $ClockMHz
        ddr_bandwidth_mbytes_per_second = $DdrBandwidthMBps
        program = "program.ftlpu"
        runtime_pipeline = if ($NoPipeline) { $null } else { "runtime.pipeline.csv" }
        linked_program = "linked.ftlpu"
        pre_execution_programs = "pre_execution_programs"
        icu_programs = "icu_programs"
        log = "test.log"
        icu_export_log = "icu_export.log"
    } | ConvertTo-Json | Set-Content -LiteralPath $manifest -Encoding utf8
} finally {
    $env:FTLPU_DDR_BANDWIDTH_MBPS = $oldBandwidth
    $env:FTLPU_QWEN_C2C_PIPELINE_CSV = $oldPipeline
    $env:FTLPU_QWEN_C2C_LINKED_BINARY = $oldLinked
    $env:FTLPU_QWEN_C2C_PRE_EXECUTION_DIR = $oldPreExecution
}

Write-Host "Updated Qwen2.5 prefill results: $resultDir"
