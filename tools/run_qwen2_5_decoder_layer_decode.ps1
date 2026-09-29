param(
    [Parameter(Mandatory = $true)]
    [string]$Program,
    [Parameter(Mandatory = $true)]
    [string]$Fixture,
    [Parameter(Mandatory = $true)]
    [uint32]$DdrBandwidthMBps,
    [Parameter(Mandatory = $true)]
    [uint32]$ClockMHz,
    [Parameter(Mandatory = $true)]
    [string]$ResultDir,
    [string]$PipelineCsv = "",
    [string]$MemCsv = "",
    [switch]$SkipNumericChecks
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$testExe = Join-Path $repoRoot "build-ftlpu-vs2026-direct\runtime\compiled_qwen_real_decoder_layer_runtime_test.exe"
$icuExporter = Join-Path $repoRoot "build-ftlpu-vs2026-direct\runtime\ftlpu_icu_program_export.exe"

function Get-CompatibleRelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BasePath,
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $base = [System.IO.Path]::GetFullPath($BasePath)
    if (-not $base.EndsWith([string][System.IO.Path]::DirectorySeparatorChar)) {
        $base += [System.IO.Path]::DirectorySeparatorChar
    }
    $baseUri = [System.Uri]$base
    $pathUri = [System.Uri]([System.IO.Path]::GetFullPath($Path))
    return [System.Uri]::UnescapeDataString(
        $baseUri.MakeRelativeUri($pathUri).ToString()
    ).Replace('/', [System.IO.Path]::DirectorySeparatorChar)
}

$resultDir = [System.IO.Path]::GetFullPath($ResultDir)
$programCopy = Join-Path $resultDir "program.ftlpu"
$linkedProgram = Join-Path $resultDir "linked.ftlpu"
$preExecutionDir = Join-Path $resultDir "pre_execution_programs"
$kvDir = Join-Path $resultDir "kv"
$icuProgramDir = Join-Path $resultDir "runtime_icu_programs"
$testLog = Join-Path $resultDir "test.log"
$icuExportLog = Join-Path $resultDir "icu_export.log"
$manifest = Join-Path $resultDir "run.json"
$pipelineCsvPath = if ($PipelineCsv) {
    [System.IO.Path]::GetFullPath($PipelineCsv)
} else { "" }
$memCsvPath = if ($MemCsv) {
    [System.IO.Path]::GetFullPath($MemCsv)
} else { "" }

foreach ($path in @($testExe, $icuExporter, $Program, $Fixture)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required decode input does not exist: $path"
    }
}

New-Item -ItemType Directory -Path $resultDir -Force | Out-Null
foreach ($directory in @($preExecutionDir, $kvDir, $icuProgramDir)) {
    if (Test-Path -LiteralPath $directory) {
        Remove-Item -LiteralPath $directory -Recurse -Force
    }
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}
Copy-Item -LiteralPath $Program -Destination $programCopy -Force
foreach ($path in @($linkedProgram, $testLog, $icuExportLog, $manifest)) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Force
    }
}

$savedEnvironment = @{
    Bandwidth = $env:FTLPU_DDR_BANDWIDTH_MBPS
    Pipeline = $env:FTLPU_QWEN_PIPELINE_CSV
    MemTrace = $env:FTLPU_QWEN_MEM_CSV
    Linked = $env:FTLPU_QWEN_C2C_LINKED_BINARY
    PreExecution = $env:FTLPU_QWEN_C2C_PRE_EXECUTION_DIR
    KvDump = $env:FTLPU_QWEN_KV_DUMP_DIR
    SkipNumericChecks = $env:FTLPU_SKIP_QWEN_NUMERIC_CHECK
}

try {
    $env:FTLPU_DDR_BANDWIDTH_MBPS = [string]$DdrBandwidthMBps
    if ($pipelineCsvPath) {
        $pipelineParent = Split-Path -Parent $pipelineCsvPath
        New-Item -ItemType Directory -Path $pipelineParent -Force | Out-Null
        Remove-Item -LiteralPath $pipelineCsvPath -Force -ErrorAction SilentlyContinue
        $env:FTLPU_QWEN_PIPELINE_CSV = $pipelineCsvPath
    } else {
        Remove-Item Env:FTLPU_QWEN_PIPELINE_CSV -ErrorAction SilentlyContinue
    }
    if ($memCsvPath) {
        $memParent = Split-Path -Parent $memCsvPath
        New-Item -ItemType Directory -Path $memParent -Force | Out-Null
        Remove-Item -LiteralPath $memCsvPath -Force -ErrorAction SilentlyContinue
        $env:FTLPU_QWEN_MEM_CSV = $memCsvPath
    } else {
        Remove-Item Env:FTLPU_QWEN_MEM_CSV -ErrorAction SilentlyContinue
    }
    $env:FTLPU_QWEN_C2C_LINKED_BINARY = $linkedProgram
    $env:FTLPU_QWEN_C2C_PRE_EXECUTION_DIR = $preExecutionDir
    $env:FTLPU_QWEN_KV_DUMP_DIR = $kvDir
    if ($SkipNumericChecks) {
        $env:FTLPU_SKIP_QWEN_NUMERIC_CHECK = "1"
    } else {
        Remove-Item Env:FTLPU_SKIP_QWEN_NUMERIC_CHECK -ErrorAction SilentlyContinue
    }

    $started = Get-Date
    $savedErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & $testExe $programCopy $Fixture 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedErrorActionPreference
    }
    $output | Set-Content -LiteralPath $testLog -Encoding utf8

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
    $failure = $null
    if ($exitCode -ne 0) {
        $failure = (($output | Out-String).Trim() -split "`r?`n" |
            Select-Object -First 3) -join " "
    } elseif ($icuExitCode -ne 0) {
        $failure = "runtime ICU export failed with exit code $icuExitCode"
    }
    $status = if ($null -eq $failure) { "passed" } else { "failed" }
    [ordered]@{
        test = "qwen2_5_decoder_layer_decode"
        status = $status
        exit_code = $exitCode
        icu_export_exit_code = $icuExitCode
        started_at = $started.ToString("o")
        finished_at = $finished.ToString("o")
        elapsed_seconds = ($finished - $started).TotalSeconds
        lpu_clock_mhz = $ClockMHz
        ddr_bandwidth_mbytes_per_second = $DdrBandwidthMBps
        numeric_validation = if ($SkipNumericChecks) { "skipped" } else { "enabled" }
        failure = $failure
        program = "program.ftlpu"
        linked_program = if (Test-Path -LiteralPath $linkedProgram) { "linked.ftlpu" } else { $null }
        pre_execution_programs = "pre_execution_programs"
        kv = "kv"
        runtime_icu_programs = if ($null -ne $icuExitCode) { "runtime_icu_programs" } else { $null }
        pipeline_csv = if ($pipelineCsvPath -and (Test-Path -LiteralPath $pipelineCsvPath)) {
            Get-CompatibleRelativePath $resultDir $pipelineCsvPath
        } else { $null }
        mem_csv = if ($memCsvPath -and (Test-Path -LiteralPath $memCsvPath)) {
            Get-CompatibleRelativePath $resultDir $memCsvPath
        } else { $null }
        log = "test.log"
        icu_export_log = if ($null -ne $icuExitCode) { "icu_export.log" } else { $null }
    } | ConvertTo-Json | Set-Content -LiteralPath $manifest -Encoding utf8

    if ($null -ne $failure) {
        throw $failure
    }
} finally {
    $env:FTLPU_DDR_BANDWIDTH_MBPS = $savedEnvironment.Bandwidth
    $env:FTLPU_QWEN_PIPELINE_CSV = $savedEnvironment.Pipeline
    $env:FTLPU_QWEN_MEM_CSV = $savedEnvironment.MemTrace
    $env:FTLPU_QWEN_C2C_LINKED_BINARY = $savedEnvironment.Linked
    $env:FTLPU_QWEN_C2C_PRE_EXECUTION_DIR = $savedEnvironment.PreExecution
    $env:FTLPU_QWEN_KV_DUMP_DIR = $savedEnvironment.KvDump
    $env:FTLPU_SKIP_QWEN_NUMERIC_CHECK = $savedEnvironment.SkipNumericChecks
}

Write-Host "Updated Qwen2.5 decode result: $resultDir"
