param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,
    [Parameter(Mandatory = $true)]
    [uint32]$PastLen,
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
$selector = Join-Path $PSScriptRoot "select_qwen2_5_decode_bucket.py"
$runner = Join-Path $PSScriptRoot "run_qwen2_5_decoder_layer_decode.ps1"
foreach ($path in @($Manifest, $selector, $runner)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required decode bucket input does not exist: $path"
    }
}

$selectionText = & python -B $selector --manifest $Manifest --past-len $PastLen
if ($LASTEXITCODE -ne 0) {
    throw "Decode bucket selection failed for past_len=$PastLen"
}
$selection = $selectionText | ConvertFrom-Json
$arguments = @{
    Program = $selection.program
    Fixture = $selection.fixture
    DdrBandwidthMBps = $DdrBandwidthMBps
    ClockMHz = $ClockMHz
    ResultDir = $ResultDir
}
if ($PipelineCsv) { $arguments.PipelineCsv = $PipelineCsv }
if ($MemCsv) { $arguments.MemCsv = $MemCsv }
if ($SkipNumericChecks) { $arguments.SkipNumericChecks = $true }

& $runner @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Decode bucket runtime failed for past_len=$PastLen"
}
