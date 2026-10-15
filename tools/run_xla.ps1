<# Reuse the dedicated Linux interpreter from the repository root. #>
$PythonArguments = $args
$frontendRepository = Split-Path -Parent $PSScriptRoot
$PythonArguments = @($PythonArguments | ForEach-Object {
    if ($_ -match '^[A-Za-z]:[\\/]') {
        $linuxArgument = & wsl.exe -d FTLPU-XLA -u root --exec wslpath -a $_
        if ($LASTEXITCODE -ne 0) { throw "Cannot translate Windows path: $_" }
        $linuxArgument.Trim()
    } else {
        $_
    }
})
& wsl.exe -d FTLPU-XLA -u root --cd $frontendRepository --exec env `
    PJRT_DEVICE=CPU HF_HOME=/opt/ftlpu-xla/hf-cache HF_HUB_CACHE=.cache/hf-hub `
    PYTHONPYCACHEPREFIX=/opt/ftlpu-xla/pycache `
    /opt/ftlpu-xla/venv/bin/python @PythonArguments
exit $LASTEXITCODE
