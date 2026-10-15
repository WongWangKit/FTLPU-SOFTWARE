#!/usr/bin/env bash
# Configure one reusable Linux environment; run as root for /opt installation.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${FTLPU_XLA_HOME:-/opt/ftlpu-xla}"
mkdir -p "$env_dir"
exec > >(tee -a "$env_dir/setup.log") 2>&1

if ! command -v python3 >/dev/null || ! python3 -m venv --help >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv ca-certificates
fi
if [[ ! -x "$env_dir/venv/bin/python" ]] || ! "$env_dir/venv/bin/python" -m pip --version >/dev/null 2>&1; then
  # Some distributions have the venv module but lack ensurepip until the
  # python3-venv package is installed.
  if ! python3 -m venv "$env_dir/venv"; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv ca-certificates
    python3 -m venv "$env_dir/venv"
  fi
fi
"$env_dir/venv/bin/python" -m pip install pip==26.2.1
"$env_dir/venv/bin/python" -m pip install --index-url https://pypi.org/simple \
  -r "$repo_dir/tools/requirements.xla-linux.lock.txt"
"$env_dir/venv/bin/python" -m pip check
"$env_dir/venv/bin/python" -m pip freeze > "$env_dir/requirements.lock.txt"
mkdir -p "$env_dir/hf-cache"
PJRT_DEVICE=CPU "$env_dir/venv/bin/python" - <<'PY'
import torch
import torch_xla
from torch_xla.stablehlo import exported_program_to_stablehlo

model = torch.nn.Linear(4, 3).eval()
inputs = (torch.randn(2, 4),)
with torch.inference_mode():
    exported = torch.export.export(model, inputs, strict=False)
    program = exported_program_to_stablehlo(exported)
    torch.testing.assert_close(program(*inputs).cpu(), model(*inputs))
print(f'CPU XLA conversion verified: torch={torch.__version__}, torch_xla={torch_xla.__version__}')
PY
printf 'Environment: %s/venv\nDependency lock: %s/requirements.lock.txt\n' "$env_dir" "$env_dir"
