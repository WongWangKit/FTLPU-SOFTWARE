#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/software/runtime/cmodel_device_backend.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main()
{
#ifdef FTLPU_TRANSFORMER_EVAL_CONFIG
    const std::string json = R"json(
{
  "name": "lpu_cmodel_transformer_eval_v1",
  "memory": {
    "hemispheres": 2,
    "slices_per_hemisphere": 44,
    "banks_per_slice": 2,
    "words_per_bank": 40960,
    "bytes_per_word": 8
  },
  "streams": {
    "streams_per_direction": 32,
    "encoded_streams": 64
  },
  "throughput": {
    "tile_rows": 4,
    "lanes_per_tile": 8,
    "mem_read_bytes_per_cycle": 8,
    "mem_write_bytes_per_cycle": 8,
    "mxm_rows": 32,
    "mxm_columns": 32,
    "mxm_load_streams_per_cycle": 16,
    "mxm_load_bytes_per_cycle": 64,
    "mxms_per_hemisphere": 2,
    "vxm_alus": 8
  }
}
)json";
#else
    const std::string json = R"json(
{
  "name": "lpu_cmodel_groqlike_v1",
  "memory": {
    "hemispheres": 2,
    "slices_per_hemisphere": 44,
    "banks_per_slice": 2,
    "words_per_bank": 4096,
    "bytes_per_word": 16
  },
  "streams": {
    "streams_per_direction": 32,
    "encoded_streams": 64
  },
  "throughput": {
    "tile_rows": 20,
    "lanes_per_tile": 16,
    "mem_read_bytes_per_cycle": 16,
    "mem_write_bytes_per_cycle": 16,
    "mxm_rows": 320,
    "mxm_columns": 320,
    "mxm_load_streams_per_cycle": 16,
    "mxm_load_bytes_per_cycle": 256,
    "mxms_per_hemisphere": 1,
    "vxm_alus": 8
  }
}
)json";
#endif
    std::string error;
    auto compiler_target =
        ftlpu::compiler::target::LPUTargetModel::from_json(
            json, error);
    if (mlir::failed(compiler_target)) {
        throw std::runtime_error(
            "failed to parse compiler target: " + error);
    }
    const auto backend_target =
        ftlpu::software::runtime::make_cmodel_target_description();
    if (compiler_target->executable_target_identity()
        != backend_target.executable_target) {
        throw std::runtime_error(
            "compiler and CModel backend executable ABI identities differ");
    }
    std::cout << "executable_target_abi_test passed\n";
    return 0;
}
