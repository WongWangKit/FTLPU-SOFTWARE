#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/program/program_image.hpp"

int main()
{
    static_assert(ftlpu::hw::kEncodedInstructionPacketBytes == 16);
    ftlpu::ProgramImage image;
    return image.empty() ? 0 : 1;
}
