#pragma once
#include "ftlpu/compiler/Target/vector32_triangular_solve.hpp"
#include "ftlpu/icu/fu_3d_instruction.hpp"
#include "ftlpu/sxm/instruction.hpp"

namespace ftlpu::compiler {
struct Vector32SolveMemIssue {
    std::size_t side, slice, cycle;
    MemIcuInstruction instruction;
};
struct Vector32SolveVxmIssue {
    std::size_t queue, cycle;
    VxmCompactInstruction instruction;
};
struct Vector32SolveSxmIssue {
    std::size_t side, cycle, count, interval;
    SxmInstruction instruction;
};
struct Vector32SolveSchedule {
    std::vector<Vector32SolveMemIssue> mem;
    std::vector<Vector32SolveVxmIssue> vxm;
    std::vector<Vector32SolveSxmIssue> sxm;
    std::size_t end_cycle = 0;
};

// Resident chunk32 solve. Two hemispheres own alternating 32-value blocks.
// Slice pairs: D=0/1, B=2/3, expanded -L=4/5, one=6/7. Each pair stores
// low/high BF16 bytes; lane coordinates stay unchanged throughout the solve.
// All operands arrive via MEM, including coefficients. No SXM is required.
// Conservative task spacing is intentional until correctness is established.
inline Vector32SolveSchedule schedule_vector32_triangular_solve(
    std::size_t width, bool packed_coefficients=false) {
    if (width % 64) throw std::invalid_argument("mirrored resident solve requires pairs of 32-value blocks");
    Vector32SolveSchedule result;
    const auto blocks_per_side = width / 64;
    std::size_t start = 32;
    if (packed_coefficients) {
        // Packed -L: pair 8/9, one 32-element row per word. Expand only
        // strict-lower entries using native transpose-broadcast + permute.
        // Destination pitch 32 makes every column's MEM domain affine.
        // Its live entries total 31 KiB; the reserved span is 64 KiB/side.
        const auto edge=hw::kMemGroups+hw::kC2cToSxmStreamRegisterColumns;
        constexpr std::size_t expansion_interval=8;
        for (std::size_t j=0; j<31; ++j) {
            const auto count=31-j, transpose=start, permute=transpose+5;
            for (std::size_t side=0; side<2; ++side) {
                auto trans=SxmInstruction::Transpose({{16},{17}},{});
                trans.input_row=SxmInstruction::kBroadcastInputRows;
                for (std::size_t s=16; s<32; ++s) trans.dst_streams.push_back({s});
                SxmInstruction::PermuteMap map{};
                for (std::size_t lane=0; lane<32; ++lane) map[lane]=(j/8)*8+lane%8;
                auto perm=SxmInstruction::Permute({}, {}, map);
                for (std::size_t s=16; s<32; ++s) perm.src_streams.push_back({s});
                for (std::size_t s=32; s<48; ++s) perm.dst_streams.push_back({s});
                perm.output_row=j%8<7?SxmInstruction::kConsumeOutputRowBase+j%8:7;
                result.sxm.push_back({side,transpose,count,expansion_interval,trans});
                result.sxm.push_back({side,permute,count,expansion_interval,perm});
                for (std::size_t byte=0; byte<2; ++byte) {
                    IcuLoop3D loop{0,{count,1,1},{expansion_interval,1,1}};
                    result.mem.push_back({side,8+byte,transpose-edge+(8+byte)/hw::kMemSlicesPerGroup,
                        MemIcuInstruction::Read3D(loop,MemIcuAddress3D::Affine(j+1,{1,0,0}),StreamId::East(16+byte))});
                    result.mem.push_back({side,4+byte,permute+edge-(4+byte)/hw::kMemSlicesPerGroup,
                        MemIcuInstruction::Write3D(loop,MemIcuAddress3D::Affine((j+1)*32+j,{32,0,0}),StreamId::West((j%8)*2+byte))});
                }
            }
            start+=count*expansion_interval+64;
        }
    }
    for (const auto& task : plan_vector32_triangular_solve(32, width)) {
        std::size_t queue_cursor[2] = {start-1,start-1};
        for (const auto& packet : task.packets())
            result.vxm.push_back({packet.queue,queue_cursor[packet.queue]++,packet.instruction});
        for (std::size_t side=0; side<2; ++side)
            for (std::size_t byte=0; byte<2; ++byte) {
                auto memory = [&](std::size_t slice, std::size_t cycle,
                                  std::size_t count, std::size_t address,
                                  std::int64_t stride, std::size_t stream, bool write=false) {
                    IcuLoop3D loop{0,{count,1,1},{1,1,1}};
                    const auto addr=MemIcuAddress3D::Affine(address,{stride,0,0});
                    // Callers express the group-0 issue time. Other groups
                    // need earlier reads so both operand bundles coincide.
                    const auto hop = slice / hw::kMemSlicesPerGroup;
                    result.mem.push_back({side,slice,write ? cycle+hop : cycle-hop,write
                        ? MemIcuInstruction::Write3D(loop,addr,StreamId::East(stream))
                        : MemIcuInstruction::Read3D(loop,addr,StreamId::West(stream))});
                };
                const auto block=task.block/2;
                if (task.row) {
                    memory(byte,start-1,task.row,block,blocks_per_side,side*16+byte);
                    memory(4+byte,start-1,task.row,packed_coefficients?task.row*32:task.row*(task.row-1)/2,1,side*16+2+byte);
                }
                memory(2+byte,start+task.row-1,1,task.row*blocks_per_side+block,0,side*16+byte);
                memory(6+byte,start+task.row-1,1,0,0,side*16+2+byte);
                // The registered multiply, accumulator and BF16 output reach
                // group-0 MEM's read boundary three cycles after input issue.
                memory(byte,start+task.row+3,1,task.row*blocks_per_side+block,0,side*8+byte,true);
            }
        start += 80;
    }
    result.end_cycle = start;
    return result;
}
} // namespace ftlpu::compiler
