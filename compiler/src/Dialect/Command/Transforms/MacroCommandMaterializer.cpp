#include "ftlpu/compiler/Dialect/Command/Transforms/macro_command_materializer.hpp"
#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"
#include "ftlpu/icu/active_macro_context.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
namespace ftlpu::compiler::command {
using namespace software::runtime;
namespace {
constexpr auto kInstruction =
    static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction);

std::size_t checked_add(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("Macro coarse-domain cycle overflow");
    return a + b;
}

std::size_t checked_multiply(std::size_t a, std::size_t b)
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("Macro coarse-domain count overflow");
    return a * b;
}

std::int64_t checked_operand_add(std::int64_t a, std::int64_t b)
{
    if ((b > 0 && a > std::numeric_limits<std::int64_t>::max() - b)
        || (b < 0 && a < std::numeric_limits<std::int64_t>::min() - b))
        throw std::overflow_error("Macro coarse-domain operand overflow");
    return a + b;
}

std::int64_t checked_operand_multiply(std::int64_t a, std::size_t b)
{
    if (b > static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("Macro coarse-domain operand count overflow");
    const auto n = static_cast<std::int64_t>(b);
    if ((a > 0 && n > std::numeric_limits<std::int64_t>::max() / a)
        || (a < 0 && a != -1
            && n > std::numeric_limits<std::int64_t>::min() / a))
        throw std::overflow_error("Macro coarse-domain operand multiply overflow");
    return a * n;
}

InstructionKind instruction_kind(QueueKind kind)
{
    switch (kind) {
    case QueueKind::Mem: return InstructionKind::Mem;
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute: return InstructionKind::Mxm;
    case QueueKind::MxmDequant: return InstructionKind::MxmDequant;
    default:
        throw std::invalid_argument("Macro coarse lowering requires a MEM/MXM queue");
    }
}

std::uint64_t native_bits(const QueueCommand& command)
{
    return static_cast<std::uint64_t>(command.words[0])
        | (static_cast<std::uint64_t>(command.words[1]) << 32);
}

QueueCommand native_command(const MemInstruction& instruction)
{
    const auto bits = isa::encode_mem_instruction(instruction);
    return {kInstruction, InstructionKind::Mem, 2,
        {static_cast<std::uint32_t>(bits),
            static_cast<std::uint32_t>(bits >> 32), 0, 0}};
}

QueueCommand native_command(const MxmControlInstruction& instruction)
{
    const auto bits = isa::encode_mxm_instruction(instruction);
    return {kInstruction, InstructionKind::Mxm, 2,
        {static_cast<std::uint32_t>(bits),
            static_cast<std::uint32_t>(bits >> 32), 0, 0}};
}

QueueCommand native_command(const MxmDequantInstruction& instruction)
{
    return {kInstruction, InstructionKind::MxmDequant, 1,
        {static_cast<std::uint32_t>(
             isa::encode_mxm_dequant_instruction(instruction)), 0, 0, 0}};
}

QueueCommand induce(const QueueCommand& base, IcuInductionTarget target,
    std::int64_t delta)
{
    switch (base.instruction_kind) {
    case InstructionKind::Mem:
        return native_command(::ftlpu::detail::apply_macro_induction(
            isa::decode_mem_instruction(native_bits(base)), target, delta));
    case InstructionKind::Mxm:
        return native_command(::ftlpu::detail::apply_macro_induction(
            isa::decode_mxm_instruction(native_bits(base)), target, delta));
    case InstructionKind::Sxm:
    case InstructionKind::MxmDequant:
        if (target == IcuInductionTarget::None && delta == 0) return base;
        break;
    default: break;
    }
    throw std::invalid_argument("invalid Macro coarse-domain induction target");
}

void validate_context(const QueueCommand& native,
    const IcuMacroSchedule& schedule)
{
    if (native.instruction_kind == InstructionKind::Mem) {
        static_cast<void>(IcuActiveMacroContextCodec256<MemInstruction>::
            pack_initial(schedule, isa::decode_mem_instruction(native_bits(native))));
    } else if (native.instruction_kind == InstructionKind::Mxm) {
        static_cast<void>(IcuActiveMacroContextCodec256<MxmControlInstruction>::
            pack_initial(schedule, isa::decode_mxm_instruction(native_bits(native))));
    } else if(native.instruction_kind==InstructionKind::Sxm) {
        const auto ins=decode_sxm_command(native);
        if(ins.opcode==SxmOpcode::Transpose) (void)IcuSxmActiveMacroContextCodec<true>::pack_initial(schedule,ins);
        else (void)IcuSxmActiveMacroContextCodec<false>::pack_initial(schedule,ins);
    } else {
        static_cast<void>(IcuActiveMacroContextCodec256<MxmDequantInstruction>::
            pack_initial(schedule, isa::decode_mxm_dequant_instruction(native.words[0])));
    }
}

struct Axis {
    std::size_t count;
    std::size_t cycle_stride;
    std::int64_t operand_stride;
};

struct Domain {
    std::size_t start;
    QueueCommand native;
    IcuInductionTarget target;
    std::vector<Axis> axes;
};

struct Box {
    DomainCoordinate origin {};
    std::array<std::size_t, 3> counts {1, 1, 1};
    std::array<std::size_t, 3> steps {1, 1, 1};
};

std::size_t packet_end(const AffineLoopDomain& loop, std::size_t cursor)
{
    auto end = checked_add(cursor, loop.wait_cycle);
    for (std::size_t axis = 0; axis < 3; ++axis)
        end = checked_add(end,
            checked_multiply(loop.counts[axis] - 1, loop.cycle_strides[axis]));
    return checked_add(end, 1);
}

std::vector<Box> split_boxes(const AffineLoopDomain& loop, MxmBufferSelection bufferMode,
    std::size_t terminalDimension = AffineLoopDomain::kDimensions)
{
    const auto bufferAxis=bufferMode==MxmBufferSelection::Fixed?3:static_cast<std::size_t>(bufferMode)-1;
    const auto cut=std::min(bufferAxis,terminalDimension);
    Box base; base.counts=loop.counts;
    if(cut>=3) return {base};
    std::vector<Box> boxes;
    for(std::size_t axis=cut;axis<3;++axis) base.counts[axis]=1;
    const auto visit=[&](auto&& self,int axis,Box box)->void {
        if(axis<static_cast<int>(cut)) { boxes.push_back(box);return; }
        for(std::size_t i=0;i<loop.counts[axis];++i) {
            box.origin.index[axis]=i;self(self,axis-1,box);
        }
    };
    visit(visit,2,base);
    return boxes;
}

template <typename Instruction, typename Native>
Domain box_domain(const Instruction& instruction, const Box& box,
    std::size_t cursor, const std::array<std::int64_t, 3>& operandStrides,
    IcuInductionTarget target, Native expand)
{
    Domain domain {checked_add(checked_add(cursor,instruction.loop.start_cycle),
                       ::ftlpu::compiler::detail::affine_issue_offset(instruction.loop, box.origin)),
        native_command(expand(instruction, box.origin)), target, {}};
    for (std::size_t axis = 0; axis < 3; ++axis)
        domain.axes.push_back({box.counts[axis],
            checked_multiply(instruction.loop.cycle_strides[axis], box.steps[axis]),
            checked_operand_multiply(operandStrides[axis], box.steps[axis])});
    return domain;
}

void flatten_axes(Domain& domain)
{
    std::erase_if(domain.axes, [](const Axis& axis) { return axis.count == 1; });
    std::sort(domain.axes.begin(), domain.axes.end(),
        [](const Axis& a, const Axis& b) { return a.cycle_stride < b.cycle_stride; });
    // Adjacent affine axes can be flattened algebraically. Keep counts within
    // the physical Active Context field instead of creating oversized loops.
    for (std::size_t i = 1; i < domain.axes.size();) {
        auto& inner = domain.axes[i - 1];
        const auto outer = domain.axes[i];
        if (inner.count <= 65536 / outer.count
            && outer.cycle_stride == checked_multiply(inner.count, inner.cycle_stride)
            && outer.operand_stride
                == checked_operand_multiply(inner.operand_stride, inner.count)) {
            inner.count *= outer.count;
            domain.axes.erase(domain.axes.begin() + i);
        } else {
            ++i;
        }
    }
}

struct Pending {
    std::size_t source;
    QueueCommand command;
};

IcuMacroSchedule make_schedule(const Domain& domain,
    const std::vector<std::size_t>& keep)
{
    IcuMacroSchedule schedule;
    schedule.start_cycle = domain.start;
    schedule.induction_target = domain.target;
    if (!keep.empty()) {
        const auto& axis = domain.axes[keep[0]];
        schedule.inner_count = axis.count;
        schedule.inner_interval = axis.cycle_stride;
        schedule.inner_stride = axis.operand_stride;
    }
    if (keep.size() > 1) {
        const auto& axis = domain.axes[keep[1]];
        schedule.outer_count = axis.count;
        schedule.outer_interval = axis.cycle_stride;
        schedule.outer_stride = axis.operand_stride;
    }
    return schedule;
}

void emit_domain(Domain domain, std::size_t source,
    std::size_t budget, std::vector<Pending>& output)
{
    flatten_axes(domain);
    struct Plan { std::vector<std::size_t> keep; std::size_t entries; };
    std::vector<Plan> plans;
    const auto addPlan = [&](std::vector<std::size_t> keep) {
        for(std::size_t i=0;i<keep.size();++i) if(keep[i]!=i) return;
        auto entries = std::size_t {1};
        for (std::size_t axis = 0; axis < domain.axes.size(); ++axis) {
            if (std::find(keep.begin(), keep.end(), axis) == keep.end()) {
                if (domain.axes[axis].count > budget / entries) return;
                entries *= domain.axes[axis].count;
            }
        }
        plans.push_back({std::move(keep), entries});
    };
    for (std::size_t a = 0; a < domain.axes.size(); ++a)
        for (std::size_t b = a + 1; b < domain.axes.size(); ++b) addPlan({a, b});
    for (std::size_t a = 0; a < domain.axes.size(); ++a) addPlan({a});
    addPlan({});
    std::stable_sort(plans.begin(), plans.end(),
        [](const Plan& a, const Plan& b) { return a.entries < b.entries; });

    const Plan* selected = nullptr;
    for (const auto& plan : plans) {
        try {
            validate_context(domain.native, make_schedule(domain, plan.keep));
            selected = &plan;
            break;
        } catch (const std::logic_error&) {
            // Another axis pair may fit the real row-transition field even
            // when the largest-work pair does not. No work points are expanded.
        }
    }
    if (!selected)
        throw std::invalid_argument(
            "Macro coarse domain exceeds the descriptor budget or Active Context fields");
    std::vector<std::size_t> drop;
    for (std::size_t axis = 0; axis < domain.axes.size(); ++axis)
        if (std::find(selected->keep.begin(), selected->keep.end(), axis)
            == selected->keep.end()) drop.push_back(axis);
    auto schedule = make_schedule(domain, selected->keep);
    const auto emit = [&](auto&& self, std::size_t position,
                          std::size_t cycle, std::int64_t operand) -> void {
        if (position == drop.size()) {
            schedule.start_cycle = cycle;
            auto native = induce(domain.native, domain.target, operand);
            validate_context(native, schedule);
            output.push_back({source, encode_macro_schedule_command(std::move(native), schedule)});
            return;
        }
        const auto& axis = domain.axes[drop[position]];
        for (std::size_t index = 0; index < axis.count; ++index)
            self(self, position + 1,
                checked_add(cycle, checked_multiply(index, axis.cycle_stride)),
                checked_operand_add(operand, checked_operand_multiply(axis.operand_stride, index)));
    };
    emit(emit, 0, domain.start, 0);
}

mlir::ArrayAttr wordArray(mlir::OpBuilder& builder, const auto& values)
{
    llvm::SmallVector<mlir::Attribute> result;
    for(auto value:values) result.push_back(builder.getI64IntegerAttr(value));
    return builder.getArrayAttr(result);
}

void emitMacroOps(mlir::OpBuilder& builder,mlir::Location location,QueueKind kind,
    std::size_t queue,std::vector<Pending> commands,int64_t addressBinding=-1,
    llvm::StringRef access={},int64_t scaleBinding=-1)
{
    std::stable_sort(commands.begin(),commands.end(),[](const auto& a,const auto& b) {
        return decode_macro_schedule_command(a.command).start_cycle<decode_macro_schedule_command(b.command).start_cycle;
    });
    std::optional<std::size_t> previousEnd;
    for(const auto& item:commands) {
        const auto& command=item.command;
        const auto schedule=decode_macro_schedule_command(command);
        if(previousEnd && schedule.start_cycle<=*previousEnd)
            throw std::invalid_argument("affine lowering would interleave separate Macros on one ICU");
        previousEnd=schedule.start_cycle+(schedule.inner_count-1)*schedule.inner_interval
            +(schedule.outer_count-1)*schedule.outer_interval;
        mlir::OperationState state(location,MacroOp::getOperationName());
        state.addAttributes({builder.getNamedAttr("cycle",builder.getI64IntegerAttr(schedule.start_cycle)),
            builder.getNamedAttr("queue",builder.getI64IntegerAttr(queue)),
            builder.getNamedAttr("kind",builder.getI64IntegerAttr(static_cast<unsigned>(kind))),
            builder.getNamedAttr("instruction_kind",builder.getI64IntegerAttr(static_cast<unsigned>(command.instruction_kind))),
            builder.getNamedAttr("words",wordArray(builder,std::span(command.words.data(),command.word_count))),
            builder.getNamedAttr("payload",wordArray(builder,command.extension_words))});
        if(addressBinding>=0) {
            state.addAttribute("address_binding",builder.getI64IntegerAttr(addressBinding));
            if(!access.empty()) state.addAttribute("address_binding_access",builder.getStringAttr(access));
        }
        if(scaleBinding>=0) state.addAttribute("scale_binding",builder.getI64IntegerAttr(scaleBinding));
        builder.create(state);
    }
}

template <typename Fn> mlir::LogicalResult checked(std::string* error,Fn fn)
{
    if(error) error->clear();
    try {fn();return mlir::success();}
    catch(const std::exception& e){if(error)*error=e.what();return mlir::failure();}
}
} // namespace

std::vector<QueueCommand> lowerMacroDomain(std::size_t start,QueueCommand native,IcuInductionTarget target,
    std::array<std::size_t,3> counts,std::array<std::size_t,3> intervals,std::array<std::int64_t,3> strides)
{
    Domain domain{start,std::move(native),target,{}};
    for(std::size_t i=0;i<3;++i) domain.axes.push_back({counts[i],intervals[i],strides[i]});
    std::vector<Pending> output;emit_domain(std::move(domain),0,65536,output);
    std::vector<QueueCommand> commands;
    for(auto& p:output) commands.push_back(std::move(p.command));
    return commands;
}

QueueCommand readMacroCommand(mlir::Operation* op)
{
    QueueCommand command;
    command.command=static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Extended);
    command.instruction_kind=static_cast<InstructionKind>(op->getAttrOfType<mlir::IntegerAttr>("instruction_kind").getInt());
    auto words=op->getAttrOfType<mlir::ArrayAttr>("words");
    if(words.size()>command.words.size()) throw std::invalid_argument("Macro native FU payload is too large");
    command.word_count=static_cast<std::uint16_t>(words.size());
    for(std::size_t i=0;i<words.size();++i) command.words[i]=llvm::cast<mlir::IntegerAttr>(words[i]).getInt();
    for(auto word:op->getAttrOfType<mlir::ArrayAttr>("payload")) command.extension_words.push_back(llvm::cast<mlir::IntegerAttr>(word).getInt());
    const auto schedule=decode_macro_schedule_command(command);
    if(schedule.start_cycle!=static_cast<std::size_t>(op->getAttrOfType<mlir::IntegerAttr>("cycle").getInt()))
        throw std::invalid_argument("Macro metadata differs from its issue schedule");
    auto native=command;native.command=kInstruction;
    native.extension_words.erase(native.extension_words.begin(),native.extension_words.begin()+9);
    validate_context(native,schedule);
    return command;
}

mlir::LogicalResult materializeMemMacroCommand(mlir::OpBuilder& builder,mlir::Location location,
    std::size_t queue,const MemAddressDomain& instruction,int64_t binding,llvm::StringRef access,std::string* error)
{
    return checked(error,[&] {
        auto geometry=instruction;geometry.loop.start_cycle=0;
        ::ftlpu::compiler::detail::validate_mem_domain(geometry);
        const auto& loop=instruction.loop;const auto& address=instruction.address;
        std::vector<Pending> output;
        const auto emit=[&](Domain domain){emit_domain(std::move(domain),0,65536,output);};
        const auto group=address.outer_group_size;
        if(group==1 || address.outer_group_stride==checked_operand_multiply(address.outer_inner_stride,group)) {
            Box box;box.counts=loop.counts;
            emit(box_domain(instruction,box,0,{address.inner_stride,address.middle_stride,
                group==1?address.outer_group_stride:address.outer_inner_stride},IcuInductionTarget::MemAddress,::ftlpu::compiler::detail::mem_domain_instruction));
        } else {
            // Each complete outer group is contiguous in time. Preserve that
            // ordering rather than creating interleaving parity descriptors.
            for(std::size_t base=0;base<loop.counts[2];base+=group) {
                const DomainCoordinate origin{{0,0,base}};
                emit({loop.start_cycle+::ftlpu::compiler::detail::affine_issue_offset(loop,origin),native_command(::ftlpu::compiler::detail::mem_domain_instruction(instruction,origin)),
                    IcuInductionTarget::MemAddress,{{loop.counts[0],loop.cycle_strides[0],address.inner_stride},
                    {loop.counts[1],loop.cycle_strides[1],address.middle_stride},
                    {std::min(group,loop.counts[2]-base),loop.cycle_strides[2],address.outer_inner_stride}}});
            }
        }
        emitMacroOps(builder,location,QueueKind::Mem,queue,std::move(output),binding,access);
    });
}

mlir::LogicalResult materializeMxmLoadMacroCommand(mlir::OpBuilder& builder,mlir::Location location,
    std::size_t queue,const MxmLoadDomain& instruction,std::string* error)
{
    return checked(error,[&]{
        std::vector<Pending> output;
        const auto target=instruction.opcode==MxmLoadDomainOperation::LoadActivation?IcuInductionTarget::None:IcuInductionTarget::MxmWeightColumn;
        for(const auto& box:split_boxes(instruction.loop,instruction.weight_buffer_mode))
            emit_domain(box_domain(instruction,box,0,instruction.weight_column_strides,target,::ftlpu::compiler::detail::mxm_load_domain_instruction),0,65536,output);
        emitMacroOps(builder,location,QueueKind::MxmLoad,queue,std::move(output));
    });
}
mlir::LogicalResult materializeMxmDequantMacroCommand(mlir::OpBuilder& builder,mlir::Location location,
    std::size_t queue,const MxmDequantDomain& instruction,int64_t binding,std::string* error)
{
    return checked(error,[&]{
        std::vector<Pending> output;Box box;box.counts=instruction.loop.counts;
        emit_domain(box_domain(instruction,box,0,{0,0,0},IcuInductionTarget::None,
            [](const auto& v,const auto&){return v.instruction;}),0,65536,output);
        emitMacroOps(builder,location,QueueKind::MxmDequant,queue,std::move(output),-1,{},binding);
    });
}
mlir::LogicalResult materializeMxmComputeMacroCommand(mlir::OpBuilder& builder,mlir::Location location,
    std::size_t queue,const MxmComputeDomain& instruction,std::string* error)
{
    return checked(error,[&]{
        std::vector<Pending> output;
        const auto terminal=instruction.opcode==MxmComputeDomainOperation::Compute?instruction.terminal_dimension:3;
        for(const auto& box:split_boxes(instruction.loop,instruction.weight_buffer_mode,terminal))
            emit_domain(box_domain(instruction,box,0,instruction.accumulator_address_strides,IcuInductionTarget::MxmAccumulatorAddress,
                ::ftlpu::compiler::detail::mxm_compute_domain_instruction),0,65536,output);
        emitMacroOps(builder,location,QueueKind::MxmCompute,queue,std::move(output));
    });
}

mlir::LogicalResult materializeSxmMacroCommand(mlir::OpBuilder& builder,mlir::Location location,
    bool transpose,std::size_t queue,const SxmLaunchDomain& instruction,std::string* error)
{
    return checked(error,[&]{
        std::vector<Pending> output;
        const auto& loop=instruction.loop;
        if(instruction.permute_map_stride==0) {
            emit_domain({loop.start_cycle+loop.wait_cycle,encode_sxm_command(instruction.instruction),IcuInductionTarget::None,
                {{loop.counts[0],loop.cycle_strides[0],0},{loop.counts[1],loop.cycle_strides[1],0}}},0,65536,output);
        } else {
            // The mapping changes each launch; these are distinct FU templates,
            // emitted directly from the finite periodic mapping domain.
            for(std::size_t outer=0;outer<loop.counts[1];++outer) for(std::size_t inner=0;inner<loop.counts[0];++inner) {
                auto fu=instruction.instruction;
                for(auto& lane:fu.permute_map) lane=(lane+(outer*loop.counts[0]+inner)*instruction.permute_map_stride)%SxmInstruction::kTotalLanes;
                emit_domain({loop.start_cycle+::ftlpu::compiler::detail::affine_issue_offset(loop,DomainCoordinate{{inner,outer,0}}),encode_sxm_command(fu),IcuInductionTarget::None,{}},0,65536,output);
            }
        }
        emitMacroOps(builder,location,transpose?QueueKind::SxmTranspose:QueueKind::SxmPermute,queue,std::move(output));
    });
}

mlir::LogicalResult materializeMemTransferPair(mlir::OpBuilder& builder,mlir::Location location,
    std::size_t cycle,std::size_t queue,const MemTransferPairDomain& instruction,std::string* error)
{
    return checked(error,[&]{
        std::vector<Pending> output;
        for(std::size_t outer=0;outer<instruction.counts[1];++outer)
            for(std::size_t inner=0;inner<instruction.counts[0];++inner)
                for(bool read:{false,true}) {
                    const auto& strides=read?instruction.read_cycle_strides:instruction.write_cycle_strides;
                    const auto offset=instruction.start_wait+(read?instruction.read_start_offset:0)
                        +inner*strides[0]+outer*strides[1];
                    const auto fu=::ftlpu::compiler::detail::mem_transfer_pair_instruction(instruction,inner,outer,read);
                    emit_domain({cycle+offset,native_command(fu),IcuInductionTarget::MemAddress,{}},0,65536,output);
                }
        emitMacroOps(builder,location,QueueKind::Mem,queue,std::move(output));
    });
}

mlir::LogicalResult materializeFfnUpMacroCommands(mlir::OpBuilder& builder,mlir::Location location,
    const schedule::FfnUp3DLoweringResult& lowering,std::string* error)
{
    for(const auto& c:lowering.mem_commands) {auto i=c.instruction;i.loop.start_cycle=c.cycle;
        if(mlir::failed(materializeMemMacroCommand(builder,location,c.queue,i,-1,{},error))) return mlir::failure();}
    for(const auto& c:lowering.mxm_load_commands) {auto i=c.instruction;i.loop.start_cycle=c.cycle;
        if(mlir::failed(materializeMxmLoadMacroCommand(builder,location,c.queue,i,error))) return mlir::failure();}
    for(const auto& c:lowering.mxm_dequant_commands) {auto i=c.instruction;i.loop.start_cycle=c.cycle;
        if(mlir::failed(materializeMxmDequantMacroCommand(builder,location,c.queue,i,-1,error))) return mlir::failure();}
    for(const auto& c:lowering.mxm_compute_commands) {auto i=c.instruction;i.loop.start_cycle=c.cycle;
        if(mlir::failed(materializeMxmComputeMacroCommand(builder,location,c.queue,i,error))) return mlir::failure();}
    return mlir::success();
}
} // namespace ftlpu::compiler::command
