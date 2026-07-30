#include "ftlpu/software/runtime/cmodel_device_backend.hpp"

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/dma/dma.hpp"
#include "ftlpu/program/program.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ftlpu::software::runtime {
namespace {

constexpr const char* active_executable_target_name() noexcept
{
#ifdef FTLPU_TRANSFORMER_EVAL_CONFIG
    return "lpu_cmodel_transformer_eval_v1";
#else
    return "lpu_cmodel_groqlike_v1";
#endif
}

ExecutableTargetParameters active_executable_parameters()
{
    return {
        static_cast<std::int64_t>(hw::kHemispheres),
        static_cast<std::int64_t>(hw::kMemSliceColumns),
        static_cast<std::int64_t>(hw::kSramBanksPerTileBlock),
        static_cast<std::int64_t>(hw::kSramWordsPerBank),
        static_cast<std::int64_t>(hw::kSramWordBytes),
        static_cast<std::int64_t>(hw::kTileRows),
        static_cast<std::int64_t>(hw::kLanesPerTile),
        static_cast<std::int64_t>(hw::kPhysicalVectorBytes),
        static_cast<std::int64_t>(hw::kStreamsPerDirection),
        static_cast<std::int64_t>(hw::kMemReadBytesPerCycle),
        static_cast<std::int64_t>(hw::kMemWriteBytesPerCycle),
        static_cast<std::int64_t>(hw::kMxmCount),
        static_cast<std::int64_t>(hw::kMxmRows),
        static_cast<std::int64_t>(hw::kMxmColumns),
        static_cast<std::int64_t>(hw::kMxmLoadStreamsPerCycle),
        static_cast<std::int64_t>(hw::kMxmLoadBytesPerCycle),
        static_cast<std::int64_t>(hw::kVxmAluCount),
    };
}

std::uint64_t active_backend_abi()
{
    TargetAbiHasher hash;
    hash.add(1); // CModelDeviceBackend contract version.
    hash.add(static_cast<std::int64_t>(
        executable_target_abi(active_executable_parameters())));
    return hash.value();
}

} // namespace

TargetDescription make_cmodel_target_description()
{
    const auto parameters = active_executable_parameters();
    return TargetDescription {
        "ftlpu.cmodel",
        active_backend_abi(),
        {
            active_executable_target_name(),
            executable_target_abi(parameters),
        },
        hw::kTileRows,
        hw::kLanesPerTile,
        hw::kPhysicalVectorBytes,
        hw::kHemispheres,
        hw::kMemSliceColumns,
        hw::kSramBanksPerTileBlock,
        hw::kSramWordsPerBank,
        hw::kSramWordBytes,
        MemGlobalAddress24::kBits,
        hw::kEncodedInstructionPacketBytes,
        hw::kIcuFetchBufferBytes,
        hw::kIcuFetchPackets,
        hw::kMxmCount,
        hw::kMxmRows,
        hw::kMxmColumns,
        hw::kMxmK,
        hw::kMxmWeightBytesPerValue,
        hw::kMxmActivationBytesPerValue,
        hw::kMxmLoadStreamsPerCycle,
        hw::kMxmLoadBytesPerCycle,
        hw::kVxmAluCount,
        hw::kHemispheres,
        hw::kStreamsPerDirection,
        hw::kMemReadBytesPerCycle,
        hw::kMemWriteBytesPerCycle,
    };
}

struct CModelDeviceBackend::Impl {
    Impl()
        : system(std::make_unique<TspSliceSystem>())
        , dma(host, memory)
        , target(make_cmodel_target_description())
    {
        for (std::size_t hemisphere = 0;
             hemisphere < hw::kHemispheres;
             ++hemisphere) {
            memory.bind_hemisphere(
                hemisphere,
                system
                    ->mem(static_cast<Hemisphere>(hemisphere))
                    .memory_model());
        }
    }

    void execute_dma(
        DmaDescriptor descriptor,
        std::size_t bytes)
    {
        if (!dma.enqueue(std::move(descriptor)).valid()) {
            throw std::logic_error(
                "CModel DMA returned an invalid transfer ID");
        }
        while (!dma.idle()) {
            if (!dma.tick()) {
                throw std::logic_error(
                    "CModel DMA stalled before completing a transfer");
            }
        }
        if (!dma.completion_ready()
            || !dma.pop_completion().id.valid()) {
            throw std::logic_error(
                "CModel DMA did not report transfer completion");
        }
        ++statistics.dma_transfer_count;
        statistics.system_cycles += 0;
        (void)bytes;
    }

    void execute_program_dma(
        HostBufferId buffer,
        const ProgramSramLayout& layout)
    {
        const auto descriptors = layout.make_dma_descriptors(buffer);
        for (const auto& descriptor : descriptors) {
            execute_dma(
                descriptor,
                descriptor.vector_count * target.vector_bytes);
            statistics.dma_upload_bytes +=
                descriptor.vector_count * target.vector_bytes;
        }
    }

    std::unique_ptr<TspSliceSystem> system;
    GlobalMemoryAddressSpace memory{};
    HostMemorySpace host{};
    DmaEngine dma;
    TargetDescription target{};
    DeviceBackendState state{DeviceBackendState::Empty};
    DeviceBackendStatistics statistics{};
    std::optional<program::AutonomousProgram> launched{};
    std::size_t execution_cycles{0};
};

CModelDeviceBackend::CModelDeviceBackend()
    : impl_(std::make_unique<Impl>())
{
}

CModelDeviceBackend::~CModelDeviceBackend() = default;
CModelDeviceBackend::CModelDeviceBackend(CModelDeviceBackend&&) noexcept = default;
CModelDeviceBackend& CModelDeviceBackend::operator=(
    CModelDeviceBackend&&) noexcept = default;

const TargetDescription& CModelDeviceBackend::target() const noexcept
{
    return impl_->target;
}

DeviceBackendState CModelDeviceBackend::state() const noexcept
{
    return impl_->state;
}

DeviceBackendStatistics CModelDeviceBackend::statistics() const noexcept
{
    return impl_->statistics;
}

void CModelDeviceBackend::reset_execution_state()
{
    if (impl_->state == DeviceBackendState::Running) {
        throw std::logic_error(
            "cannot reset CModelDeviceBackend while it is running");
    }
    impl_->system->icu().reset();
    for (std::size_t hemisphere = 0;
         hemisphere < hw::kHemispheres;
         ++hemisphere) {
        auto& mem =
            impl_->system->mem(static_cast<Hemisphere>(hemisphere));
        mem.memory_model().reset_execution_state();
        mem.stream_fabric().reset();
        impl_->system
            ->sxm(static_cast<Hemisphere>(hemisphere))
            .reset();
    }
    impl_->system->vxm().reset();
    for (std::size_t mxm = 0; mxm < hw::kMxmCount; ++mxm) {
        impl_->system->mxm_unit(mxm).reset();
    }
    impl_->dma.reset();
    impl_->launched.reset();
    impl_->execution_cycles = 0;
    impl_->state = DeviceBackendState::Empty;
}

void CModelDeviceBackend::load(const DeviceProgram& device_program)
{
    if (impl_->state != DeviceBackendState::Empty) {
        throw std::logic_error(
            "CModelDeviceBackend load requires an empty execution state");
    }
    if (device_program.execution_cycles == 0) {
        throw std::invalid_argument(
            "DeviceProgram execution_cycles must be non-zero");
    }
    auto launched =
        program::AutonomousProgramBuilder::Build(device_program.image);
    const auto buffer =
        impl_->host.register_buffer(launched.layout.host_bytes());
    try {
        impl_->execute_program_dma(buffer, launched.layout);
        impl_->host.unregister_buffer(buffer);
    } catch (...) {
        impl_->host.unregister_buffer(buffer);
        throw;
    }
    impl_->execution_cycles = device_program.execution_cycles;
    impl_->launched = std::move(launched);
    impl_->state = DeviceBackendState::Loaded;
}

void CModelDeviceBackend::upload(
    MemGlobalAddress24 destination,
    std::span<const std::uint8_t> bytes,
    DmaPurpose purpose)
{
    if (impl_->state == DeviceBackendState::Running) {
        throw std::logic_error(
            "cannot upload while CModelDeviceBackend is running");
    }
    if (bytes.empty() || bytes.size() % impl_->target.vector_bytes != 0) {
        throw std::invalid_argument(
            "CModelDeviceBackend upload requires whole physical vectors");
    }
    if (!destination.slice_byte_address().word_aligned()) {
        throw std::invalid_argument(
            "CModelDeviceBackend upload address is not vector aligned");
    }
    auto host_bytes =
        std::vector<std::uint8_t>(bytes.begin(), bytes.end());
    const auto buffer =
        impl_->host.register_buffer(std::move(host_bytes));
    try {
        impl_->execute_dma(
            DmaDescriptor {
                DmaDirection::HostToMemory,
                purpose,
                buffer,
                0,
                destination,
                bytes.size() / impl_->target.vector_bytes,
            },
            bytes.size());
        impl_->statistics.dma_upload_bytes += bytes.size();
        impl_->host.unregister_buffer(buffer);
    } catch (...) {
        impl_->host.unregister_buffer(buffer);
        throw;
    }
}

void CModelDeviceBackend::launch()
{
    if (impl_->state != DeviceBackendState::Loaded
        || !impl_->launched.has_value()) {
        throw std::logic_error(
            "CModelDeviceBackend launch requires a loaded program");
    }
    load_bootstrap_preamble(
        impl_->system->icu(), impl_->launched->preamble);
    impl_->state = DeviceBackendState::Running;
}

void CModelDeviceBackend::wait(std::ostream* log)
{
    if (impl_->state != DeviceBackendState::Running
        || !impl_->launched.has_value()) {
        throw std::logic_error(
            "CModelDeviceBackend wait requires a running program");
    }
    const auto cycles =
        impl_->launched->schedule_epoch_cycle
        + impl_->execution_cycles;
    for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
        if (log != nullptr) {
            impl_->system->tick(*log);
        } else {
            impl_->system->tick({});
        }
    }
    impl_->statistics.system_cycles += cycles;
    impl_->state = DeviceBackendState::Complete;
}

std::vector<std::uint8_t> CModelDeviceBackend::download(
    MemGlobalAddress24 source,
    std::size_t byte_count,
    DmaPurpose purpose)
{
    if (impl_->state == DeviceBackendState::Running) {
        throw std::logic_error(
            "cannot download while CModelDeviceBackend is running");
    }
    if (byte_count == 0
        || byte_count % impl_->target.vector_bytes != 0) {
        throw std::invalid_argument(
            "CModelDeviceBackend download requires whole physical vectors");
    }
    if (!source.slice_byte_address().word_aligned()) {
        throw std::invalid_argument(
            "CModelDeviceBackend download address is not vector aligned");
    }
    const auto buffer = impl_->host.allocate_buffer(byte_count);
    try {
        impl_->execute_dma(
            DmaDescriptor {
                DmaDirection::MemoryToHost,
                purpose,
                buffer,
                0,
                source,
                byte_count / impl_->target.vector_bytes,
            },
            byte_count);
        auto result = impl_->host.buffer(buffer);
        impl_->statistics.dma_download_bytes += byte_count;
        impl_->host.unregister_buffer(buffer);
        return result;
    } catch (...) {
        impl_->host.unregister_buffer(buffer);
        throw;
    }
}

TspSliceSystem& CModelDeviceBackend::system() noexcept
{
    return *impl_->system;
}

const TspSliceSystem& CModelDeviceBackend::system() const noexcept
{
    return *impl_->system;
}

} // namespace ftlpu::software::runtime
