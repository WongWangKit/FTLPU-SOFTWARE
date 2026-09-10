#pragma once

#include <optional>
#include <string_view>

namespace ftlpu::compiler::target {

enum class IcuCompressionMode {
    None,
    Repeat,
    Macro,
    MacroSlice,
};

inline std::optional<IcuCompressionMode> parse_icu_compression_mode(
    std::string_view value)
{
    if (value == "none") return IcuCompressionMode::None;
    if (value == "repeat" || value == "control")
        return IcuCompressionMode::Repeat;
    if (value == "macro") return IcuCompressionMode::Macro;
    if (value == "macro-slice") return IcuCompressionMode::MacroSlice;
    return std::nullopt;
}

inline std::string_view icu_compression_mode_name(IcuCompressionMode mode)
{
    switch (mode) {
    case IcuCompressionMode::None: return "none";
    case IcuCompressionMode::Repeat: return "repeat";
    case IcuCompressionMode::Macro: return "macro";
    case IcuCompressionMode::MacroSlice: return "macro-slice";
    }
    return "macro";
}

inline bool icu_compression_uses_macro(IcuCompressionMode mode)
{
    return mode == IcuCompressionMode::Macro
        || mode == IcuCompressionMode::MacroSlice;
}

inline bool icu_compression_uses_mem_slice_program(
    IcuCompressionMode mode)
{
    return mode == IcuCompressionMode::MacroSlice;
}

inline IcuCompressionMode set_mem_slice_program(
    IcuCompressionMode mode, bool enabled)
{
    if (!icu_compression_uses_macro(mode)) return mode;
    return enabled ? IcuCompressionMode::MacroSlice
                   : IcuCompressionMode::Macro;
}

} // namespace ftlpu::compiler::target
