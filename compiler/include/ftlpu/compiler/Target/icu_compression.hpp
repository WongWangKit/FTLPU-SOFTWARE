#pragma once

#include <optional>
#include <string_view>

namespace ftlpu::compiler::target {

enum class IcuCompressionMode {
    None,
    Control,
    Macro,
    RawMacro,
};

inline std::optional<IcuCompressionMode> parse_icu_compression_mode(
    std::string_view value)
{
    if (value == "none") return IcuCompressionMode::None;
    if (value == "control" || value == "repeat") return IcuCompressionMode::Control;
    if (value == "macro") return IcuCompressionMode::Macro;
    if (value == "raw-macro") return IcuCompressionMode::RawMacro;
    return std::nullopt;
}

inline std::string_view icu_compression_mode_name(IcuCompressionMode mode)
{
    switch (mode) {
    case IcuCompressionMode::None: return "none";
    case IcuCompressionMode::Control: return "control";
    case IcuCompressionMode::Macro: return "macro";
    case IcuCompressionMode::RawMacro: return "raw-macro";
    }
    return "macro";
}

} // namespace ftlpu::compiler::target
