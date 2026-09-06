#pragma once

#include <optional>
#include <string_view>

namespace ftlpu::compiler::target {

enum class IcuCompressionMode {
    None,
    Control,
    Macro,
};

inline std::optional<IcuCompressionMode> parse_icu_compression_mode(
    std::string_view value)
{
    if (value == "none") return IcuCompressionMode::None;
    if (value == "control") return IcuCompressionMode::Control;
    if (value == "macro") return IcuCompressionMode::Macro;
    return std::nullopt;
}

inline std::string_view icu_compression_mode_name(IcuCompressionMode mode)
{
    switch (mode) {
    case IcuCompressionMode::None: return "none";
    case IcuCompressionMode::Control: return "control";
    case IcuCompressionMode::Macro: return "macro";
    }
    return "macro";
}

} // namespace ftlpu::compiler::target
