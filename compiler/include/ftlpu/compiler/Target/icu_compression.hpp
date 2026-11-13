#pragma once
#include <optional>
#include <string_view>
namespace ftlpu::compiler::target {
enum class IcuCompressionMode { RawMacro };
inline std::optional<IcuCompressionMode> parse_icu_compression_mode(std::string_view value) {
    if(value=="raw-macro" || value=="macro") return IcuCompressionMode::RawMacro;
    return std::nullopt;
}
inline std::string_view icu_compression_mode_name(IcuCompressionMode) { return "raw-macro"; }
} // namespace ftlpu::compiler::target
