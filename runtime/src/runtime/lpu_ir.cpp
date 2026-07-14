#include "ftlpu/software/runtime/lpu_ir.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ftlpu::software::runtime {

namespace {

std::string trim(std::string value)
{
    const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::vector<std::string> tokenize(const std::string& line)
{
    std::istringstream is(line);
    std::vector<std::string> tokens;
    std::string token;
    while (is >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::size_t parse_size(const std::string& token, const char* field)
{
    std::size_t offset = 0;
    const auto value = std::stoull(token, &offset, 0);
    if (offset != token.size()) {
        throw std::runtime_error(std::string("invalid integer field: ") + field);
    }
    return static_cast<std::size_t>(value);
}

float parse_float(const std::string& token, const char* field)
{
    std::size_t offset = 0;
    const auto value = std::stof(token, &offset);
    if (offset != token.size()) {
        throw std::runtime_error(std::string("invalid float field: ") + field);
    }
    return value;
}

VxmAluOpcode parse_vxm_opcode(const std::string& token)
{
    if (token == "pass") {
        return VxmAluOpcode::Pass;
    }
    if (token == "add") {
        return VxmAluOpcode::Add;
    }
    if (token == "sub") {
        return VxmAluOpcode::Subtract;
    }
    if (token == "mul") {
        return VxmAluOpcode::Multiply;
    }
    if (token == "div") {
        return VxmAluOpcode::Divide;
    }
    if (token == "neg") {
        return VxmAluOpcode::Negate;
    }
    if (token == "exp") {
        return VxmAluOpcode::Exp;
    }
    if (token == "cast") {
        return VxmAluOpcode::Cast;
    }
    throw std::runtime_error("unsupported VXM opcode in LPU IR: " + token);
}

VxmLaneOperand parse_operand(const std::vector<std::string>& tokens, std::size_t& index)
{
    if (index + 1 >= tokens.size()) {
        throw std::runtime_error("missing VXM operand");
    }

    const auto kind = tokens[index++];
    const auto value = tokens[index++];
    if (kind == "stream_i32") {
        return VxmLaneOperand::StreamInt32(parse_size(value, "stream_i32"));
    }
    if (kind == "alu") {
        return VxmLaneOperand::Alu(parse_size(value, "alu"));
    }
    if (kind == "imm") {
        return VxmLaneOperand::Imm(parse_float(value, "imm"));
    }
    throw std::runtime_error("unsupported VXM operand kind in LPU IR: " + kind);
}

VxmCastTarget parse_cast_target(const std::string& token)
{
    if (token == "fp32") {
        return VxmCastTarget::Float32;
    }
    if (token == "i8") {
        return VxmCastTarget::Int8;
    }
    throw std::runtime_error("unsupported VXM cast target in LPU IR: " + token);
}

void parse_line(IcuProgram& program, const std::vector<std::string>& tokens)
{
    if (tokens.empty()) {
        return;
    }

    const auto& kind = tokens[0];
    if (kind == "mem") {
        if (tokens.size() != 6) {
            throw std::runtime_error("mem syntax: mem <cycle> <column> <read|write> <address> <stream>");
        }
        const auto cycle = parse_size(tokens[1], "cycle");
        const auto column = parse_size(tokens[2], "column");
        const auto address = parse_size(tokens[4], "address");
        const auto stream = parse_size(tokens[5], "stream");
        if (tokens[3] == "read") {
            program.emit_mem(cycle, column, MemInstruction::Read(address, stream));
            return;
        }
        if (tokens[3] == "write") {
            program.emit_mem(cycle, column, MemInstruction::Write(address, stream));
            return;
        }
        throw std::runtime_error("unsupported MEM opcode in LPU IR: " + tokens[3]);
    }

    if (kind == "mxm_load") {
        if ((tokens.size() != 5 && tokens.size() != 6) || tokens[3] != "iw") {
            throw std::runtime_error("mxm_load syntax: mxm_load <cycle> <mxm> iw <buffer>");
        }
        const auto buffer_index = tokens.size() == 6 ? 5 : 4;
        program.emit_mxm_load(
            parse_size(tokens[1], "cycle"),
            parse_size(tokens[2], "mxm"),
            MxmControlInstruction::IW(parse_size(tokens[buffer_index], "buffer")));
        return;
    }

    if (kind == "mxm_compute") {
        if ((tokens.size() != 5 && tokens.size() != 7) || tokens[3] != "compute") {
            throw std::runtime_error(
                "mxm_compute syntax: mxm_compute <cycle> <mxm> compute <buffer> [<activation_stream> <output_stream>]");
        }
        const auto activation_stream = tokens.size() == 7 ? parse_size(tokens[5], "activation_stream") : 0;
        const auto output_stream = tokens.size() == 7 ? parse_size(tokens[6], "output_stream") : 0;
        program.emit_mxm_compute(
            parse_size(tokens[1], "cycle"),
            parse_size(tokens[2], "mxm"),
            MxmControlInstruction::Compute(parse_size(tokens[4], "buffer"), activation_stream, output_stream));
        return;
    }

    if (kind == "vxm") {
        if (tokens.size() < 6) {
            throw std::runtime_error("vxm syntax: vxm <cycle> <alu> <opcode> ...");
        }
        const auto cycle = parse_size(tokens[1], "cycle");
        const auto alu = parse_size(tokens[2], "alu");
        const auto opcode = parse_vxm_opcode(tokens[3]);

        std::size_t cursor = 4;
        auto instruction = VxmLaneAluInstruction {};
        instruction.opcode = opcode;
        instruction.lhs = parse_operand(tokens, cursor);
        instruction.rhs = VxmLaneOperand::Imm(0.0f);

        if (opcode == VxmAluOpcode::Cast) {
            if (cursor >= tokens.size()) {
                throw std::runtime_error("cast requires fp32 or i8 target");
            }
            instruction.cast_target = parse_cast_target(tokens[cursor++]);
        } else if (opcode != VxmAluOpcode::Pass && opcode != VxmAluOpcode::Negate && opcode != VxmAluOpcode::Exp) {
            instruction.rhs = parse_operand(tokens, cursor);
        }

        if (cursor < tokens.size()) {
            if (cursor + 1 >= tokens.size() || tokens[cursor] != "out") {
                throw std::runtime_error("optional VXM output syntax is: out <stream>");
            }
            instruction.output_stream = parse_size(tokens[cursor + 1], "out");
            cursor += 2;
        }
        if (cursor != tokens.size()) {
            throw std::runtime_error("trailing tokens in VXM instruction");
        }

        program.emit_vxm(cycle, alu, instruction);
        return;
    }

    throw std::runtime_error("unknown LPU IR directive: " + kind);
}

} // namespace

IcuProgram parse_lpu_ir(const std::string& text)
{
    auto program = IcuProgram {};
    std::istringstream lines(text);
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(lines, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        try {
            parse_line(program, tokenize(line));
        } catch (const std::exception& ex) {
            std::ostringstream os;
            os << "LPU IR line " << line_number << ": " << ex.what();
            throw std::runtime_error(os.str());
        }
    }
    return program;
}

} // namespace ftlpu::software::runtime
