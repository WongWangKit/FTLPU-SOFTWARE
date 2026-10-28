#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/issue_inspector.hpp"

#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

// Compare one ICU at a time: a complete decoder can contain millions of FU
// issues, so retaining both expanded whole-chip images is unnecessary.
int main(int argc, char** argv) try {
    using namespace ftlpu::software::runtime;
    if (argc != 3)
        throw std::runtime_error("usage: ftlpu_compare_compute_issues left.ftlpu right.ftlpu");
    const auto left = read_binary_program(argv[1]);
    const auto right = read_binary_program(argv[2]);
    if (left.target_abi != right.target_abi || left.target_name != right.target_name
        || left.max_cycle != right.max_cycle)
        throw std::runtime_error("target or logical horizon differs");
    using Key = std::pair<QueueKind, std::size_t>;
    std::set<Key> keys;
    for (const auto* program : {&left, &right}) {
        std::set<Key> seen;
        for (const auto& queue : program->queues) {
            const Key key{queue.kind, queue.index};
            if (!seen.insert(key).second) throw std::runtime_error("duplicate ICU queue");
            keys.insert(key);
        }
    }
    std::size_t issues = 0;
    for (const auto& key : keys) {
        BinaryProgram a, b;
        a.target_name = b.target_name = left.target_name;
        a.target_abi = b.target_abi = left.target_abi;
        a.max_cycle = b.max_cycle = left.max_cycle;
        a.hardware = left.hardware;
        b.hardware = right.hardware;
        const auto copy = [&](const auto& source, auto& destination) {
            for (const auto& queue : source.queues)
                if (Key{queue.kind, queue.index} == key) destination.queues.push_back(queue);
        };
        copy(left, a);
        copy(right, b);
        const auto comparison = compare_logical_issues(a, b);
        if (!comparison.equivalent) {
            const auto& mismatch = *comparison.first_mismatch;
            throw std::runtime_error(std::string(queue_kind_name(key.first))
                + "[" + std::to_string(key.second) + "] cycle="
                + std::to_string(mismatch.cycle) + ": " + mismatch.reason);
        }
        issues += inspect_logical_issues(a).functional_issues;
    }
    std::cout << "compute_issue_comparison equivalent=1 queues=" << keys.size()
              << " functional_issues=" << issues << " max_cycle=" << left.max_cycle
              << " transport_excluded=1\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "compute_issue_comparison failed: " << error.what() << '\n';
    return 1;
}
