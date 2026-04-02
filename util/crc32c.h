#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vecbase {

std::uint32_t CRC32C(const std::byte *data, std::size_t size);
std::uint32_t CRC32C(const std::vector<std::byte> &data);

} // namespace vecbase
