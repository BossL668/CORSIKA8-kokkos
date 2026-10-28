/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/utility/Sha256.hpp>

#include <array>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace corsika::kokkos {

  namespace {

    constexpr std::array<std::uint32_t, 64> RoundConstants{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
        0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
        0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
        0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
        0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

    constexpr std::uint32_t rotate_right(std::uint32_t value, unsigned int shift) {
      return (value >> shift) | (value << (32U - shift));
    }

    void transform(std::array<std::uint32_t, 8>& state, std::uint8_t const* block) {
      std::array<std::uint32_t, 64> words{};
      for (std::size_t index = 0; index < 16; ++index) {
        auto const offset = 4 * index;
        words[index] = (static_cast<std::uint32_t>(block[offset]) << 24U) |
                       (static_cast<std::uint32_t>(block[offset + 1]) << 16U) |
                       (static_cast<std::uint32_t>(block[offset + 2]) << 8U) |
                       static_cast<std::uint32_t>(block[offset + 3]);
      }
      for (std::size_t index = 16; index < words.size(); ++index) {
        auto const s0 = rotate_right(words[index - 15], 7) ^
                        rotate_right(words[index - 15], 18) ^ (words[index - 15] >> 3U);
        auto const s1 = rotate_right(words[index - 2], 17) ^
                        rotate_right(words[index - 2], 19) ^ (words[index - 2] >> 10U);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
      }

      auto a = state[0];
      auto b = state[1];
      auto c = state[2];
      auto d = state[3];
      auto e = state[4];
      auto f = state[5];
      auto g = state[6];
      auto h = state[7];

      for (std::size_t index = 0; index < words.size(); ++index) {
        auto const sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        auto const choose = (e & f) ^ (~e & g);
        auto const temp1 = h + sum1 + choose + RoundConstants[index] + words[index];
        auto const sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        auto const majority = (a & b) ^ (a & c) ^ (b & c);
        auto const temp2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
      }

      state[0] += a;
      state[1] += b;
      state[2] += c;
      state[3] += d;
      state[4] += e;
      state[5] += f;
      state[6] += g;
      state[7] += h;
    }

    sha256_digest_type sha256_bytes(std::uint8_t const* data, std::size_t size) {
      if (size != 0 && data == nullptr) {
        throw std::invalid_argument("SHA-256 received a null non-empty input");
      }
      if (size > std::numeric_limits<std::uint64_t>::max() / 8U) {
        throw std::length_error("SHA-256 input is too large");
      }

      std::array<std::uint32_t, 8> state{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                                         0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                                         0x1f83d9abU, 0x5be0cd19U};

      std::size_t offset = 0;
      while (size - offset >= 64) {
        transform(state, data + offset);
        offset += 64;
      }

      std::array<std::uint8_t, 128> tail{};
      auto const remainder = size - offset;
      for (std::size_t index = 0; index < remainder; ++index) {
        tail[index] = data[offset + index];
      }
      tail[remainder] = 0x80U;
      auto const padded_size = remainder < 56 ? 64U : 128U;
      auto const bit_size = static_cast<std::uint64_t>(size) * 8U;
      for (std::size_t index = 0; index < 8; ++index) {
        tail[padded_size - 1 - index] =
            static_cast<std::uint8_t>(bit_size >> (8U * index));
      }
      transform(state, tail.data());
      if (padded_size == 128) { transform(state, tail.data() + 64); }

      sha256_digest_type digest{};
      for (std::size_t index = 0; index < state.size(); ++index) {
        digest[4 * index] = static_cast<std::uint8_t>(state[index] >> 24U);
        digest[4 * index + 1] = static_cast<std::uint8_t>(state[index] >> 16U);
        digest[4 * index + 2] = static_cast<std::uint8_t>(state[index] >> 8U);
        digest[4 * index + 3] = static_cast<std::uint8_t>(state[index]);
      }
      return digest;
    }

  } // namespace

  sha256_digest_type sha256(std::vector<std::uint8_t> const& data) {
    return sha256_bytes(data.data(), data.size());
  }

  std::string to_hex(sha256_digest_type const& digest) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (auto const byte : digest) {
      stream << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return stream.str();
  }

} // namespace corsika::kokkos
