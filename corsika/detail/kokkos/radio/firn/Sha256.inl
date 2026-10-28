/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline void Sha256::update(unsigned char const* input, std::size_t length) {
    for (std::size_t index = 0; index < length; ++index) {
      block_[blockLength_++] = input[index];
      if (blockLength_ == 64) {
        transform();
        bitLength_ += 512;
        blockLength_ = 0;
      }
    }
  }

  inline std::array<unsigned char, 32> Sha256::finish() {
    std::size_t index = blockLength_;
    block_[index++] = 0x80;
    if (index > 56) {
      while (index < 64) block_[index++] = 0;
      transform();
      index = 0;
    }
    while (index < 56) block_[index++] = 0;
    bitLength_ += static_cast<std::uint64_t>(blockLength_) * 8;
    for (int byte = 7; byte >= 0; --byte) {
      block_[56 + 7 - byte] =
          static_cast<unsigned char>((bitLength_ >> (8 * byte)) & 0xffu);
    }
    transform();
    std::array<unsigned char, 32> digest{};
    for (std::size_t word = 0; word < 8; ++word) {
      digest[4 * word] = static_cast<unsigned char>(state_[word] >> 24);
      digest[4 * word + 1] = static_cast<unsigned char>(state_[word] >> 16);
      digest[4 * word + 2] = static_cast<unsigned char>(state_[word] >> 8);
      digest[4 * word + 3] = static_cast<unsigned char>(state_[word]);
    }
    return digest;
  }

  inline std::uint32_t Sha256::rotateRight(std::uint32_t value, unsigned count) {
    return (value >> count) | (value << (32 - count));
  }

  inline void Sha256::transform() {
    static constexpr std::array<std::uint32_t, 64> frequencyIndex{
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
        0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
        0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
        0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    std::array<std::uint32_t, 64> w{};
    for (std::size_t index = 0; index < 16; ++index) {
      w[index] = (static_cast<std::uint32_t>(block_[4 * index]) << 24) |
                 (static_cast<std::uint32_t>(block_[4 * index + 1]) << 16) |
                 (static_cast<std::uint32_t>(block_[4 * index + 2]) << 8) |
                 static_cast<std::uint32_t>(block_[4 * index + 3]);
    }
    for (std::size_t index = 16; index < 64; ++index) {
      std::uint32_t const s0 = rotateRight(w[index - 15], 7) ^
                               rotateRight(w[index - 15], 18) ^ (w[index - 15] >> 3);
      std::uint32_t const s1 = rotateRight(w[index - 2], 17) ^
                               rotateRight(w[index - 2], 19) ^ (w[index - 2] >> 10);
      w[index] = w[index - 16] + s0 + w[index - 7] + s1;
    }
    std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (std::size_t index = 0; index < 64; ++index) {
      std::uint32_t const s1 =
          rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
      std::uint32_t const ch = (e & f) ^ (~e & g);
      std::uint32_t const temp1 = h + s1 + ch + frequencyIndex[index] + w[index];
      std::uint32_t const s0 =
          rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
      std::uint32_t const maj = (a & b) ^ (a & c) ^ (b & c);
      std::uint32_t const temp2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }
} // namespace corsika::kokkos::radio::firn
