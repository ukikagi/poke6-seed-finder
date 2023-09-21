#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include <cstdint>
#include <iostream>

constexpr int N = 624;
constexpr int M = 397;
constexpr uint32_t ONE = 0x1UL;
constexpr uint32_t MATRIX_A = 0x9908b0dfUL;
constexpr uint32_t UPPER_MASK = 0x80000000UL;
constexpr uint32_t LOWER_MASK = 0x7fffffffUL;

inline uint32_t temper(uint32_t x) {
  x ^= x >> 11;
  x ^= (x << 7) & 0x9d2c5680UL;
  x ^= (x << 15) & 0xefc60000UL;
  x ^= x >> 18;
  return x;
}

struct mt19937 {
  uint32_t state[N];
  int idx = 0;

  void reseed(uint32_t seed) {
    idx = N;
    state[0] = seed;
    for (int i = 1; i < N; i++) {
      state[i] = 1812433253UL * (state[i - 1] ^ (state[i - 1] >> 30)) + i;
    }
  }

  uint32_t next_u32() {
    if (idx >= N) {
      fill_next_state();
    }
    return temper(state[idx++]);
  }

  uint32_t next_iv() { return next_u32() >> 27; }

  void discard(int n) {
    for (int i = 0; i < n; i++) {
      next_u32();
    }
  }

  void fill_next_state() {
    uint32_t x;
    for (int i = 0; i < N - M; i++) {
      x = (state[i] & UPPER_MASK) | (state[i + 1] & LOWER_MASK);
      state[i] = state[i + M] ^ (x >> 1) ^ ((x & ONE) * MATRIX_A);
    }
    for (int i = N - M; i < N - 1; i++) {
      x = (state[i] & UPPER_MASK) | (state[i + 1] & LOWER_MASK);
      state[i] = state[i + M - N] ^ (x >> 1) ^ ((x & ONE) * MATRIX_A);
    }
    x = (state[N - 1] & UPPER_MASK) | (state[0] & LOWER_MASK);
    state[N - 1] = state[M - 1] ^ (x >> 1) ^ ((x & ONE) * MATRIX_A);
    idx = 0;
  }
};

extern "C" {
int find_seed_gpu() {
  mt19937 mt;
  mt.reseed(0);
  mt.discard(1000);
  std::cout << mt.next_u32() << std::endl;
  return 0;
}
}
