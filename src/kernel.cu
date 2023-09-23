#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include <cstdint>
#include <iostream>

namespace {

constexpr int N = 624;
constexpr int M = 397;
constexpr uint32_t ONE = 0x1UL;
constexpr uint32_t MATRIX_A = 0x9908b0dfUL;
constexpr uint32_t UPPER_MASK = 0x80000000UL;
constexpr uint32_t LOWER_MASK = 0x7fffffffUL;

constexpr int PRE_ADVANCE_FRAME = 63;
constexpr int MAX_FRAME = 2000;
constexpr int F1_MIN = 600;
constexpr int F1_MAX = 800;
constexpr int F2_MIN = 1500;
constexpr int F2_MAX = 1700;

__host__ __device__ inline uint32_t init_ivs(uint32_t a, uint32_t b, uint32_t c,
                                             uint32_t d, uint32_t s) {
  return (a << 20) | (b << 15) | (c << 10) | (d << 5) | s;
}

__host__ __device__ inline uint32_t temper(uint32_t x) {
  x ^= x >> 11;
  x ^= (x << 7) & 0x9d2c5680UL;
  x ^= (x << 15) & 0xefc60000UL;
  x ^= x >> 18;
  return x;
}

struct is_hit {
  const uint32_t ivs1;
  const uint32_t ivs2;

  is_hit(uint32_t _ivs1, uint32_t _ivs2) : ivs1(_ivs1), ivs2(_ivs2) {}

  __host__ __device__ bool operator()(uint32_t seed) const {
    uint32_t state[N + MAX_FRAME];
    int idx = 0;

    state[0] = seed;
#pragma unroll
    for (int i = 1; i < N; i++) {
      state[i] = 1812433253UL * (state[i - 1] ^ (state[i - 1] >> 30)) + i;
    }
#pragma unroll
    for (int i = N; i < N + MAX_FRAME; i++) {
      uint32_t x =
          (state[i - N] & UPPER_MASK) | (state[i + 1 - N] & LOWER_MASK);
      state[i] = state[i + M - N] ^ (x >> 1) ^ ((x & ONE) * MATRIX_A);
    }
#pragma unroll
    for (int i = N; i < N + MAX_FRAME; i++) {
      state[i] = temper(state[i]);
    }
    idx = N;

    idx += PRE_ADVANCE_FRAME + F1_MIN;
    uint32_t curr =
        init_ivs(state[idx++] >> 27, state[idx++] >> 27, state[idx++] >> 27,
                 state[idx++] >> 27, state[idx++] >> 27);
    bool found_f1 = false;
#pragma unroll
    for (int i = F1_MIN; i <= F1_MAX; i++) {
      curr = (curr & 0x1FFFFFFUL) << 5 | (state[idx++] >> 27);
      found_f1 = found_f1 || (curr == ivs1);
    }

    idx += F2_MIN - F1_MAX - 6;
    curr = init_ivs(state[idx++] >> 27, state[idx++] >> 27, state[idx++] >> 27,
                    state[idx++] >> 27, state[idx++] >> 27);
    bool found_f2 = false;
#pragma unroll
    for (int i = F2_MIN; i <= F2_MAX; i++) {
      curr = (curr & 0x1FFFFFFUL) << 5 | (state[idx++] >> 27);
      found_f2 = found_f2 || (curr == ivs2);
    }

    return found_f1 && found_f2;
  }
};

}  // namespace

extern "C" {
void find_seed_gpu(uint32_t seed_min, uint32_t seed_max, uint32_t ivs1,
                   uint32_t ivs2, uint32_t f1_min, uint32_t f1_max,
                   uint32_t f2_min, uint32_t f2_max) {
  int device_count;
  if (cudaGetDeviceCount(&device_count) == cudaSuccess) {
    // std::printf("%d device(s) are available :)\n", device_count);
  } else {
    std::printf("CUDA is not available :(\n");
    return;
  }

  int64_t n = (int64_t)seed_max - seed_min + 1;
  thrust::device_vector<uint32_t> dvec(n);

  auto begin = thrust::make_counting_iterator(seed_min);
  auto end =
      thrust::copy_if(begin, begin + n, dvec.begin(), is_hit(ivs1, ivs2));
  dvec.resize(thrust::distance(dvec.begin(), end));

  thrust::host_vector<uint32_t> hvec = dvec;

  std::cout << "Results:" << std::endl;
  for (int i = 0; i < hvec.size(); i++) {
    std::cout << "- Seed: " << hvec[i] << std::endl;
  }
}
}
