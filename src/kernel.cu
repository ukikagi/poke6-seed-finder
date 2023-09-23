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

__host__ __device__ inline uint32_t encode_ivs(uint32_t h, uint32_t a,
                                               uint32_t b, uint32_t c,
                                               uint32_t d, uint32_t s) {
  return (h << 25) | (a << 20) | (b << 15) | (c << 10) | (d << 5) | s;
}

__host__ __device__ inline uint32_t temper(uint32_t x) {
  x ^= x >> 11;
  x ^= (x << 7) & 0x9d2c5680UL;
  x ^= (x << 15) & 0xefc60000UL;
  x ^= x >> 18;
  return x;
}

struct mt19937 {
  uint32_t state[N];
  int idx = 0;

  __host__ __device__ void reseed(uint32_t seed) {
    idx = N;
    state[0] = seed;
    for (int i = 1; i < N; i++) {
      state[i] = 1812433253UL * (state[i - 1] ^ (state[i - 1] >> 30)) + i;
    }
  }

  __host__ __device__ uint32_t next_u32() {
    if (idx >= N) {
      fill_next_state();
    }
    return temper(state[idx++]);
  }

  __host__ __device__ uint32_t next_iv() { return next_u32() >> 27; }

  __host__ __device__ void discard(int n) {
    for (int i = 0; i < n; i++) {
      next_u32();
    }
  }

  __host__ __device__ void fill_next_state() {
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

constexpr int PRE_ADVANCE_FRAME = 63;

struct is_hit {
  const uint32_t ivs1;
  const uint32_t ivs2;
  const int f1_min;
  const int f1_max;
  const int f2_min;
  const int f2_max;

  is_hit(uint32_t _ivs1, uint32_t _ivs2, int _f1_min, int _f1_max, int _f2_min,
         int _f2_max)
      : ivs1(_ivs1),
        ivs2(_ivs2),
        f1_min(_f1_min),
        f1_max(_f1_max),
        f2_min(_f2_min),
        f2_max(_f2_max) {}

  // Advances f_max + 6 - f_min frames
  __host__ __device__ bool find_frame(mt19937& mt, uint32_t ivs, uint32_t f_min,
                                      uint32_t f_max) const {
    uint32_t curr = encode_ivs(0, mt.next_iv(), mt.next_iv(), mt.next_iv(),
                               mt.next_iv(), mt.next_iv());
    bool result = false;
    for (int i = f_min; i <= f_max; i++) {
      curr = (curr & 0x1FFFFFFUL) << 5 | mt.next_iv();
      result = result || curr == ivs;
    }
    return result;
  }

  __host__ __device__ bool operator()(uint32_t seed) const {
    mt19937 mt;
    mt.reseed(seed);

    mt.discard(PRE_ADVANCE_FRAME + f1_min);
    // Advances f1_max + 6 - f1_min frames
    bool found_f1 = find_frame(mt, ivs1, f1_min, f1_max);

    mt.discard(f2_min - f1_max - 6);
    // Advances f2_max + 6 - f2_min frames
    bool found_f2 = find_frame(mt, ivs2, f2_min, f2_max);

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
      thrust::copy_if(begin, begin + n, dvec.begin(),
                      is_hit(ivs1, ivs2, f1_min, f1_max, f2_min, f2_max));
  dvec.resize(thrust::distance(dvec.begin(), end));

  thrust::host_vector<uint32_t> hvec = dvec;

  std::cout << "Results:" << std::endl;
  for (int i = 0; i < hvec.size(); i++) {
    std::cout << "- Seed: " << hvec[i] << std::endl;
  }
}
}
