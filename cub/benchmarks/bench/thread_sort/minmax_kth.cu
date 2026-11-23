// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// k-th element of a sliding window (1D median filter), one sort per thread. Loads come from shared memory, so the
// sorting network dominates. Compares the bitonic and pairwise networks, each with a boolean comparator and with a
// min/max operator for the compare-exchange.

#include <cuda_fp16.h>

#include <cub/thread/thread_sort.cuh>
#include <cub/warp/warp_bitonic_sort.cuh>

#include <thrust/device_vector.h>

#include <cuda/functional>

#include <cstddef>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <nvbench_helper.cuh>

constexpr int threads_per_block = 256;

struct less_op
{
  template <typename T>
  __device__ __forceinline__ bool operator()(T lhs, T rhs) const
  {
    return lhs < rhs;
  }
};

// Only valid for keys without NaN
struct min_max_op
{
  template <typename T>
  __device__ __forceinline__ T min(T lhs, T rhs) const
  {
    return cuda::minimum<T>{}(lhs, rhs);
  }

  template <typename T>
  __device__ __forceinline__ T max(T lhs, T rhs) const
  {
    return cuda::maximum<T>{}(lhs, rhs);
  }

  // Per-component min/max: each thread sorts two independent arrays, one per component
  __device__ __forceinline__ __half2 min(__half2 lhs, __half2 rhs) const
  {
    return __hmin2(lhs, rhs);
  }

  __device__ __forceinline__ __half2 max(__half2 lhs, __half2 rhs) const
  {
    return __hmax2(lhs, rhs);
  }
};

struct bitonic_t
{
  template <typename KeyT, int Length, typename OpT>
  __device__ __forceinline__ static void sort(KeyT (&keys)[Length], OpT op)
  {
    using sort_t = cub::detail::WarpBitonicSort<KeyT, Length, 1>;
    typename sort_t::TempStorage storage;
    sort_t{storage}.Sort(keys, op);
  }
};

struct pairwise_t
{
  template <typename KeyT, int Length, typename OpT>
  __device__ __forceinline__ static void sort(KeyT (&keys)[Length], OpT op)
  {
    cub::NullType values[Length];
    cub::UnstablePairwiseSort(keys, values, op);
  }
};

NVBENCH_DECLARE_TYPE_STRINGS(bitonic_t, "bitonic", "");
NVBENCH_DECLARE_TYPE_STRINGS(pairwise_t, "pairwise", "");

// Key type, operator, and results per key (two for __half2, one per component)
template <typename KeyT, typename OpT, int Lanes = 1>
struct case_t
{
  using key_t                = KeyT;
  using op_t                 = OpT;
  static constexpr int lanes = Lanes;
};

using f32_less      = case_t<float, less_op>;
using f32_min_max   = case_t<float, min_max_op>;
using f16_less      = case_t<__half, less_op>;
using f16_min_max   = case_t<__half, min_max_op>;
using f16x2_min_max = case_t<__half2, min_max_op, 2>;

NVBENCH_DECLARE_TYPE_STRINGS(f32_less, "F32_less", "float, boolean comparison");
NVBENCH_DECLARE_TYPE_STRINGS(f32_min_max, "F32_min_max", "float, min/max operator");
NVBENCH_DECLARE_TYPE_STRINGS(f16_less, "F16_less", "__half, boolean comparison");
NVBENCH_DECLARE_TYPE_STRINGS(f16_min_max, "F16_min_max", "__half, min/max operator");
NVBENCH_DECLARE_TYPE_STRINGS(f16x2_min_max, "F16x2_min_max", "__half2, min/max operator");

template <typename AlgorithmT, typename KeyT, int Length, int K, typename OpT>
__global__ void sliding_window_kth(const KeyT* input, KeyT* output)
{
  static_assert(0 <= K && K < Length, "K must index into the window");

  __shared__ KeyT tile[threads_per_block + Length - 1];
  const int tid          = static_cast<int>(threadIdx.x);
  const int block_offset = static_cast<int>(blockIdx.x) * threads_per_block;
  for (int i = tid; i < threads_per_block + Length - 1; i += threads_per_block)
  {
    tile[i] = input[block_offset + i];
  }
  __syncthreads();

  KeyT keys[Length];
  for (int i = 0; i < Length; ++i)
  {
    keys[i] = tile[tid + i];
  }
  AlgorithmT::sort(keys, OpT{});

  output[block_offset + tid] = keys[K];
}

template <typename KeyT>
KeyT random_key(std::mt19937& rng)
{
  if constexpr (std::is_same_v<KeyT, __half2>)
  {
    const __half low = random_key<__half>(rng);
    return __halves2half2(low, random_key<__half>(rng));
  }
  else
  {
    std::uniform_int_distribution<int> dist(-20, 20);
    return static_cast<KeyT>(dist(rng));
  }
}

template <typename AlgorithmT, typename CaseT, int Length>
void kth_element(nvbench::state& state, nvbench::type_list<AlgorithmT, CaseT, nvbench::enum_type<Length>>)
{
  using key_t     = typename CaseT::key_t;
  using op_t      = typename CaseT::op_t;
  constexpr int k = Length / 2;

  const auto size = state.get_int64("Size");
  if (size > std::numeric_limits<int>::max() || size % (threads_per_block * CaseT::lanes) != 0)
  {
    state.skip("Size must fit into int and be a multiple of the block size times the lanes");
    return;
  }
  const int num_outputs = static_cast<int>(size) / CaseT::lanes;

  // The network is data-oblivious -> values do not matter
  std::mt19937 rng(1909);
  std::vector<key_t> h_input(num_outputs + Length - 1);
  for (key_t& key : h_input)
  {
    key = random_key<key_t>(rng);
  }
  thrust::device_vector<key_t> input(h_input.begin(), h_input.end());
  thrust::device_vector<key_t> output(num_outputs);

  state.add_element_count(static_cast<std::size_t>(size), "Results");
  state.add_global_memory_reads<key_t>(input.size());
  state.add_global_memory_writes<key_t>(output.size());

  state.exec([&](nvbench::launch& launch) {
    sliding_window_kth<AlgorithmT, key_t, Length, k, op_t>
      <<<num_outputs / threads_per_block, threads_per_block, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()));
  });
}

using algorithms = nvbench::type_list<bitonic_t, pairwise_t>;
using cases      = nvbench::type_list<f32_less, f32_min_max, f16_less, f16_min_max, f16x2_min_max>;
using lengths    = nvbench::enum_type_list<16, 40, 64>;

NVBENCH_BENCH_TYPES(kth_element, NVBENCH_TYPE_AXES(algorithms, cases, lengths))
  .set_type_axes_names({"Algorithm", "Case", "Length"})
  .add_int64_power_of_two_axis("Size", {22});
