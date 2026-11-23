// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/thread/thread_sort.cuh>

#include <thrust/memory.h>

#include <cuda/functional>
#include <cuda/std/bit>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "cub_test_macros.h"
#include <c2h/extended_types.h>
#include <c2h/generators.h>
#include <c2h/vector.h>

enum class sort_algorithm
{
  stable,
  unstable
};

// use popcount to test sort stability
// example:
// {3, 5, 1}
// popcounts: 2, 2, 1
// stable sort must preserve the original order of equal elements:
// {1, 3, 5}
// unstable sort can change the order of equal elements:
// {1, 5, 3}
struct popcount_less
{
  template <typename T>
  __host__ __device__ bool operator()(const T& lhs, const T& rhs) const
  {
    using unsigned_t  = cuda::std::make_unsigned_t<T>;
    auto lhs_unsigned = static_cast<unsigned_t>(lhs);
    auto rhs_unsigned = static_cast<unsigned_t>(rhs);
    return cuda::std::popcount(lhs_unsigned) < cuda::std::popcount(rhs_unsigned);
  }
};

template <sort_algorithm Algorithm, typename KeyT, typename ValueT, int ItemsPerThread, typename CompareOp>
__device__ void sort(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], CompareOp compare_op)
{
  if constexpr (Algorithm == sort_algorithm::stable)
  {
    cub::StableOddEvenSort(keys, values, compare_op);
  }
  else
  {
    cub::UnstablePairwiseSort(keys, values, compare_op);
  }
}

template <sort_algorithm Algorithm, typename KeyT, int ItemsPerThread, typename CompareOp>
__global__ void sort_keys_kernel(const KeyT* keys_in, KeyT* keys_out, CompareOp compare_op)
{
  const int thread_offset = ItemsPerThread * static_cast<int>(threadIdx.x);
  KeyT keys[ItemsPerThread];
  cub::NullType values[ItemsPerThread];

  for (int item = 0; item < ItemsPerThread; ++item)
  {
    keys[item] = keys_in[thread_offset + item];
  }
  sort<Algorithm>(keys, values, compare_op);

  for (int item = 0; item < ItemsPerThread; ++item)
  {
    keys_out[thread_offset + item] = keys[item];
  }
}

template <sort_algorithm Algorithm, typename KeyT, typename ValueT, int ItemsPerThread, typename CompareOp>
__global__ void sort_pairs_kernel(
  const KeyT* keys_in, KeyT* keys_out, const ValueT* values_in, ValueT* values_out, CompareOp compare_op)
{
  const int thread_offset = ItemsPerThread * static_cast<int>(threadIdx.x);
  KeyT keys[ItemsPerThread];
  ValueT values[ItemsPerThread];

  for (int item = 0; item < ItemsPerThread; ++item)
  {
    keys[item]   = keys_in[thread_offset + item];
    values[item] = values_in[thread_offset + item];
  }
  sort<Algorithm>(keys, values, compare_op);

  for (int item = 0; item < ItemsPerThread; ++item)
  {
    keys_out[thread_offset + item]   = keys[item];
    values_out[thread_offset + item] = values[item];
  }
}

using key_types             = c2h::type_list<uint32_t, int64_t>;
using value_types           = c2h::type_list<uint32_t, uint64_t>;
using items_per_thread_list = c2h::enum_type_list<int, 1, 2, 3, 4, 5, 7, 8, 11, 16>;
using algorithms            = c2h::enum_type_list<sort_algorithm, sort_algorithm::stable, sort_algorithm::unstable>;

CUB_TEST("Thread sort keys works", "[sort][thread]", CUB_SMALL, key_types, items_per_thread_list, algorithms)
{
  using key_t                    = c2h::get<0, TestType>;
  constexpr int items_per_thread = c2h::get<1, TestType>::value;
  constexpr auto algorithm       = c2h::get<2, TestType>::value;
  constexpr int threads          = 128;
  constexpr int num_items        = threads * items_per_thread;

  c2h::device_vector<key_t> keys_in(num_items);
  c2h::device_vector<key_t> keys_out(num_items);
  c2h::gen(C2H_SEED(3), keys_in);
  c2h::host_vector<key_t> expected_input = keys_in;

  sort_keys_kernel<algorithm, key_t, items_per_thread><<<1, threads>>>(
    thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()), popcount_less{});
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<key_t> actual = keys_out;

  for (int thread = 0; thread < threads; ++thread)
  {
    auto begin          = thread * items_per_thread;
    auto expected_begin = expected_input.begin() + begin;
    auto actual_begin   = actual.begin() + begin;

    REQUIRE(std::is_sorted(actual_begin, actual_begin + items_per_thread, popcount_less{}));
    REQUIRE(std::is_permutation(expected_begin, expected_begin + items_per_thread, actual_begin));

    if constexpr (algorithm == sort_algorithm::stable)
    {
      std::vector<key_t> expected(expected_begin, expected_begin + items_per_thread);
      std::stable_sort(expected.begin(), expected.end(), popcount_less{});
      REQUIRE(std::equal(expected.begin(), expected.end(), actual_begin));
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// key-value sorting tests

template <typename KeyT, typename ValueT>
using pair_vector_t = std::vector<std::pair<KeyT, ValueT>>;

template <typename KeyT, typename ValueT>
pair_vector_t<KeyT, ValueT> make_pairs(const c2h::host_vector<KeyT>& keys, const c2h::host_vector<ValueT>& values)
{
  pair_vector_t<KeyT, ValueT> pairs(keys.size());
  for (size_t i = 0; i < keys.size(); ++i)
  {
    pairs[i] = {keys[i], values[i]};
  }
  return pairs;
}

CUB_TEST(
  "Thread sort pairs works", "[sort][thread]", CUB_SMALL, key_types, value_types, items_per_thread_list, algorithms)
{
  using key_t                    = c2h::get<0, TestType>;
  using value_t                  = c2h::get<1, TestType>;
  constexpr int items_per_thread = c2h::get<2, TestType>::value;
  constexpr auto algorithm       = c2h::get<3, TestType>::value;
  constexpr int threads          = 128;
  constexpr int num_items        = threads * items_per_thread;

  c2h::device_vector<key_t> keys_in(num_items);
  c2h::device_vector<key_t> keys_out(num_items);
  c2h::gen(C2H_SEED(3), keys_in);

  c2h::host_vector<value_t> values_in(num_items);
  std::iota(values_in.begin(), values_in.end(), value_t{});
  c2h::device_vector<value_t> device_values_in = values_in;
  c2h::device_vector<value_t> values_out(num_items);

  const c2h::host_vector<key_t> host_keys_in = keys_in;
  auto expected_input                        = make_pairs(host_keys_in, values_in);

  sort_pairs_kernel<algorithm, key_t, value_t, items_per_thread><<<1, threads>>>(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(device_values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    popcount_less{});
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  const c2h::host_vector<key_t> host_keys_out     = keys_out;
  const c2h::host_vector<value_t> host_values_out = values_out;
  auto actual                                     = make_pairs(host_keys_out, host_values_out);
  auto pair_less                                  = [](auto lhs, auto rhs) {
    return popcount_less{}(lhs.first, rhs.first);
  };

  for (int thread = 0; thread < threads; ++thread)
  {
    auto begin          = thread * items_per_thread;
    auto expected_begin = expected_input.begin() + begin;
    auto actual_begin   = actual.begin() + begin;

    REQUIRE(std::is_sorted(actual_begin, actual_begin + items_per_thread, pair_less));
    REQUIRE(std::is_permutation(expected_begin, expected_begin + items_per_thread, actual_begin));

    if constexpr (algorithm == sort_algorithm::stable)
    {
      pair_vector_t<key_t, value_t> expected(expected_begin, expected_begin + items_per_thread);
      std::stable_sort(expected.begin(), expected.end(), pair_less);
      REQUIRE(std::equal(expected.begin(), expected.end(), actual_begin));
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// min/max operator tests

struct min_max_op
{
  template <typename T>
  __device__ T min(T lhs, T rhs) const
  {
    return cuda::minimum<T>{}(lhs, rhs);
  }

  template <typename T>
  __device__ T max(T lhs, T rhs) const
  {
    return cuda::maximum<T>{}(lhs, rhs);
  }

#if TEST_HALF_T()
  // Per-component min/max: each thread sorts two independent arrays, one per component
  __device__ __half2 min(__half2 lhs, __half2 rhs) const
  {
    return __hmin2(lhs, rhs);
  }

  __device__ __half2 max(__half2 lhs, __half2 rhs) const
  {
    return __hmax2(lhs, rhs);
  }
#endif // TEST_HALF_T()
};

// Comparator whose min and max members are reversed, to check which members a sort uses
struct less_with_reversed_min_max
{
  template <typename T>
  __host__ __device__ bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs < rhs;
  }

  template <typename T>
  __device__ T min(T lhs, T rhs) const
  {
    return cuda::maximum<T>{}(lhs, rhs);
  }

  template <typename T>
  __device__ T max(T lhs, T rhs) const
  {
    return cuda::minimum<T>{}(lhs, rhs);
  }
};

struct only_min_op
{
  __device__ float min(float lhs, float rhs) const;
};

struct wrong_return_type_op
{
  __device__ double min(float lhs, float rhs) const;
  __device__ double max(float lhs, float rhs) const;
};

static_assert(cub::detail::is_min_max_op_v<min_max_op, float>);
static_assert(cub::detail::is_min_max_op_v<less_with_reversed_min_max, uint32_t>);
static_assert(!cub::detail::is_min_max_op_v<popcount_less, uint32_t>);
static_assert(!cub::detail::is_min_max_op_v<only_min_op, float>);
static_assert(!cub::detail::is_min_max_op_v<wrong_return_type_op, float>);

// Number of independent arrays that one key holds
template <typename KeyT>
inline constexpr int lanes_v = 1;

#if TEST_HALF_T()
template <>
inline constexpr int lanes_v<__half2> = 2;
#endif // TEST_HALF_T()

// Input value that stands for -0.0
inline constexpr int negative_zero_code = 21;

template <typename KeyT>
__host__ __device__ float decode_key(int value)
{
  if constexpr (cuda::std::is_integral_v<KeyT>)
  {
    return static_cast<float>(value == negative_zero_code ? 0 : value);
  }
  else
  {
    return value == negative_zero_code ? -0.0f : static_cast<float>(value);
  }
}

template <typename KeyT>
__device__ KeyT make_key(const int* lane_values)
{
#if TEST_HALF_T()
  if constexpr (cuda::std::is_same_v<KeyT, __half2>)
  {
    return __floats2half2_rn(decode_key<KeyT>(lane_values[0]), decode_key<KeyT>(lane_values[1]));
  }
  else if constexpr (cuda::std::is_same_v<KeyT, __half>)
  {
    return __float2half(decode_key<KeyT>(lane_values[0]));
  }
  else
#endif // TEST_HALF_T()
  {
    return static_cast<KeyT>(decode_key<KeyT>(lane_values[0]));
  }
}

template <typename KeyT>
__device__ void store_key(KeyT key, float* lane_values)
{
#if TEST_HALF_T()
  if constexpr (cuda::std::is_same_v<KeyT, __half2>)
  {
    lane_values[0] = __low2float(key);
    lane_values[1] = __high2float(key);
  }
  else if constexpr (cuda::std::is_same_v<KeyT, __half>)
  {
    lane_values[0] = __half2float(key);
  }
  else
#endif // TEST_HALF_T()
  {
    lane_values[0] = static_cast<float>(key);
  }
}

// Input is laid out as [thread][item][lane]
template <typename KeyT, int ItemsPerThread>
__global__ void sort_min_max_kernel(const int* keys_in, float* keys_out)
{
  constexpr int lanes     = lanes_v<KeyT>;
  const int thread_offset = ItemsPerThread * lanes * static_cast<int>(threadIdx.x);
  KeyT keys[ItemsPerThread];
  cub::NullType values[ItemsPerThread];

  for (int item = 0; item < ItemsPerThread; ++item)
  {
    keys[item] = make_key<KeyT>(keys_in + thread_offset + item * lanes);
  }
  cub::UnstablePairwiseSort(keys, values, min_max_op{});

  for (int item = 0; item < ItemsPerThread; ++item)
  {
    store_key(keys[item], keys_out + thread_offset + item * lanes);
  }
}

#if TEST_HALF_T()
using min_max_key_types = c2h::type_list<int32_t, float, __half, __half2>;
#else // ^^^ TEST_HALF_T() ^^^ / vvv !TEST_HALF_T() vvv
using min_max_key_types = c2h::type_list<int32_t, float>;
#endif // !TEST_HALF_T()

CUB_TEST("Unstable thread sort works with min/max operators",
         "[sort][thread]",
         CUB_SMALL,
         min_max_key_types,
         items_per_thread_list)
{
  using key_t                    = c2h::get<0, TestType>;
  constexpr int items_per_thread = c2h::get<1, TestType>::value;
  constexpr int lanes            = lanes_v<key_t>;
  constexpr int threads          = 128;
  constexpr int num_items        = threads * items_per_thread * lanes;

  // Small range for many duplicates, including -0.0 and +0.0; exactly representable in every key type
  c2h::device_vector<int> keys_in(num_items);
  c2h::device_vector<float> keys_out(num_items);
  c2h::gen(C2H_SEED(3), keys_in, -20, negative_zero_code);

  sort_min_max_kernel<key_t, items_per_thread>
    <<<1, threads>>>(thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  const c2h::host_vector<int> input    = keys_in;
  const c2h::host_vector<float> actual = keys_out;

  for (int thread = 0; thread < threads; ++thread)
  {
    for (int lane = 0; lane < lanes; ++lane)
    {
      std::vector<float> expected(items_per_thread);
      std::vector<float> actual_lane(items_per_thread);
      for (int item = 0; item < items_per_thread; ++item)
      {
        const int index   = (thread * items_per_thread + item) * lanes + lane;
        expected[item]    = decode_key<key_t>(input[index]);
        actual_lane[item] = actual[index];
      }
      std::sort(expected.begin(), expected.end());
      REQUIRE(expected == actual_lane);

      // -0.0 == +0.0, so the comparison above does not check that the keys are a permutation of the input
      const auto is_negative_zero = [](float key) {
        return key == 0.0f && std::signbit(key);
      };
      REQUIRE(std::count_if(expected.begin(), expected.end(), is_negative_zero)
              == std::count_if(actual_lane.begin(), actual_lane.end(), is_negative_zero));
    }
  }
}

CUB_TEST("Thread sort uses operator() unless sorting keys-only with the unstable sort",
         "[sort][thread]",
         CUB_SMALL,
         items_per_thread_list,
         algorithms)
{
  using key_t                    = uint32_t;
  using value_t                  = uint32_t;
  constexpr int items_per_thread = c2h::get<0, TestType>::value;
  constexpr auto algorithm       = c2h::get<1, TestType>::value;
  constexpr int threads          = 128;
  constexpr int num_items        = threads * items_per_thread;

  c2h::device_vector<key_t> keys_in(num_items);
  c2h::device_vector<key_t> keys_out(num_items);
  c2h::device_vector<key_t> pair_keys_out(num_items);
  c2h::device_vector<value_t> values_in(num_items);
  c2h::device_vector<value_t> values_out(num_items);
  c2h::gen(C2H_SEED(3), keys_in);

  sort_keys_kernel<algorithm, key_t, items_per_thread><<<1, threads>>>(
    thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()), less_with_reversed_min_max{});
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  sort_pairs_kernel<algorithm, key_t, value_t, items_per_thread><<<1, threads>>>(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(pair_keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    less_with_reversed_min_max{});
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  const c2h::host_vector<key_t> keys      = keys_out;
  const c2h::host_vector<key_t> pair_keys = pair_keys_out;

  for (int thread = 0; thread < threads; ++thread)
  {
    const auto keys_begin      = keys.begin() + thread * items_per_thread;
    const auto pair_keys_begin = pair_keys.begin() + thread * items_per_thread;

    // The reversed min and max members sort descending
    if constexpr (algorithm == sort_algorithm::unstable)
    {
      REQUIRE(std::is_sorted(keys_begin, keys_begin + items_per_thread, std::greater<>{}));
    }
    else
    {
      REQUIRE(std::is_sorted(keys_begin, keys_begin + items_per_thread));
    }
    REQUIRE(std::is_sorted(pair_keys_begin, pair_keys_begin + items_per_thread));
  }
}
