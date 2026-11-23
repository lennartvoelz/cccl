// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
// Type trait to detect if the passed functor has min and max members that return KeyT. Such an operator replaces the
// comparator in keys-only sorts and must keep the keys a permutation: for every pair of keys a and b,
// {min(a, b), max(a, b)} must equal {a, b}, with min(a, b) ordered before or equivalent to max(a, b) under one strict
// weak order for all pairs
template <typename CompareOp, typename KeyT, typename = void>
inline constexpr bool is_min_max_op_v = false;

// TODO: simplify and make more readable
template <typename CompareOp, typename KeyT>
inline constexpr bool is_min_max_op_v<
  CompareOp,
  KeyT,
  ::cuda::std::void_t<decltype((::cuda::std::declval<CompareOp&>().min)(
                        ::cuda::std::declval<const KeyT&>(), ::cuda::std::declval<const KeyT&>())),
                      decltype((::cuda::std::declval<CompareOp&>().max)(
                        ::cuda::std::declval<const KeyT&>(), ::cuda::std::declval<const KeyT&>()))>> =
  ::cuda::std::is_same_v<::cuda::std::remove_cvref_t<decltype((::cuda::std::declval<CompareOp&>().min)(
                           ::cuda::std::declval<const KeyT&>(), ::cuda::std::declval<const KeyT&>()))>,
                         KeyT>
  && ::cuda::std::is_same_v<::cuda::std::remove_cvref_t<decltype((::cuda::std::declval<CompareOp&>().max)(
                              ::cuda::std::declval<const KeyT&>(), ::cuda::std::declval<const KeyT&>()))>,
                            KeyT>;
} // namespace detail

CUB_NAMESPACE_END
