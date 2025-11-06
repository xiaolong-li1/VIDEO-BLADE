      
"""
    Original Author: Eric Lin (xihlin) (https://huggingface.co/microsoft/Phi-3-small-8k-instruct/blob/main/triton_flash_blocksparse_attn.py)
"""
"""
    Modified by Yizhao Gao
    Use binary block mask for simplicity. Need to be updated to varlen version for batched inference.
    Added backward propagation implementation based on FlashAttention paper and reference implementation.
"""


from typing import TypeVar
from functools import lru_cache
import math
import torch
import numpy as np

import triton
import triton.language as tl
import torch.nn.functional as F
import torch
import os


import dataclasses




def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

@triton.jit
def _fwd_kernel_inner(
    acc,
    l_i,
    m_i,
    q,
    pooling_block_idx,
    k_ptrs,
    v_ptrs,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    pooling_bias: tl.constexpr,  # log(pooling_level)
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    POOLING_BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    for inner_idx in range(POOLING_BLOCK_N // BLOCK_N):
        start_n = pooling_block_idx * POOLING_BLOCK_N + inner_idx * BLOCK_N
        # -- compute qk ----

        k = tl.load(k_ptrs + start_n * stride_kt)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
        qk += pooling_bias

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk * RCP_LN2)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # update acc
        v = tl.load(
            v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k
        )

        p = p.to(v.type.element_ty)

        acc += tl.dot(p, v)
        # update m_i and l_i
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _fwd_kernel_inner_1(
    acc,
    l_i,
    m_i,
    q,
    pooling_block_idx,
    k_ptrs,
    v_ptrs,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # POOLING_BLOCK_N
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_n = pooling_block_idx * BLOCK_N
    # -- compute qk ----

    k = tl.load(k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < seqlen_k)

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk * RCP_LN2)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    # update acc
    v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k)

    p = p.to(v.type.element_ty)

    acc += tl.dot(p, v)
    # update m_i and l_i
    m_i = m_ij
    return acc, l_i, m_i

@triton.jit
def _fwd_kernel_inner_2(
    acc,
    l_i,
    m_i,
    q,
    pooling_block_idx,
    k_ptrs,
    v_ptrs,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_n = pooling_block_idx * BLOCK_N
    # -- compute qk ----

    k = tl.load(k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < seqlen_k)

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    qk += 0.6931471805599453094  # log(2)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk * RCP_LN2)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    # update acc
    v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k)

    p = p.to(v.type.element_ty)

    acc += tl.dot(p, v)
    # update m_i and l_i
    m_i = m_ij
    return acc, l_i, m_i

@triton.jit
def _fwd_kernel_inner_4(
    acc,
    l_i,
    m_i,
    q,
    k_block_col_idx,
    k_ptrs,
    v_ptrs,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_n = k_block_col_idx * BLOCK_N
    # -- compute qk ----

    k = tl.load(k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < seqlen_k)

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    qk += 1.3862943611198906188  # log(4)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk * RCP_LN2)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    # update acc
    v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k)

    p = p.to(v.type.element_ty)

    acc += tl.dot(p, v)
    # update m_i and l_i
    m_i = m_ij
    return acc, l_i, m_i
@triton.jit
def _fwd_kernel_inner_8(
    acc,
    l_i,
    m_i,
    q,
    k_block_col_idx,
    k_ptrs,
    v_ptrs,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_n = k_block_col_idx * BLOCK_N
    # -- compute qk ----

    k = tl.load(k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < seqlen_k)

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    qk += 2.0794415416798359283  # log(8)

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk * RCP_LN2)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    # update acc
    v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k)

    p = p.to(v.type.element_ty)

    acc += tl.dot(p, v)
    # update m_i and l_i
    m_i = m_ij
    return acc, l_i, m_i
@triton.jit
def _fwd_kernel_inner_16(
    acc,
    l_i,
    m_i,
    q,
    k_block_col_idx,
    k_ptrs,
    v_ptrs,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = k_block_col_idx * BLOCK_N
    # -- compute qk ----

    if LAST_K_BLOCK:
        k = tl.load(
            k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < seqlen_k
        )
    else:
        k = tl.load(k_ptrs + start_n * stride_kt)

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    if LAST_K_BLOCK:
        qk += tl.where(offs_n[None, :] + start_n < seqlen_k, 0, -float("inf"))
    qk += 2.7725887222397812377  # log(16)

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.exp(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    # update acc
    if LAST_K_BLOCK:
        v = tl.load(
            v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k
        )
    else:
        v = tl.load(v_ptrs + start_n * stride_vt)

    p = p.to(v.type.element_ty)

    acc += tl.dot(p, v)
    # update m_i and l_i
    m_i = m_ij
    return acc, l_i, m_i

@triton.jit
def _fwd_kernel(
    Q,
    K,
    K_2,
    K_4,
    K_8,
    K_16,
    V,
    V_2,
    V_4,
    V_8,
    V_16,
    sm_scale,
    block_mask_ptr,
    Out,
    L, M,  # For backward pass: L is row-wise sum, M is row-wise max
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_kz_2,
    stride_kh_2,
    stride_kn_2,
    stride_kd_2,
    stride_kz_4,
    stride_kh_4,
    stride_kn_4,
    stride_kd_4,
    stride_kz_8,
    stride_kh_8,
    stride_kn_8,
    stride_kd_8,
    stride_kz_16,
    stride_kh_16,
    stride_kn_16,
    stride_kd_16,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_vz_2,
    stride_vh_2,
    stride_vn_2,
    stride_vd_2,
    stride_vz_4,
    stride_vh_4,
    stride_vn_4,
    stride_vd_4,
    stride_vz_8,
    stride_vh_8,
    stride_vn_8,
    stride_vd_8,
    stride_vz_16,
    stride_vh_16,
    stride_vn_16,
    stride_vd_16,
    stride_bmz,
    stride_bmh,
    stride_bmm,
    stride_bmn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_od,
    H,
    N_CTX,
    N_CTX_2,
    N_CTX_4,
    N_CTX_8,
    N_CTX_16,
    INFERENCE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    POOLING_BLOCK_N: tl.constexpr,
    POOLING_BLOCK_N_2: tl.constexpr,
    POOLING_BLOCK_N_4: tl.constexpr,
    POOLING_BLOCK_N_8: tl.constexpr,
    POOLING_BLOCK_N_16: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    LOG_1 = 0.0
    LOG_2 = 0.6931471805599453094  # log(2)
    LOG_4 = 1.3862943611198906188  # log(4)
    LOG_8 = 2.0794415416798359283  # log(8)

    Q_LEN = N_CTX
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    off_z = off_hz // H
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    K_2 += off_z * stride_kz_2 + off_h * stride_kh_2
    K_4 += off_z * stride_kz_4 + off_h * stride_kh_4
    K_8 += off_z * stride_kz_8 + off_h * stride_kh_8
    K_16 += off_z * stride_kz_16 + off_h * stride_kh_16
    V += off_z * stride_vz + off_h * stride_vh
    V_2 += off_z * stride_vz_2 + off_h * stride_vh_2
    V_4 += off_z * stride_vz_4 + off_h * stride_vh_4
    V_8 += off_z * stride_vz_8 + off_h * stride_vh_8
    V_16 += off_z * stride_vz_16 + off_h * stride_vh_16
    block_mask_ptr += off_z * stride_bmz + off_h * stride_bmh

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_n_1 = tl.arange(0, POOLING_BLOCK_N)
    offs_n_2 = tl.arange(0, POOLING_BLOCK_N_2)
    offs_n_4 = tl.arange(0, POOLING_BLOCK_N_4)
    offs_n_8 = tl.arange(0, POOLING_BLOCK_N_8)
    offs_n_16 = tl.arange(0, POOLING_BLOCK_N_16)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd

    if POOLING_BLOCK_N < BLOCK_N:
        off_k_1 = offs_n_1[None, :] * stride_kn + offs_d[:, None] * stride_kd
        off_v_1 = offs_n_1[:, None] * stride_vn + offs_d[None, :] * stride_vd
    else:
        off_k_1 = offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        off_v_1 = offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    if POOLING_BLOCK_N_2 < BLOCK_N:
        off_k_2 = offs_n_2[None, :] * stride_kn_2 + offs_d[:, None] * stride_kd_2
        off_v_2 = offs_n_2[:, None] * stride_vn_2 + offs_d[None, :] * stride_vd_2
    else:
        off_k_2 = offs_n[None, :] * stride_kn_2 + offs_d[:, None] * stride_kd_2
        off_v_2 = offs_n[:, None] * stride_vn_2 + offs_d[None, :] * stride_vd_2
    if POOLING_BLOCK_N_4 < BLOCK_N:
        off_k_4 = offs_n_4[None, :] * stride_kn_4 + offs_d[:, None] * stride_kd_4
        off_v_4 = offs_n_4[:, None] * stride_vn_4 + offs_d[None, :] * stride_vd_4
    else:
        off_k_4 = offs_n[None, :] * stride_kn_4 + offs_d[:, None] * stride_kd_4
        off_v_4 = offs_n[:, None] * stride_vn_4 + offs_d[None, :] * stride_vd_4
    if POOLING_BLOCK_N_8 < BLOCK_N:
        off_k_8 = offs_n_8[None, :] * stride_kn_8 + offs_d[:, None] * stride_kd_8
        off_v_8 = offs_n_8[:, None] * stride_vn_8 + offs_d[None, :] * stride_vd_8
    else:
        off_k_8 = offs_n[None, :] * stride_kn_8 + offs_d[:, None] * stride_kd_8
        off_v_8 = offs_n[:, None] * stride_vn_8 + offs_d[None, :] * stride_vd_8

    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs_1 = K + off_k_1
    k_ptrs_2 = K_2 + off_k_2
    k_ptrs_4 = K_4 + off_k_4
    k_ptrs_8 = K_8 + off_k_8
    # k_ptrs_16 = K_16 + off_k_16
    v_ptrs_1 = V + off_v_1
    v_ptrs_2 = V_2 + off_v_2
    v_ptrs_4 = V_4 + off_v_4
    v_ptrs_8 = V_8 + off_v_8
    # v_ptrs_16 = V_16 + off_v_16
    mask_ptrs = block_mask_ptr + start_m * stride_bmm

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN)

    pooling_block_start = 0
    # k_block_end = tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N)
    pooling_block_end = tl.cdiv(N_CTX, POOLING_BLOCK_N)  # seq_len_k 是 K 的序列长度

    # loop over k, v and update accumulator
    for pb_idx in range(pooling_block_start, pooling_block_end):
        mask = tl.load(mask_ptrs + pb_idx * stride_bmn)

        # Check if this is the last block
        is_last_block = pb_idx == pooling_block_end - 1

        if mask > 0:
            if mask < 4:
                if mask == 1:
                    if POOLING_BLOCK_N <= BLOCK_N:
                        acc, l_i, m_i = _fwd_kernel_inner_1(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_1,
                            v_ptrs_1,
                            offs_n_1,
                            stride_kn,
                            stride_vn,
                            sm_scale,
                            N_CTX,
                            is_last_block,
                            BLOCK_M,
                            POOLING_BLOCK_N,
                        )
                    else:
                        acc, l_i, m_i = _fwd_kernel_inner(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_1,
                            v_ptrs_1,
                            offs_n,
                            stride_kn,
                            stride_vn,
                            sm_scale,
                            N_CTX,
                            LOG_1,
                            is_last_block,
                            BLOCK_M,
                            BLOCK_N,
                            POOLING_BLOCK_N,
                        )
                if mask == 2:
                    if POOLING_BLOCK_N_2 <= BLOCK_N:
                        acc, l_i, m_i = _fwd_kernel_inner_2(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_2,
                            v_ptrs_2,
                            offs_n_2,
                            stride_kn_2,
                            stride_vn_2,
                            sm_scale,
                            N_CTX_2,
                            is_last_block,
                            BLOCK_M,
                            POOLING_BLOCK_N_2,
                        )
                    else:
                        acc, l_i, m_i = _fwd_kernel_inner(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_2,
                            v_ptrs_2,
                            offs_n,
                            stride_kn_2,
                            stride_vn_2,
                            sm_scale,
                            N_CTX_2,
                            LOG_2,
                            is_last_block,
                            BLOCK_M,
                            BLOCK_N,
                            POOLING_BLOCK_N_2,
                        )
            else:
                if mask == 8:
                    if POOLING_BLOCK_N_8 <= BLOCK_N:
                        acc, l_i, m_i = _fwd_kernel_inner_8(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_8,
                            v_ptrs_8,
                            offs_n_8,
                            stride_kn_8,
                            stride_vn_8,
                            sm_scale,
                            N_CTX_8,
                            is_last_block,
                            BLOCK_M,
                            POOLING_BLOCK_N_8,
                        )
                    else:
                        acc, l_i, m_i = _fwd_kernel_inner(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_8,
                            v_ptrs_8,
                            offs_n,
                            stride_kn_8,
                            stride_vn_8,
                            sm_scale,
                            N_CTX_8,
                            LOG_8,
                            is_last_block,
                            BLOCK_M,
                            BLOCK_N,
                            POOLING_BLOCK_N_8,
                        )
                if mask == 4:
                    if POOLING_BLOCK_N_4 <= BLOCK_N:
                        acc, l_i, m_i = _fwd_kernel_inner_4(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_4,
                            v_ptrs_4,
                            offs_n_4,
                            stride_kn_4,
                            stride_vn_4,
                            sm_scale,
                            N_CTX_4,
                            is_last_block,
                            BLOCK_M,
                            POOLING_BLOCK_N_4,
                        )
                    else:
                        acc, l_i, m_i = _fwd_kernel_inner(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_4,
                            v_ptrs_4,
                            offs_n,
                            stride_kn_4,
                            stride_vn_4,
                            sm_scale,
                            N_CTX_4,
                            LOG_4,
                            is_last_block,
                            BLOCK_M,
                            BLOCK_N,
                            POOLING_BLOCK_N_4,
                        )

    # Store L and M for backward pass
    if not INFERENCE:
        l_ptrs = L + off_hz * N_CTX + offs_m
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(l_ptrs, l_i, mask=offs_m < Q_LEN)
        tl.store(m_ptrs, m_i, mask=offs_m < Q_LEN)

    m_i += tl.math.log(l_i)
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    acc = acc.to(Out.dtype.element_ty)

    off_o = (
        off_z * stride_oz
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < Q_LEN)

# Backward preprocessing kernel
@triton.jit
def _bwd_preprocess(
    Out, DO, L,
    NewDO, Delta,
    N_CTX, H,
    stride_oz, stride_oh, stride_om, stride_od,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Offset pointers for batch/head
    Out += off_z * stride_oz + off_h * stride_oh
    DO += off_z * stride_oz + off_h * stride_oh
    NewDO += off_z * stride_oz + off_h * stride_oh
    
    # Process sequence blocks within this batch/head
    for block_idx in range(tl.cdiv(N_CTX, BLOCK_M)):
        off_m = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        off_d = tl.arange(0, D_HEAD)
        
        # load
        o = tl.load(Out + off_m[:, None] * stride_om + off_d[None, :] * stride_od, mask=off_m[:, None] < N_CTX).to(tl.float32)
        do = tl.load(DO + off_m[:, None] * stride_om + off_d[None, :] * stride_od, mask=off_m[:, None] < N_CTX).to(tl.float32)
        denom = tl.load(L + off_hz * N_CTX + off_m, mask=off_m < N_CTX).to(tl.float32)
        
        # compute
        do = do / denom[:, None]
        delta = tl.sum(o * do, axis=1)
        
        # write-back
        tl.store(NewDO + off_m[:, None] * stride_om + off_d[None, :] * stride_od, do, mask=off_m[:, None] < N_CTX)
        tl.store(Delta + off_hz * N_CTX + off_m, delta, mask=off_m < N_CTX)

# Small reusable backward kernel for processing one KV sub-block against all Q blocks
@triton.jit
def _bwd_kernel_small_32(
    Q, K, V, sm_scale, DO, DQ, DK, DV,
    block_mask_ptr, D_ptrs, m_ptrs,
    offs_n_sub, offs_d,
    stride_qm, stride_qd, stride_om, stride_od,
    stride_kn, stride_kd, stride_vn, stride_vd,
    stride_bmm, stride_bmn,
    N_CTX, start_n_block,
    BLOCK_M: tl.constexpr,
    BLOCK_N_SMALL: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr,
):
    # Load K and V for this sub-block (fixed for all Q iterations)
    k_ptrs_sub = K + (offs_n_sub[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    v_ptrs_sub = V + (offs_n_sub[:, None] * stride_vn + offs_d[None, :] * stride_vd)
    k_sub = tl.load(k_ptrs_sub, mask=offs_n_sub[:, None] < N_CTX)
    v_sub = tl.load(v_ptrs_sub, mask=offs_n_sub[:, None] < N_CTX)
    
    # Initialize accumulators for this KV sub-block
    dv_sub = tl.zeros([BLOCK_N_SMALL, BLOCK_DMODEL], dtype=tl.float32)
    dk_sub = tl.zeros([BLOCK_N_SMALL, BLOCK_DMODEL], dtype=tl.float32)
    
    # Check if this is the last K sub-block
    is_last_k_block = tl.max(offs_n_sub) >= N_CTX - 1
    
    # Loop over all Q row blocks (each of size BLOCK_M)
    offs_m = tl.arange(0, BLOCK_M)
    m_block_end = tl.cdiv(N_CTX, BLOCK_M)
    
    for row_idx in range(m_block_end):
        # Check mask for this Q row block and KV column block
        mask_ptr = block_mask_ptr + row_idx * stride_bmm + start_n_block * stride_bmn
        mask = tl.load(mask_ptr)
        
        # Only process if mask is 1 (full resolution)
        if mask == 1:
            start_m = row_idx * BLOCK_M
            offs_m_curr = start_m + offs_m
            
            # Load Q and DO for this row block
            q_ptrs = Q + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)
            do_ptrs = DO + (offs_m_curr[:, None] * stride_om + offs_d[None, :] * stride_od)
            dq_ptrs = DQ + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)
            
            valid_mask = offs_m_curr < N_CTX
            q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < N_CTX)
            do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < N_CTX)
            m = tl.load(m_ptrs + offs_m_curr, mask=valid_mask)
            Di = tl.load(D_ptrs + offs_m_curr, mask=valid_mask)
            
            # Compute attention scores
            qk = tl.dot(q.to(tl.float32), tl.trans(k_sub.to(tl.float32)))
            qk *= sm_scale
            
            # Apply boundary mask for last block
            if is_last_k_block:
                qk += tl.where(offs_n_sub[None, :] < N_CTX, 0, -float("inf"))
            
            p = tl.exp(qk - m[:, None])
            
            # Accumulate gradients for this KV sub-block
            dv_sub += tl.dot(tl.trans(p), do.to(tl.float32))
            
            dp = tl.zeros([BLOCK_M, BLOCK_N_SMALL], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do.to(tl.float32), tl.trans(v_sub.to(tl.float32)))
            
            ds = (p * dp) * sm_scale
            dk_sub += tl.dot(tl.trans(ds), q.to(tl.float32))
            
            # Compute and accumulate dq for this Q block
            dq = tl.dot(ds, k_sub.to(tl.float32))
            tl.atomic_add(dq_ptrs, dq, mask=valid_mask[:, None])
    
    # Write-back DK and DV for this KV sub-block
    dv_ptrs = DV + (offs_n_sub[:, None] * stride_vn + offs_d[None, :] * stride_vd)
    dk_ptrs = DK + (offs_n_sub[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    tl.store(dv_ptrs, dv_sub.to(DV.dtype.element_ty), mask=offs_n_sub[:, None] < N_CTX)
    tl.store(dk_ptrs, dk_sub.to(DK.dtype.element_ty), mask=offs_n_sub[:, None] < N_CTX)

# Backward kernel for mask=1 (full resolution) - calls small kernel 4 times for KV sub-blocks
@triton.jit
def _bwd_kernel_mask_1(
    Q, K, V, sm_scale,
    block_mask_ptr,
    Out, DO,
    DQ, DK, DV,
    L, M, D,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_bmz, stride_bmh, stride_bmm, stride_bmn,
    stride_oz, stride_oh, stride_om, stride_od,
    H, N_CTX,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N_SMALL: tl.constexpr
):
    start_n = tl.program_id(0)  # Which KV column block group
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_oz + off_h * stride_oh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_kz + off_h * stride_kh
    DV += off_z * stride_vz + off_h * stride_vh
    block_mask_ptr += off_z * stride_bmz + off_h * stride_bmh

    # Constants for sub-block processing
    
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Create base offsets for each sub-block (constexpr)
    offs_n_base = tl.arange(0, BLOCK_N_SMALL)
    
    # Pointer to row-wise quantities
    D_ptrs = D + off_hz * N_CTX
    m_ptrs = M + off_hz * N_CTX
    
    # Calculate base starting position for this column block
    n_start_base = start_n * BLOCK_N
    
    # Call small kernel 4 times for 4 KV sub-blocks
    # Sub-block 0: [0, 32)
    offs_n_sub_0 = n_start_base + offs_n_base
    _bwd_kernel_small_32(
        Q, K, V, sm_scale, DO, DQ, DK, DV,
        block_mask_ptr, D_ptrs, m_ptrs,
        offs_n_sub_0, offs_d,
        stride_qm, stride_qd, stride_om, stride_od,
        stride_kn, stride_kd, stride_vn, stride_vd,
        stride_bmm, stride_bmn,
        N_CTX, start_n,
        BLOCK_M, BLOCK_N_SMALL, BLOCK_DMODEL
    )
    
    # Sub-block 1: [32, 64)
    offs_n_sub_1 = n_start_base + BLOCK_N_SMALL + offs_n_base
    _bwd_kernel_small_32(
        Q, K, V, sm_scale, DO, DQ, DK, DV,
        block_mask_ptr, D_ptrs, m_ptrs,
        offs_n_sub_1, offs_d,
        stride_qm, stride_qd, stride_om, stride_od,
        stride_kn, stride_kd, stride_vn, stride_vd,
        stride_bmm, stride_bmn,
        N_CTX, start_n,
        BLOCK_M, BLOCK_N_SMALL, BLOCK_DMODEL
    )
    
    # Sub-block 2: [64, 96)
    offs_n_sub_2 = n_start_base + 2 * BLOCK_N_SMALL + offs_n_base
    _bwd_kernel_small_32(
        Q, K, V, sm_scale, DO, DQ, DK, DV,
        block_mask_ptr, D_ptrs, m_ptrs,
        offs_n_sub_2, offs_d,
        stride_qm, stride_qd, stride_om, stride_od,
        stride_kn, stride_kd, stride_vn, stride_vd,
        stride_bmm, stride_bmn,
        N_CTX, start_n,
        BLOCK_M, BLOCK_N_SMALL, BLOCK_DMODEL
    )
    
    # Sub-block 3: [96, 128)
    offs_n_sub_3 = n_start_base + 3 * BLOCK_N_SMALL + offs_n_base
    _bwd_kernel_small_32(
        Q, K, V, sm_scale, DO, DQ, DK, DV,
        block_mask_ptr, D_ptrs, m_ptrs,
        offs_n_sub_3, offs_d,
        stride_qm, stride_qd, stride_om, stride_od,
        stride_kn, stride_kd, stride_vn, stride_vd,
        stride_bmm, stride_bmn,
        N_CTX, start_n,
        BLOCK_M, BLOCK_N_SMALL, BLOCK_DMODEL
    )

# Backward kernel for mask=2 (2x downsampled)
@triton.jit
def _bwd_kernel_mask_2(
    Q, K_2, V_2, sm_scale,
    block_mask_ptr,
    Out, DO,
    DQ, DK_2, DV_2,
    L, M, D,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz_2, stride_kh_2, stride_kn_2, stride_kd_2,
    stride_vz_2, stride_vh_2, stride_vn_2, stride_vd_2,
    stride_bmz, stride_bmh, stride_bmm, stride_bmn,
    stride_oz, stride_oh, stride_om, stride_od,
    H, N_CTX, N_CTX_2,
    BLOCK_M: tl.constexpr, 
    BLOCK_N_2: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K_2 += off_z * stride_kz_2 + off_h * stride_kh_2
    V_2 += off_z * stride_vz_2 + off_h * stride_vh_2
    DO += off_z * stride_oz + off_h * stride_oh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK_2 += off_z * stride_kz_2 + off_h * stride_kh_2
    DV_2 += off_z * stride_vz_2 + off_h * stride_vh_2
    block_mask_ptr += off_z * stride_bmz + off_h * stride_bmh

    offs_n_2 = start_n * BLOCK_N_2 + tl.arange(0, BLOCK_N_2)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # initialize pointers to value-like data
    k_ptrs_2 = K_2 + (offs_n_2[:, None] * stride_kn_2 + offs_d[None, :] * stride_kd_2)
    v_ptrs_2 = V_2 + (offs_n_2[:, None] * stride_vn_2 + offs_d[None, :] * stride_vd_2)
    
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    m_ptrs = M + off_hz * N_CTX
    
    # initialize dv and dk with float32 accumulator
    dv_2 = tl.zeros([BLOCK_N_2, BLOCK_DMODEL], dtype=tl.float32)
    dk_2 = tl.zeros([BLOCK_N_2, BLOCK_DMODEL], dtype=tl.float32)
    
    # k and v stay in SRAM throughout
    k_2 = tl.load(k_ptrs_2, mask=offs_n_2[:, None] < N_CTX_2)
    v_2 = tl.load(v_ptrs_2, mask=offs_n_2[:, None] < N_CTX_2)

    # loop over rows
    m_block_start = 0
    m_block_end = tl.cdiv(N_CTX,BLOCK_M) # Fixed upper bound to avoid dynamic range issues

    for row_idx in range(m_block_start, m_block_end):
        # Only process if within actual sequence length
        mask_ptr = block_mask_ptr + row_idx * stride_bmm + start_n * stride_bmn
        mask = tl.load(mask_ptr)
        
        # Only process if mask is 2 (2x downsampled)
        if mask == 2:
            start_m = row_idx * BLOCK_M
            offs_m_curr = start_m + offs_m
            q_ptrs = Q + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)
            do_ptrs = DO + (offs_m_curr[:, None] * stride_om + offs_d[None, :] * stride_od)
            dq_ptrs = DQ + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)

            # load q, do on-chip
            # Add boundary check for offs_m_curr to prevent out-of-bounds access
            valid_mask = offs_m_curr < N_CTX
            q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < N_CTX)
            do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < N_CTX)
            m = tl.load(m_ptrs + offs_m_curr, mask=valid_mask)
            Di = tl.load(D_ptrs + offs_m_curr, mask=valid_mask)

            # Check if this is the last K block for 2x downsampled
            is_last_k_block_2 = (start_n + 1) * BLOCK_N_2 >= N_CTX_2
            
            # Similar logic for 2x downsampled K,V with log(2) offset
            qk = tl.dot(q.to(tl.float32), tl.trans(k_2.to(tl.float32)))
            qk *= sm_scale
            qk += 0.6931471805599453094  # log(2)
            
            # Apply boundary mask for last block
            if is_last_k_block_2:
                qk += tl.where(offs_n_2[None, :] < N_CTX_2, 0, -float("inf"))
            
            p = tl.exp(qk - m[:, None])
            
            dv_2 += tl.dot(tl.trans(p), do.to(tl.float32))
            
            dp = tl.zeros([BLOCK_M, BLOCK_N_2], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do.to(tl.float32), tl.trans(v_2.to(tl.float32)))
            
            ds = (p * dp) * sm_scale
            dk_2 += tl.dot(tl.trans(ds), q.to(tl.float32))
            
            dq = tl.dot(ds, k_2.to(tl.float32))
            tl.atomic_add(dq_ptrs, dq, mask=valid_mask[:, None])

    # write-back with dtype conversion to match expected output types
    dv_ptrs_2 = DV_2 + (offs_n_2[:, None] * stride_vn_2 + offs_d[None, :] * stride_vd_2)
    dk_ptrs_2 = DK_2 + (offs_n_2[:, None] * stride_kn_2 + offs_d[None, :] * stride_kd_2)
    tl.store(dv_ptrs_2, dv_2.to(DV_2.dtype.element_ty), mask=offs_n_2[:, None] < N_CTX_2)
    tl.store(dk_ptrs_2, dk_2.to(DK_2.dtype.element_ty), mask=offs_n_2[:, None] < N_CTX_2)

# Backward kernel for mask=4 (4x downsampled)
@triton.jit
def _bwd_kernel_mask_4(
    Q, K_4, V_4, sm_scale,
    block_mask_ptr,
    Out, DO,
    DQ, DK_4, DV_4,
    L, M, D,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz_4, stride_kh_4, stride_kn_4, stride_kd_4,
    stride_vz_4, stride_vh_4, stride_vn_4, stride_vd_4,
    stride_bmz, stride_bmh, stride_bmm, stride_bmn,
    stride_oz, stride_oh, stride_om, stride_od,
    H, N_CTX, N_CTX_4,
    BLOCK_M: tl.constexpr, 
    BLOCK_N_4: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K_4 += off_z * stride_kz_4 + off_h * stride_kh_4
    V_4 += off_z * stride_vz_4 + off_h * stride_vh_4
    DO += off_z * stride_oz + off_h * stride_oh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK_4 += off_z * stride_kz_4 + off_h * stride_kh_4
    DV_4 += off_z * stride_vz_4 + off_h * stride_vh_4
    block_mask_ptr += off_z * stride_bmz + off_h * stride_bmh

    offs_n_4 = start_n * BLOCK_N_4 + tl.arange(0, BLOCK_N_4)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # initialize pointers to value-like data
    k_ptrs_4 = K_4 + (offs_n_4[:, None] * stride_kn_4 + offs_d[None, :] * stride_kd_4)
    v_ptrs_4 = V_4 + (offs_n_4[:, None] * stride_vn_4 + offs_d[None, :] * stride_vd_4)
    
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    m_ptrs = M + off_hz * N_CTX
    
    # initialize dv and dk with float32 accumulator
    dv_4 = tl.zeros([BLOCK_N_4, BLOCK_DMODEL], dtype=tl.float32)
    dk_4 = tl.zeros([BLOCK_N_4, BLOCK_DMODEL], dtype=tl.float32)
    
    # k and v stay in SRAM throughout
    k_4 = tl.load(k_ptrs_4, mask=offs_n_4[:, None] < N_CTX_4)
    v_4 = tl.load(v_ptrs_4, mask=offs_n_4[:, None] < N_CTX_4)

    # loop over rows
    m_block_start = 0
    m_block_end = tl.cdiv(N_CTX,BLOCK_M) # Fixed upper bound to avoid dynamic range issues

    for row_idx in range(m_block_start, m_block_end):
        # Only process if within actual sequence length
        mask_ptr = block_mask_ptr + row_idx * stride_bmm + start_n * stride_bmn
        mask = tl.load(mask_ptr)
        
        # Only process if mask is 4 (4x downsampled)
        if mask == 4:
            start_m = row_idx * BLOCK_M
            offs_m_curr = start_m + offs_m
            q_ptrs = Q + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)
            do_ptrs = DO + (offs_m_curr[:, None] * stride_om + offs_d[None, :] * stride_od)
            dq_ptrs = DQ + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)

            # load q, do on-chip
            # Add boundary check for offs_m_curr to prevent out-of-bounds access
            valid_mask = offs_m_curr < N_CTX
            q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < N_CTX)
            do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < N_CTX)
            m = tl.load(m_ptrs + offs_m_curr, mask=valid_mask)
            Di = tl.load(D_ptrs + offs_m_curr, mask=valid_mask)

            # Check if this is the last K block for 4x downsampled
            is_last_k_block_4 = (start_n + 1) * BLOCK_N_4 >= N_CTX_4
            
            # Similar logic for 4x downsampled K,V with log(4) offset
            qk = tl.dot(q.to(tl.float32), tl.trans(k_4.to(tl.float32)))
            qk *= sm_scale
            qk += 1.3862943611198906188  # log(4)
            
            # Apply boundary mask for last block
            if is_last_k_block_4:
                qk += tl.where(offs_n_4[None, :]< N_CTX_4, 0, -float("inf"))
            
            p = tl.exp(qk - m[:, None])
            
            dv_4 += tl.dot(tl.trans(p), do.to(tl.float32))
            
            dp = tl.zeros([BLOCK_M, BLOCK_N_4], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do.to(tl.float32), tl.trans(v_4.to(tl.float32)))
            
            ds = (p * dp) * sm_scale
            dk_4 += tl.dot(tl.trans(ds), q.to(tl.float32))
            
            dq = tl.dot(ds, k_4.to(tl.float32))
            tl.atomic_add(dq_ptrs, dq, mask=valid_mask[:, None])

    # write-back with dtype conversion to match expected output types
    dv_ptrs_4 = DV_4 + (offs_n_4[:, None] * stride_vn_4 + offs_d[None, :] * stride_vd_4)
    dk_ptrs_4 = DK_4 + (offs_n_4[:, None] * stride_kn_4 + offs_d[None, :] * stride_kd_4)
    tl.store(dv_ptrs_4, dv_4.to(DV_4.dtype.element_ty), mask=offs_n_4[:, None] < N_CTX_4)
    tl.store(dk_ptrs_4, dk_4.to(DK_4.dtype.element_ty), mask=offs_n_4[:, None] < N_CTX_4)

# Backward kernel for mask=8 (8x downsampled)
@triton.jit
def _bwd_kernel_mask_8(
    Q, K_8, V_8, sm_scale,
    block_mask_ptr,
    Out, DO,
    DQ, DK_8, DV_8,
    L, M, D,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz_8, stride_kh_8, stride_kn_8, stride_kd_8,
    stride_vz_8, stride_vh_8, stride_vn_8, stride_vd_8,
    stride_bmz, stride_bmh, stride_bmm, stride_bmn,
    stride_oz, stride_oh, stride_om, stride_od,
    H, N_CTX, N_CTX_8,
    BLOCK_M: tl.constexpr, 
    BLOCK_N_8: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K_8 += off_z * stride_kz_8 + off_h * stride_kh_8
    V_8 += off_z * stride_vz_8 + off_h * stride_vh_8
    DO += off_z * stride_oz + off_h * stride_oh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK_8 += off_z * stride_kz_8 + off_h * stride_kh_8
    DV_8 += off_z * stride_vz_8 + off_h * stride_vh_8
    block_mask_ptr += off_z * stride_bmz + off_h * stride_bmh

    offs_n_8 = start_n * BLOCK_N_8 + tl.arange(0, BLOCK_N_8)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # initialize pointers to value-like data
    k_ptrs_8 = K_8 + (offs_n_8[:, None] * stride_kn_8 + offs_d[None, :] * stride_kd_8)
    v_ptrs_8 = V_8 + (offs_n_8[:, None] * stride_vn_8 + offs_d[None, :] * stride_vd_8)
    
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    m_ptrs = M + off_hz * N_CTX
    
    # initialize dv and dk with float32 accumulator
    dv_8 = tl.zeros([BLOCK_N_8, BLOCK_DMODEL], dtype=tl.float32)
    dk_8 = tl.zeros([BLOCK_N_8, BLOCK_DMODEL], dtype=tl.float32)
    
    # k and v stay in SRAM throughout
    k_8 = tl.load(k_ptrs_8, mask=offs_n_8[:, None] < N_CTX_8)
    v_8 = tl.load(v_ptrs_8, mask=offs_n_8[:, None] < N_CTX_8)

    # loop over rows
    m_block_start = 0
    m_block_end = tl.cdiv(N_CTX,BLOCK_M) # Fixed upper bound to avoid dynamic range issues

    for row_idx in range(m_block_start, m_block_end):
        # Only process if within actual sequence length
        mask_ptr = block_mask_ptr + row_idx * stride_bmm + start_n * stride_bmn
        mask = tl.load(mask_ptr)
        
        # Only process if mask is 8 (8x downsampled)
        if mask == 8:
            start_m = row_idx * BLOCK_M
            offs_m_curr = start_m + offs_m
            q_ptrs = Q + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)
            do_ptrs = DO + (offs_m_curr[:, None] * stride_om + offs_d[None, :] * stride_od)
            dq_ptrs = DQ + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)

            # load q, do on-chip
            # Add boundary check for offs_m_curr to prevent out-of-bounds access
            valid_mask = offs_m_curr < N_CTX
            q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < N_CTX)
            do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < N_CTX)
            m = tl.load(m_ptrs + offs_m_curr, mask=valid_mask)
            Di = tl.load(D_ptrs + offs_m_curr, mask=valid_mask)

            # Check if this is the last K block for 8x downsampled
            is_last_k_block_8 = (start_n + 1) * BLOCK_N_8 >= N_CTX_8
            
            # Similar logic for 8x downsampled K,V with log(8) offset
            qk = tl.dot(q.to(tl.float32), tl.trans(k_8.to(tl.float32)))
            qk *= sm_scale
            qk += 2.0794415416798359283  # log(8)
            
            # Apply boundary mask for last block
            if is_last_k_block_8:
                qk += tl.where(offs_n_8[None, :]  < N_CTX_8, 0, -float("inf"))
            
            p = tl.exp(qk - m[:, None])
            
            dv_8 += tl.dot(tl.trans(p), do.to(tl.float32))
            
            dp = tl.zeros([BLOCK_M, BLOCK_N_8], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do.to(tl.float32), tl.trans(v_8.to(tl.float32)))
            
            ds = (p * dp) * sm_scale
            dk_8 += tl.dot(tl.trans(ds), q.to(tl.float32))
            
            dq = tl.dot(ds, k_8.to(tl.float32))
            tl.atomic_add(dq_ptrs, dq, mask=valid_mask[:, None])

    # write-back with dtype conversion to match expected output types
    dv_ptrs_8 = DV_8 + (offs_n_8[:, None] * stride_vn_8 + offs_d[None, :] * stride_vd_8)
    dk_ptrs_8 = DK_8 + (offs_n_8[:, None] * stride_kn_8 + offs_d[None, :] * stride_kd_8)
    tl.store(dv_ptrs_8, dv_8.to(DV_8.dtype.element_ty), mask=offs_n_8[:, None] < N_CTX_8)
    tl.store(dk_ptrs_8, dk_8.to(DK_8.dtype.element_ty), mask=offs_n_8[:, None] < N_CTX_8)

def pad_to_multiple(x, multiple):
    """
    在序列维度（dim=2）上填充 x，使其长度为 multiple 的倍数。
    x: [B, H, L, D]
    """
    L = x.size(2)
    remainder = L % multiple
    if remainder != 0:
        pad_len = multiple - remainder
        # 对序列维度在后面补 pad_len 个零（注意 F.pad 参数顺序：最后两个数字对应 dim=2 的左右补充）
        x = F.pad(x, (0, 0, 0, pad_len),mode='replicate')
    return x

def pooling(x, zoom_ratio):
    """
    对输入张量 x 进行池化操作，使用平均池化。
    x: [B, H, L, D]
    zoom_ratio: 池化的缩放比例
    """
    B, H, L, D = x.shape
    
    # 确保序列长度能被zoom_ratio整除，否则pad
    remainder = L % zoom_ratio
    if remainder != 0:
        pad_len = zoom_ratio - remainder
        # 在序列维度后面补充，使用replicate模式
        x = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
        L = x.shape[2]  # 更新长度
    
    # 使用平均池化
    x = torch.mean(x.view(B, H, -1, zoom_ratio, D), dim=3)
    return x

def _forward(
    ctx,
    q,
    k,
    v,
    block_sparse_mask,
    sm_scale,
    BLOCK_M=64,
    BLOCK_N=64,
    POOLING_BLOCK_N=128,
    num_warps=None,
    num_stages=1,
    out=None,
):


    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert k.shape[2] == v.shape[2]
    o = out if out is not None else torch.empty_like(q).contiguous()
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1])

    assert q.shape[-1] in [64, 128]
    BLOCK_DMODEL = q.shape[-1]

    if is_hip():
        num_warps, num_stages = 8, 1
    else:
        num_warps, num_stages = 4, 2

    # Check if this is for training (gradient required)
    inference = (not q.requires_grad) and (not k.requires_grad) and (not v.requires_grad)
    H = q.shape[1]

    k_padding = pad_to_multiple(k, POOLING_BLOCK_N)
    v_padding = pad_to_multiple(v, POOLING_BLOCK_N)
    k_2 = pooling(k_padding, 2)
    v_2 = pooling(v_padding, 2)
    k_4 = pooling(k_2, 2)
    v_4 = pooling(v_2, 2)
    k_8 = pooling(k_4, 2)
    v_8 = pooling(v_4, 2)
    k_16 = pooling(k_8, 2)
    v_16 = pooling(v_8, 2)
    N_CTX = k.shape[2]
    N_CTX_2 = k_2.shape[2]
    N_CTX_4 = k_4.shape[2]
    N_CTX_8 = k_8.shape[2]
    N_CTX_16 = k_16.shape[2]
    #print(f"N_CTX: {N_CTX}, N_CTX_2: {N_CTX_2}, N_CTX_4: {N_CTX_4}, N_CTX_8: {N_CTX_8}, N_CTX_16: {N_CTX_16}")
    # Allocate L and M for backward pass if not inference
    if inference:
        L = m = None
    else:
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    with torch.cuda.device(q.device.index): 
        _fwd_kernel[grid](
            q, k, k_2, k_4, k_8, k_16, v, v_2, v_4, v_8, v_16, sm_scale,
            block_sparse_mask,
            o,
            L, m,  # Add L and M for backward pass
            *q.stride(), 
            *k.stride(), 
            *k_2.stride(),
            *k_4.stride(),
            *k_8.stride(),
            *k_16.stride(),
            *v.stride(), 
            *v_2.stride(),
            *v_4.stride(),
            *v_8.stride(),
            *v_16.stride(),
            *block_sparse_mask.stride(), 
            *o.stride(),
            H, N_CTX, N_CTX_2, N_CTX_4, N_CTX_8, N_CTX_16,
            inference,
            BLOCK_M,
            BLOCK_N,
            POOLING_BLOCK_N,
            POOLING_BLOCK_N // 2,  # POOLING_BLOCK_N_2
            POOLING_BLOCK_N // 4,  # POOLING_BLOCK_N_4
            POOLING_BLOCK_N // 8,  # POOLING_BLOCK_N_8
            POOLING_BLOCK_N // 16,  # POOLING_BLOCK_N_16
            BLOCK_DMODEL,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    # Save for backward
    if not inference:
        ctx.save_for_backward(q, k, k_2, k_4, k_8, k_16, v, v_2, v_4, v_8, v_16, o, L, m, block_sparse_mask)
        ctx.sm_scale = sm_scale
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.POOLING_BLOCK_N = POOLING_BLOCK_N
        ctx.BLOCK_DMODEL = BLOCK_DMODEL
        ctx.grid = grid
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages

    return o

def _backward(ctx, do):
    q, k, k_2, k_4, k_8, k_16, v, v_2, v_4, v_8, v_16, o, L, m, block_sparse_mask = ctx.saved_tensors

    if not do.is_contiguous():
        do = do.contiguous()
        
    if not o.is_contiguous():
        raise ValueError(f'output is not contiguous: {o.stride()=}')

    # Store original k,v sizes before any padding was applied
    original_k_size = k.shape[2]  # This is N_CTX from forward pass
    original_v_size = v.shape[2]  # Should be same as k

    # Create gradient tensors with same dtype and device as do
    dq = torch.zeros_like(q, dtype=torch.float32, device=do.device)  # Use float32 for dq accumulation
    dk = torch.zeros_like(k, dtype=do.dtype, device=do.device)
    dk_2 = torch.zeros_like(k_2, dtype=do.dtype, device=do.device)
    dk_4 = torch.zeros_like(k_4, dtype=do.dtype, device=do.device)
    dk_8 = torch.zeros_like(k_8, dtype=do.dtype, device=do.device)
    dv = torch.zeros_like(v, dtype=do.dtype, device=do.device)
    dv_2 = torch.zeros_like(v_2, dtype=do.dtype, device=do.device)
    dv_4 = torch.zeros_like(v_4, dtype=do.dtype, device=do.device)
    dv_8 = torch.zeros_like(v_8, dtype=do.dtype, device=do.device)
    do_scaled = torch.empty_like(do)
    delta = torch.empty_like(L)

    assert o.stride() == do.stride() == do_scaled.stride()

    # Preprocessing: compute do_scaled and delta
    grid_preprocess = (q.shape[0] * q.shape[1],)
    _bwd_preprocess[grid_preprocess](
        o, do, L,
        do_scaled, delta,
        q.shape[2], q.shape[1],  # N_CTX, H
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        BLOCK_M=ctx.BLOCK_M, D_HEAD=q.shape[-1],
    )

    # Call 4 separate backward kernels for different mask cases
    N_CTX = k.shape[2]
    N_CTX_2 = k_2.shape[2]
    N_CTX_4 = k_4.shape[2]
    N_CTX_8 = k_8.shape[2]
    
    # Kernel for mask=1 (full resolution) - same grid as before since we moved Q loops inside
    grid_backward_1 = (triton.cdiv(k.shape[2], ctx.BLOCK_N), ctx.grid[1])
    _bwd_kernel_mask_1[grid_backward_1](
        q, k, v, ctx.sm_scale,
        block_sparse_mask,
        o, do_scaled,
        dq, dk, dv,
        L, m, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        block_sparse_mask.stride(0), block_sparse_mask.stride(1), block_sparse_mask.stride(2), block_sparse_mask.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[1], N_CTX,
        BLOCK_M=ctx.BLOCK_M,
        BLOCK_N=ctx.BLOCK_N,
        BLOCK_DMODEL=ctx.BLOCK_DMODEL,
        BLOCK_N_SMALL=32,
        num_warps=ctx.num_warps,
        num_stages=1,
    )
    
    # Kernel for mask=2 (2x downsampled)
    POOLING_BLOCK_N_2 = getattr(ctx, 'POOLING_BLOCK_N', ctx.BLOCK_N) // 2
    grid_backward_2 = (triton.cdiv(k_2.shape[2], max(16, POOLING_BLOCK_N_2)), ctx.grid[1])
    _bwd_kernel_mask_2[grid_backward_2](
        q, k_2, v_2, ctx.sm_scale,
        block_sparse_mask,
        o, do_scaled,
        dq, dk_2, dv_2,
        L, m, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_2.stride(0), k_2.stride(1), k_2.stride(2), k_2.stride(3),
        v_2.stride(0), v_2.stride(1), v_2.stride(2), v_2.stride(3),
        block_sparse_mask.stride(0), block_sparse_mask.stride(1), block_sparse_mask.stride(2), block_sparse_mask.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[1], N_CTX, N_CTX_2,
        BLOCK_M=ctx.BLOCK_M,
        BLOCK_N_2=max(16, POOLING_BLOCK_N_2),
        BLOCK_DMODEL=ctx.BLOCK_DMODEL,
        num_warps=ctx.num_warps,
        num_stages=1,
    )
    
    # Kernel for mask=4 (4x downsampled)
    POOLING_BLOCK_N_4 = getattr(ctx, 'POOLING_BLOCK_N', ctx.BLOCK_N) // 4
    grid_backward_4 = (triton.cdiv(k_4.shape[2], max(16, POOLING_BLOCK_N_4)), ctx.grid[1])
    _bwd_kernel_mask_4[grid_backward_4](
        q, k_4, v_4, ctx.sm_scale,
        block_sparse_mask,
        o, do_scaled,
        dq, dk_4, dv_4,
        L, m, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_4.stride(0), k_4.stride(1), k_4.stride(2), k_4.stride(3),
        v_4.stride(0), v_4.stride(1), v_4.stride(2), v_4.stride(3),
        block_sparse_mask.stride(0), block_sparse_mask.stride(1), block_sparse_mask.stride(2), block_sparse_mask.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[1], N_CTX, N_CTX_4,
        BLOCK_M=ctx.BLOCK_M,
        BLOCK_N_4=max(16, POOLING_BLOCK_N_4),
        BLOCK_DMODEL=ctx.BLOCK_DMODEL,
        num_warps=ctx.num_warps,
        num_stages=1,
    )
    
    # Kernel for mask=8 (8x downsampled)
    POOLING_BLOCK_N_8 = getattr(ctx, 'POOLING_BLOCK_N', ctx.BLOCK_N) // 8
    grid_backward_8 = (triton.cdiv(k_8.shape[2], max(16, POOLING_BLOCK_N_8)), ctx.grid[1])
    _bwd_kernel_mask_8[grid_backward_8](
        q, k_8, v_8, ctx.sm_scale,
        block_sparse_mask,
        o, do_scaled,
        dq, dk_8, dv_8,
        L, m, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_8.stride(0), k_8.stride(1), k_8.stride(2), k_8.stride(3),
        v_8.stride(0), v_8.stride(1), v_8.stride(2), v_8.stride(3),
        block_sparse_mask.stride(0), block_sparse_mask.stride(1), block_sparse_mask.stride(2), block_sparse_mask.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[1], N_CTX, N_CTX_8,
        BLOCK_M=ctx.BLOCK_M,
        BLOCK_N_8=max(16, POOLING_BLOCK_N_8),
        BLOCK_DMODEL=ctx.BLOCK_DMODEL,
        num_warps=ctx.num_warps,
        num_stages=1,
    )

    # Combine gradients from multi-scale K,V into the original K,V
    # Need to handle padding/truncation correctly since k,v were padded before pooling
    
    # Start with gradients from full-resolution attention
    dk_combined = dk.clone()
    dv_combined = dv.clone()
    
    # Create padded gradients to match the padded k,v used in forward pass
    # This ensures we can properly upsample from multi-scale gradients
    k_padded_size = k.shape[2]  # This is the padded size used in forward
    
    # Add contributions from downsampled versions through upsampling
    # The upsampled gradients should be added to the padded space, then truncated
    
    # Upsample dk_2 (2x downsampled) back to padded size
    B, H, L_2, D = dk_2.shape
    dk_2_upsampled = dk_2.repeat_interleave(2, dim=2)  # Each pooled element affects 2 consecutive elements
    # Truncate to match padded k size
    if dk_2_upsampled.size(2) > k_padded_size:
        dk_2_upsampled = dk_2_upsampled[:, :, :k_padded_size, :]
    # Add to combined gradients with proper scaling for average pooling
    dk_combined[:, :, :dk_2_upsampled.size(2), :] += dk_2_upsampled / 2
    
    # Upsample dk_4 (4x downsampled) back to padded size  
    B, H, L_4, D = dk_4.shape
    dk_4_upsampled = dk_4.repeat_interleave(4, dim=2)  # Each pooled element affects 4 consecutive elements
    if dk_4_upsampled.size(2) > k_padded_size:
        dk_4_upsampled = dk_4_upsampled[:, :, :k_padded_size, :]
    dk_combined[:, :, :dk_4_upsampled.size(2), :] += dk_4_upsampled / 4
    
    # Upsample dk_8 (8x downsampled) back to padded size
    B, H, L_8, D = dk_8.shape  
    dk_8_upsampled = dk_8.repeat_interleave(8, dim=2)  # Each pooled element affects 8 consecutive elements
    if dk_8_upsampled.size(2) > k_padded_size:
        dk_8_upsampled = dk_8_upsampled[:, :, :k_padded_size, :]
    dk_combined[:, :, :dk_8_upsampled.size(2), :] += dk_8_upsampled / 8
    
    # Similar process for dv gradients
    v_padded_size = v.shape[2]  # This is the padded size used in forward
    
    # Upsample dv_2 (2x downsampled) back to padded size
    B, H, L_2, D = dv_2.shape
    dv_2_upsampled = dv_2.repeat_interleave(2, dim=2)
    if dv_2_upsampled.size(2) > v_padded_size:
        dv_2_upsampled = dv_2_upsampled[:, :, :v_padded_size, :]
    dv_combined[:, :, :dv_2_upsampled.size(2), :] += dv_2_upsampled / 2
    
    # Upsample dv_4 (4x downsampled) back to padded size
    B, H, L_4, D = dv_4.shape
    dv_4_upsampled = dv_4.repeat_interleave(4, dim=2)
    if dv_4_upsampled.size(2) > v_padded_size:
        dv_4_upsampled = dv_4_upsampled[:, :, :v_padded_size, :]
    dv_combined[:, :, :dv_4_upsampled.size(2), :] += dv_4_upsampled / 4
    
    # Upsample dv_8 (8x downsampled) back to padded size
    B, H, L_8, D = dv_8.shape
    dv_8_upsampled = dv_8.repeat_interleave(8, dim=2)
    if dv_8_upsampled.size(2) > v_padded_size:
        dv_8_upsampled = dv_8_upsampled[:, :, :v_padded_size, :]
    dv_combined[:, :, :dv_8_upsampled.size(2), :] += dv_8_upsampled / 8
    
    # Truncate back to original size (remove padding effects)
    # This ensures gradients only flow to the original unpadded k,v
    dk_final = dk_combined[:, :, :original_k_size, :]
    dv_final = dv_combined[:, :, :original_v_size, :]
    
    # Convert dq back to original dtype to match do
    dq = dq.to(do.dtype)
    
    return dq, dk_final, dv_final, None, None


class _sparse_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, block_sparse_dense, sm_scale):
        # shape constraints
        return _forward(ctx, q, k, v, block_sparse_dense, sm_scale)

    @staticmethod
    def backward(ctx, do):
        return _backward(ctx, do)

def sparse_attention_factory(BLOCK_M=128, BLOCK_N=128, POOLING_BLOCK_N=128, **kwargs):
    class _sparse_attention_config(_sparse_attention):
        @staticmethod
        def forward(ctx, q, k, v, block_sparse_dense, sm_scale=None):
            sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
            # shape constraints
            return _forward(
                ctx,
                q,
                k,
                v,
                block_sparse_dense,
                sm_scale,
                BLOCK_M,
                BLOCK_N,
                POOLING_BLOCK_N,
                **kwargs,
            )
    return _sparse_attention_config.apply

block_sparse_triton_fn = _sparse_attention.apply

    