# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Mechanical matrix-descriptor encoding and its documented address walk.

Canonical coordinates determine the descriptor's byte strides. Only the finite
hardware swizzle modes are checked; no coordinate permutation or instruction
selection is performed. PTX ISA 9.7.18.3.1 and 9.7.18.4.1.
"""

import tvm_ffi

from tvm_tirx.backend.cuda.cpp.descriptors import encode_smem_descriptor_base_uint64
from tvm_tirx.ir import Bind, Call, IntImm, TensorLoad, Var
from tvm_tirx.ir.prim import EQ, BitwiseAnd, BitwiseXor, Let, LShift, RShift
from tvm_tirx.runtime import MemoryScope
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import ComposeLayout


def swizzle_bytes(offset, swizzle):
    if swizzle == 4:
        # 128B span, 32B atomicity: XOR address bits [8:7] into [6:5].
        return T.bitwise_xor(offset, T.bitwise_and(offset // 4, 96))
    return T.bitwise_xor(offset, T.bitwise_and(offset // 8, (2**swizzle - 1) * 16))


def _normalize_descriptor_proof(ctx, expression):
    """Normalize a pure address predicate without changing the emitted address.

    A stage held in a local scalar is unknown to the layout prover. Snapshot
    its reads within this expression as one unconstrained variable, then expose
    the arithmetic behind the swizzle bits. No stage value or range is assumed.
    """
    expression = ctx.index_coordinates(expression)
    expression = tvm_ffi.structural_map(
        expression,
        (
            Let,
            lambda node: tvm_ffi.structural_map(
                node.body,
                (Var, lambda var: node.value if var.same_as(node.var) else var),
                order="post",
            ),
        ),
        order="post",
    )
    calls = []
    tvm_ffi.structural_map(expression, (Call, lambda node: calls.append(node) or node))
    if calls:
        return ctx.analyzer.simplify(expression)
    snapshots = {}

    def snapshot(load):
        ty = load.source.ty
        if (
            getattr(ty, "scope", None) != MemoryScope("local")
            or len(ty.shape) != 1
            or not isinstance(ty.shape[0], IntImm)
            or int(ty.shape[0]) != 1
            or len(load.indices) != 1
            or not isinstance(load.indices[0], IntImm)
            or int(load.indices[0]) != 0
        ):
            return load
        if load.source not in snapshots:
            snapshots[load.source] = Var("descriptor_scalar", load.ty)
        return snapshots[load.source]

    expression = tvm_ffi.structural_map(expression, (TensorLoad, snapshot), order="post")

    def shift(node, left):
        if not isinstance(node.b, IntImm):
            return node
        amount, dtype = int(node.b), node.a.ty.dtype
        if not 0 <= amount < dtype.bits - 1:
            return node
        return node.a * (1 << amount) if left else node.a // (1 << amount)

    def mask(node):
        value, bits = node.a, node.b
        if not isinstance(bits, IntImm):
            if not isinstance(value, IntImm):
                return node
            value, bits = bits, value
        bits = int(bits)
        if bits <= 0 or bits >= 1 << (value.ty.dtype.bits - 1):
            return node
        unit = bits & -bits
        size = bits // unit + 1
        return node if size & (size - 1) else value // unit % size * unit

    def xor(node):
        left, right = (ctx.analyzer.simplify(value) for value in (node.a, node.b))
        if isinstance(left, IntImm) and int(left) == 0:
            return right
        if isinstance(right, IntImm) and int(right) == 0:
            return left
        if isinstance(left, IntImm) and isinstance(right, IntImm):
            return IntImm(left.ty.dtype, int(left) ^ int(right))
        lm, rm = (ctx.analyzer.modular_set(value) for value in (left, right))
        factors = int(lm.coeff) | int(lm.base) | int(rm.coeff) | int(rm.base)
        unit = factors & -factors if factors else 1
        if 1 < unit < 1 << (left.ty.dtype.bits - 1):
            return T.bitwise_xor(left // unit, right // unit) * unit
        bound = ctx.analyzer.const_int_bound(right)
        if 0 <= bound.min_value and bound.max_value < 1 << 15:
            period = 1 << int(bound.max_value).bit_length()
            outer = ctx.analyzer.const_int_bound(left)
            if period < 1 << (left.ty.dtype.bits - 1) and (
                outer.min_value < 0 or outer.max_value >= period
            ):
                return left // period * period + T.bitwise_xor(left % period, right)
        return T.bitwise_xor(left, right)

    for _ in range(4):
        expression = tvm_ffi.structural_map(
            expression, (LShift, lambda node: shift(node, True)), order="post"
        )
        expression = tvm_ffi.structural_map(
            expression, (RShift, lambda node: shift(node, False)), order="post"
        )
        expression = tvm_ffi.structural_map(expression, (BitwiseAnd, mask), order="post")
        expression = tvm_ffi.structural_map(expression, (BitwiseXor, xor), order="post")
        expression = ctx.analyzer.simplify(expression)
    # Cancel the unknown common origin before enumerating known repeat axes.
    expression = tvm_ffi.structural_map(
        expression,
        (EQ, lambda node: ctx.analyzer.simplify(node.a - node.b) == IntImm(node.a.ty.dtype, 0)),
        order="post",
    )
    return ctx.analyzer.simplify(expression)


def _is_call(expression, name):
    return isinstance(expression, Call) and getattr(expression.op, "name", None) == name


def _shuffled_pointer(expression):
    """Recognize a caller's explicit full-warp broadcast of pointer bits."""
    if not _is_call(expression, "tirx.reinterpret"):
        return None
    shuffle = expression.args[0]
    if not _is_call(shuffle, "tirx.gpu_warp_shuffle") or len(shuffle.args) != 5:
        return None
    for value, expected in zip((shuffle.args[0], *shuffle.args[2:]), (0xFFFFFFFF, 0, 32, 32)):
        if not isinstance(value, IntImm) or int(value) != expected:
            return None
    source = shuffle.args[1]
    return source.args[0] if _is_call(source, "tirx.reinterpret") else None


def _owned_shared_pointer(pointer, roots, definitions, seen=frozenset()):
    if _is_call(pointer, "tirx.ptr_byte_offset"):
        return (
            isinstance(pointer.args[1], IntImm)
            and int(pointer.args[1]) >= 0
            and (_owned_shared_pointer(pointer.args[0], roots, definitions, seen))
        )
    if _is_call(pointer, "tirx.reinterpret"):
        return _owned_shared_pointer(pointer.args[0], roots, definitions, seen)
    if not _is_call(pointer, "tirx.tensor_data_ptr"):
        return False
    tensor = roots.get(pointer.args[0], pointer.args[0])
    if tensor in seen or tensor.ty.scope not in (MemoryScope("shared"), MemoryScope("shared.dyn")):
        return False
    definition = definitions.get(tensor)
    if _is_call(definition, "tirx.alloc_tensor"):
        return True
    return _is_call(definition, "tirx.decl_tensor") and _owned_shared_pointer(
        definition.args[0], roots, definitions, seen | {tensor}
    )


def _cached_matrix_descriptor(ctx, smem, encoded, swizzle, atom_origin):
    roots = ctx.sctx.cache_get("tirx.storage_roots")
    definitions = ctx.sctx.cache_get("tirx.storage_definitions")
    if roots is None or definitions is None:
        return None
    root = roots.get(smem.tensor)
    definition = definitions.get(root) if root is not None else None
    if not _is_call(definition, "tirx.decl_tensor"):
        return None
    pointer = _shuffled_pointer(definition.args[0])
    if pointer is None:
        return None
    key = f"explicit_mma_low:{hash(root)}:{encoded}"
    low = ctx.sctx.cache_get(key)
    field = T.uint32(encoded & 0xFFFFFFFF)
    if low is None:
        low = Var("matrix_descriptor_lo", "uint32")
        start = T.cuda.cvta_generic_to_shared(root.data) // T.uint32(16)
        value = T.bitwise_or(T.bitwise_and(start, T.uint32(0x3FFF)), field)
        ctx.sctx.add_post_tensor_def_stmt(root, Bind(low, value))
        ctx.sctx.cache_set(key, low)
    physical = smem.memory(atom_origin) * smem.bits // 8
    offset = swizzle_bytes(physical, swizzle) if swizzle else physical
    start = low + T.cast(offset // 16, ty="uint32")
    if not _owned_shared_pointer(pointer, roots, definitions):
        start = T.bitwise_or(T.bitwise_and(start, T.uint32(0x3FFF)), field)
    # Owned local SMEM and each valid MMA view lie in the 14-bit start window.
    # Their difference cannot carry into the layout fields. Other roots keep
    # the complete modular address calculation above. No collective is added.
    return T.bitwise_or(T.cast(start, ty="uint64"), T.uint64(encoded >> 32) << T.uint64(32))


def _encode_matrix_descriptor(ctx, smem, ldo, sdo, swizzle, atom_origin):
    if swizzle not in range(5) or any(x < 0 or x % 16 or x >= 2**18 for x in (ldo, sdo)):
        raise ValueError("matrix descriptor requires swizzle 0..4 and 16-byte strides below 2^18")
    alignment = int(smem.ty.data_alignment)
    if alignment < 16 or alignment % 16:
        raise ValueError(f"{smem.name}: requires a known 16-byte-aligned base")
    ctx.require(
        _normalize_descriptor_proof(ctx, smem.memory(atom_origin) * smem.bits % 128 == 0),
        f"{smem.name}: atom address must be 16-byte aligned",
    )
    if swizzle:
        period = 512 if swizzle == 4 else 128 << swizzle
        if int(smem.ty.data_alignment) % period:
            raise ValueError(
                "swizzled matrix descriptor requires a base aligned to its full swizzle period"
            )
    encoded = encode_smem_descriptor_base_uint64(ldo // 16, sdo // 16, swizzle)
    cached = _cached_matrix_descriptor(ctx, smem, encoded, swizzle, atom_origin)
    if cached is not None:
        return cached
    descriptor = T.uint64(encoded & 0xFFFFFFFF) | (T.uint64(encoded >> 32) << T.uint64(32))
    address = T.cuda.cvta_generic_to_shared(smem.pointer(atom_origin))
    if swizzle:
        # The descriptor starts at an unswizzled byte address. The native
        # instruction applies the permutation, including the origin's phase.
        # XOR swizzling is its own inverse; the backing base is period-aligned.
        physical = smem.memory(atom_origin) * smem.bits // 8
        address = (
            address
            - T.cast(physical, ty="uint32")
            + T.cast(swizzle_bytes(physical, swizzle), ty="uint32")
        )
    address = T.cast(address // T.uint32(16), ty="uint64")
    return T.bitwise_or(descriptor, T.bitwise_and(address, T.uint64(0x3FFF)))


def _constant_stride(ctx, difference):
    """Extract a candidate at one valid origin; the full walk is checked later."""
    value = _normalize_descriptor_proof(ctx, difference)
    if not isinstance(value, IntImm):
        origins = {var: bounds.start for var, bounds in ctx.static_ranges.items()}
        origins.update(dict.fromkeys([*ctx.group, *(ctx.repeat or ())], 0))
        value = tvm_ffi.structural_map(
            ctx.index_coordinates(value),
            (Var, lambda var: IntImm(var.ty.dtype, origins[var]) if var in origins else var),
            order="post",
        )
        value = ctx.analyzer.simplify(value)
    if not isinstance(value, IntImm):
        raise ValueError("descriptor strides must be compile-time constants")
    bits = int(value)
    if bits < 0 or bits % 128 or bits >= 8 * 2**18:
        raise ValueError("descriptor strides must be 16-byte multiples below 2^18")
    return bits // 8


def _layout_swizzle(smem):
    layout = smem.layout
    if isinstance(layout, ComposeLayout) and layout.swizzle_inner:
        atomic_bits = (1 << int(layout.per_element)) * smem.bits
        if atomic_bits == 128 and int(layout.atom_len) == 3:
            return int(layout.swizzle_len)
        if atomic_bits == 256 and int(layout.atom_len) == 2 and int(layout.swizzle_len) == 2:
            return 4
    return 0


def matrix_descriptor(ctx, smem, atom_shape, *, major="k", copy=False):
    """Extract, validate and encode a shared matrix's canonical address walk.

    cp coordinates are (row, b32_word); MMA coordinates are (MN, K).
    The major order fixes which coordinate enumerates hardware rows. Row-group
    and column-chunk boundaries determine the two strides. An unobserved LDO
    is 16B, an unobserved SDO is zero. Every element, including its region
    origin and group/repeat coordinates, must match the resulting descriptor.
    """
    modes = [4] if not copy and major == "mn" and smem.bits == 32 else list(range(4))
    preferred = _layout_swizzle(smem)
    if preferred in modes:
        modes.remove(preferred)
        modes.insert(0, preferred)
    rows, columns = atom_shape if major == "k" else atom_shape[::-1]

    def physical(row, column):
        coord = (row, column) if major == "k" else (column, row)
        return smem.memory(coord) * smem.bits

    failures = []
    for mode in modes:
        row_bits = (128 if mode == 4 else 16 << mode) * 8
        group_rows = 4 if mode == 4 else 8

        def permute(bits):
            return swizzle_bytes(bits // 8, mode) * 8 + bits % 8 if mode else bits

        base = permute(physical(0, 0))
        swapped = major == "mn" and mode == 0
        ldo, sdo = 16, 0
        try:
            if rows > group_rows:
                stride = _constant_stride(ctx, permute(physical(group_rows, 0)) - base)
                if swapped:
                    ldo = stride
                else:
                    sdo = stride
            chunk = (row_bits + smem.bits - 1) // smem.bits
            if chunk < columns and not (copy and mode):
                stride = _constant_stride(
                    ctx, permute(physical(0, chunk)) - base - (chunk * smem.bits % row_bits)
                )
                if swapped:
                    sdo = stride
                else:
                    ldo = stride
            row_stride, column_stride = (ldo, sdo) if swapped else (sdo, ldo)
            for row in range(rows):
                for column in range(columns):
                    column_bits = column * smem.bits
                    if not (copy and mode):
                        column_bits = (
                            column_bits // row_bits * column_stride * 8 + column_bits % row_bits
                        )
                    offset = row // group_rows * row_stride * 8 + row % group_rows * row_bits
                    ctx.require(
                        _normalize_descriptor_proof(
                            ctx, physical(row, column) == permute(base + offset + column_bits)
                        ),
                        f"{smem.name}: descriptor address walk at row={row}, column={column}",
                    )
            return _encode_matrix_descriptor(ctx, smem, ldo, sdo, mode, (0, 0))
        except ValueError as error:
            failures.append(f"swizzle {mode}: {error}")
    raise ValueError(f"{smem.name}: no matching shared matrix descriptor; " + "; ".join(failures))
