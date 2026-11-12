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
"""LLVM math lowering, including FP16 compute precision and scalar libm ABIs."""

import math
import re

import numpy as np
import pytest

import tvm_tirx
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

pytestmark = pytest.mark.skipif(not env.has_llvm(), reason="need llvm")

UNARY_MATH = [
    ("exp", np.exp),
    ("exp2", np.exp2),
    ("exp10", lambda x: np.power(10.0, x)),
    ("log", np.log),
    ("log2", np.log2),
    ("log10", np.log10),
    ("log1p", np.log1p),
    ("sin", np.sin),
    ("cos", np.cos),
    ("tan", np.tan),
    ("sinh", np.sinh),
    ("cosh", np.cosh),
    ("tanh", np.tanh),
    ("asin", np.arcsin),
    ("acos", np.arccos),
    ("atan", np.arctan),
    ("asinh", np.arcsinh),
    ("acosh", np.arccosh),
    ("atanh", np.arctanh),
    ("erf", np.vectorize(math.erf)),
    ("sqrt", np.sqrt),
    ("rsqrt", lambda x: 1.0 / np.sqrt(x)),
    ("sigmoid", lambda x: np.exp(-np.logaddexp(0.0, -x))),
    ("floor", np.floor),
    ("ceil", np.ceil),
    ("trunc", np.trunc),
    ("round", np.rint),
    ("nearbyint", np.rint),
    ("abs", np.fabs),
]


def _run_unary(op, inputs, lanes):
    dtype = inputs.dtype.name
    n = T.dynamic("n", "int32")

    @T.function
    def kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        for i in range(n // lanes):
            if I.constexpr(lanes == 1):
                B[i] = op(A[i])
            else:
                B[T.Ramp(i * lanes, 1, lanes)] = op(A[T.Ramp(i * lanes, 1, lanes)])

    assert len(inputs) % lanes == 0
    compiled = tvm_tirx.compile(kernel, target="llvm")
    output = tvm_tirx.runtime.tensor(np.empty_like(inputs))
    compiled(tvm_tirx.runtime.tensor(inputs), output)
    return output.numpy()


def _run_binary(op, lhs, rhs, lanes):
    dtype, rhs_dtype = lhs.dtype.name, rhs.dtype.name
    n = T.dynamic("n", "int32")

    @T.function
    def kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), rhs_dtype), C: T.Tensor((n,), dtype)):
        for i in range(n // lanes):
            if I.constexpr(lanes == 1):
                C[i] = op(A[i], B[i])
            else:
                C[T.Ramp(i * lanes, 1, lanes)] = op(
                    A[T.Ramp(i * lanes, 1, lanes)], B[T.Ramp(i * lanes, 1, lanes)]
                )

    assert len(lhs) == len(rhs) and len(lhs) % lanes == 0
    compiled = tvm_tirx.compile(kernel, target="llvm")
    output = tvm_tirx.runtime.tensor(np.empty_like(lhs))
    compiled(tvm_tirx.runtime.tensor(lhs), tvm_tirx.runtime.tensor(rhs), output)
    return output.numpy()


def _assert_result(actual, expected, exact=False):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    zeros = expected == 0
    np.testing.assert_array_equal(actual[zeros], expected[zeros])
    np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))
    finite = np.isfinite(expected)
    if exact:
        bits = np.dtype(f"uint{expected.dtype.itemsize * 8}")
        np.testing.assert_array_equal(actual[finite].view(bits), expected[finite].view(bits))
    elif expected.dtype == np.float16:
        # Compare in FP16 ULPs, so a fixed absolute tolerance cannot hide small-value errors.
        def ordered(values):
            bits = values.view(np.uint16).astype(np.int32)
            return np.where(bits & 0x8000, 0x8000 - (bits & 0x7FFF), 0x8000 + bits)

        np.testing.assert_allclose(
            ordered(actual[finite]), ordered(expected[finite]), rtol=0, atol=1
        )
    else:
        rtol = 1e-6 if expected.dtype == np.float32 else 1e-14
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=rtol, atol=0)


@pytest.mark.parametrize("name, reference", UNARY_MATH)
@pytest.mark.parametrize("lanes", [1, 4])
def test_unary_float16_all_values(name, reference, lanes):
    inputs = np.arange(65536, dtype=np.uint16).view(np.float16)
    actual = _run_unary(getattr(tvm_tirx.tirx, name), inputs, lanes)
    with np.errstate(all="ignore"):
        expected = reference(inputs.astype(np.float64)).astype(np.float16)
    _assert_result(actual, expected)


@pytest.mark.parametrize("name, reference", UNARY_MATH)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("lanes", [1, 4])
def test_unary_math(name, reference, dtype, lanes):
    info = np.finfo(dtype)
    inputs = np.array(
        [
            -np.inf,
            -100.0,
            -2.0,
            -1.0,
            -0.5,
            -1e-5,
            -info.tiny,
            -info.smallest_subnormal,
            -0.0,
            0.0,
            info.smallest_subnormal,
            info.tiny,
            1e-5,
            0.5,
            1.0,
            2.0,
            3.0,
            100.0,
            np.inf,
            np.nan,
        ],
        dtype=dtype,
    )
    actual = _run_unary(getattr(tvm_tirx.tirx, name), inputs, lanes)
    with np.errstate(all="ignore"):
        expected = reference(inputs.astype(np.float64)).astype(dtype)
    _assert_result(actual, expected)


@pytest.mark.parametrize("name", ["atanh", "tanh", "asinh", "sinh", "sin", "tan", "log1p"])
@pytest.mark.parametrize("lanes", [1, 4])
def test_float16_math_preserves_tiny_inputs(name, lanes):
    # These functions round to x for this interval. Losing x in 1 +/- x is incorrect.
    inputs = np.arange(0x0C00, dtype=np.uint16).view(np.float16)
    inputs = np.concatenate([inputs, -inputs])
    actual = _run_unary(getattr(tvm_tirx.tirx, name), inputs, lanes)
    reference = dict(UNARY_MATH)[name]
    expected = reference(inputs.astype(np.float64)).astype(np.float16)
    _assert_result(actual, expected, exact=True)


@pytest.mark.parametrize(
    "name, reference",
    [
        ("atan2", np.arctan2),
        ("hypot", np.hypot),
        ("fmod", np.fmod),
        ("pow", np.power),
        ("copysign", np.copysign),
        ("nextafter", np.nextafter),
    ],
)
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("lanes", [1, 4])
def test_binary_math(name, reference, dtype, lanes):
    info = np.finfo(dtype)
    values = np.array(
        [
            -np.inf,
            -info.max,
            -10,
            -2,
            -1,
            -0.5,
            -info.tiny,
            -info.smallest_subnormal,
            -0.0,
            0.0,
            info.smallest_subnormal,
            info.tiny,
            0.5,
            1,
            2,
            3,
            10,
            info.max,
            np.inf,
            np.nan,
        ],
        dtype=dtype,
    )
    lhs, rhs = np.repeat(values, len(values)), np.tile(values, len(values))
    actual = _run_binary(getattr(tvm_tirx.tirx, name), lhs, rhs, lanes)
    with np.errstate(all="ignore"):
        expected = (
            reference(lhs, rhs)
            if name == "nextafter"
            else reference(lhs.astype(np.float64), rhs.astype(np.float64)).astype(dtype)
        )
    _assert_result(actual, expected, exact=name in ("nextafter", "copysign"))


@pytest.mark.parametrize("direction", [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf])
@pytest.mark.parametrize("lanes", [1, 4])
def test_nextafter_float16_all_values(direction, lanes):
    inputs = np.arange(65536, dtype=np.uint16).view(np.float16)
    towards = np.full_like(inputs, direction)
    actual = _run_binary(tvm_tirx.tirx.nextafter, inputs, towards, lanes)
    with np.errstate(all="ignore"):
        expected = np.nextafter(inputs, towards)
    _assert_result(actual, expected, exact=True)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("lanes", [1, 4])
def test_ldexp_integer_exponent(dtype, lanes):
    info = np.finfo(dtype)
    values = np.array(
        [
            -np.inf,
            -info.max,
            -1,
            -0.75,
            -info.tiny,
            -info.smallest_subnormal,
            -0.0,
            0.0,
            info.smallest_subnormal,
            info.tiny,
            0.75,
            1,
            info.max,
            np.inf,
            np.nan,
            3,
        ],
        dtype=dtype,
    )
    exponents = np.array(
        [-(2**31), -1075, -150, -25, -1, 0, 1, 25, 150, 1024, 2**31 - 1], dtype="int32"
    )
    lhs, rhs = np.repeat(values, len(exponents)), np.tile(exponents, len(values))
    actual = _run_binary(tvm_tirx.tirx.ldexp, lhs, rhs, lanes)
    with np.errstate(all="ignore"):
        expected = np.ldexp(lhs.astype(np.float64), rhs).astype(dtype)
    _assert_result(actual, expected, exact=True)


def test_float16_math_rounds_at_each_call():
    inputs = np.array([12, -20, 0.5, -0.5], dtype="float16")
    actual = _run_unary(lambda x: tvm_tirx.tirx.sqrt(tvm_tirx.tirx.exp(x)), inputs, 1)
    with np.errstate(all="ignore"):
        intermediate = np.exp(inputs.astype(np.float64)).astype(np.float16)
        expected = np.sqrt(intermediate.astype(np.float64)).astype(np.float16)
    _assert_result(actual, expected)


def test_vector_math_evaluates_argument_once():
    @T.function
    def kernel(B: T.Tensor((4,), "float16")):
        B[0:4] = T.atanh(T.call_extern("read_math_vector", ty="float16x4"))

    llvm_ir = tvm_tirx.compile(kernel, target="llvm").mod.inspect_source("ll")
    assert len(re.findall(r"call .*@read_math_vector\(", llvm_ir)) == 1


def test_scalable_math_extern_diagnostic():
    @T.function
    def kernel(A: T.Tensor((128,), "float16"), B: T.Tensor((128,), "float16")):
        B[T.Ramp(0, 1, 4 * T.vscale())] = T.atanh(A[T.Ramp(0, 1, 4 * T.vscale())])

    target = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]}
    with pytest.raises(ValueError, match="requires a scalar or fixed-length vector"):
        tvm_tirx.compile(kernel, target=target)
