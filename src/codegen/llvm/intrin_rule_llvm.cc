/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file intrin_rule_llvm.cc
 */
#ifdef TVM_LLVM_VERSION

#include "intrin_rule_llvm.h"

#include <llvm/IR/Intrinsics.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <tvm/codegen/intrin_rule.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/memory.h>

#include <limits>

namespace tvm {
namespace codegen {
using namespace tvm::prim;

namespace llvm {
namespace intrin {
using tirx::FLowerIntrinsic;

// Use FP32 for the libm ABI and for expanded math expressions that need wider
// intermediate precision. Round at the original call boundary, retaining integer
// arguments (for example, the exponent of ldexp). Direct LLVM math intrinsics
// keep their original type so LLVM can select native operations or legalize them.
template <auto Lower>
PrimExpr DispatchFloatMath(const PrimExpr& e, const Target& target) {
  PrimType ty = e.ty();
  if (!ty.MatchesElementType(kDLFloat, 16)) return Lower(e, target);
  const CallObj* call = e.as<CallObj>();
  TVM_FFI_ICHECK(call != nullptr);
  auto args = call->args.Map([](const Expr& arg) -> Expr {
    PrimExpr value = arg.as_or_throw<PrimExpr>();
    return value.ty().MatchesElementType(kDLFloat, 16) ? cast(value, value.ty().WithBits(32))
                                                       : value;
  });
  PrimExpr promoted = Call(ty.WithBits(32), call->op, args, call->attrs, call->ty_args, call->loc)
                          .as_or_throw<PrimExpr>();
  return cast(Lower(promoted, target), ty);
}

// libm has a scalar ABI. Lower fixed vectors lane by lane, binding the arguments
// once so scalarization does not repeat loads or effectful argument expressions.
PrimExpr DispatchMathExtern(const PrimExpr& e, const Target& target) {
  using ::tvm::codegen::intrin::DispatchPureExtern;
  using ::tvm::codegen::intrin::FloatSuffix;
  PrimType ty = e.ty();
  if (ty.IsScalar()) return DispatchPureExtern<FloatSuffix>(e, target);
  if (!ty.MatchesElementType(kDLFloat, 32) && !ty.MatchesElementType(kDLFloat, 64)) return e;
  TVM_FFI_CHECK(!ty.IsScalableVector(), ValueError)
      << "LLVM math extern lowering requires a scalar or fixed-length vector, but received "
      << ty->dtype;
  const CallObj* call = e.as<CallObj>();
  TVM_FFI_ICHECK(call != nullptr);
  ffi::Array<PrimVar> vars;
  for (const PrimExpr& arg : call->args.as_or_throw<ffi::Array<PrimExpr>>()) {
    TVM_FFI_ICHECK(arg.ty().IsScalar() ||
                   (arg.ty().IsFixedLengthVector() && arg.ty().lanes() == ty.lanes()));
    vars.push_back(PrimVar("math_arg", arg.ty()));
  }
  ffi::Array<PrimExpr> lanes;
  for (int lane = 0; lane < ty.lanes(); ++lane) {
    ffi::Array<Expr> args;
    for (const PrimVar& var : vars) {
      args.push_back(var.ty().IsScalar() ? PrimExpr(var) : Shuffle::ExtractElement(var, lane));
    }
    PrimExpr scalar = Call(ty.WithLanes(1), call->op, args, call->attrs, call->ty_args, call->loc)
                          .as_or_throw<PrimExpr>();
    lanes.push_back(DispatchPureExtern<FloatSuffix>(scalar, target));
  }
  PrimExpr result = Shuffle::Concat(lanes);
  for (size_t i = vars.size(); i != 0; --i) {
    result = Let(vars[i - 1], call->args[i - 1].as_or_throw<PrimExpr>(), result);
  }
  return result;
}

PrimExpr DispatchNextAfter(const PrimExpr& e, const Target& target) {
  PrimType ty = e.ty();
  if (!ty.MatchesElementType(kDLFloat, 16)) return DispatchMathExtern(e, target);
  const CallObj* call = e.as<CallObj>();
  TVM_FFI_ICHECK(call != nullptr);
  TVM_FFI_ICHECK_EQ(call->args.size(), 2U);
  PrimVar x("nextafter_x", ty), y("nextafter_y", ty);
  PrimType bits_ty = ty.WithCode(kDLUInt);
  PrimExpr x_bits = tirx::reinterpret(bits_ty, x);
  PrimExpr y_bits = tirx::reinterpret(bits_ty, y);
  PrimExpr zero = MakeConst(ty, 0);
  PrimExpr one_bit = MakeConst(bits_ty, 1);
  PrimExpr sign_bit = MakeConst(bits_ty, 0x8000);
  // Step through the binary16 representation, rather than through FP32 values.
  // Both signed zeros compare equal; stepping away from zero takes y's sign.
  PrimExpr bits = Select((x < y) == (x > zero), x_bits + one_bit, x_bits - one_bit);
  bits = Select(x == zero, (y_bits & sign_bit) | one_bit, bits);
  PrimExpr result = Select(x == y, y, tirx::reinterpret(ty, bits));
  result = Select(isnan(x) || isnan(y), x + y, result);
  return Let(x, call->args[0].as_or_throw<PrimExpr>(),
             Let(y, call->args[1].as_or_throw<PrimExpr>(), result));
}

PrimExpr DispatchExp10(const PrimExpr& e, const Target& target) {
  const CallObj* call = e.as<CallObj>();
  TVM_FFI_ICHECK(call != nullptr);
  PrimExpr x = call->args[0].as_or_throw<PrimExpr>();
  return pow(MakeConst(x.ty(), 10), x);
}

PrimExpr DispatchRSqrt(const PrimExpr& e, const Target& target) {
  const CallObj* call = e.as<CallObj>();
  TVM_FFI_ICHECK(call != nullptr);
  PrimExpr x = call->args[0].as_or_throw<PrimExpr>();
  return MakeConst(x.ty(), 1) / sqrt(x);
}

PrimExpr DispatchSigmoid(const PrimExpr& e, const Target& target) {
  const CallObj* call = e.as<CallObj>();
  TVM_FFI_ICHECK(call != nullptr);
  PrimType ty = e.ty();
  PrimVar x("sigmoid_x", ty), z("sigmoid_exp", ty);
  PrimExpr one = MakeConst(ty, 1);
  // exp(-abs(x)) cannot overflow. The negative branch retains subnormal
  // results that would be lost by computing 1 / (1 + exp(-x)).
  PrimExpr result = Select(x >= MakeConst(ty, 0), one / (one + z), z / (one + z));
  return Let(x, call->args[0].as_or_throw<PrimExpr>(), Let(z, exp(-abs(x)), result));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char* name :
       {"prim.log1p", "prim.sinh", "prim.cosh", "prim.asin", "prim.acos", "prim.atan", "prim.asinh",
        "prim.acosh", "prim.erf", "prim.atanh", "prim.tanh", "prim.tan", "prim.atan2", "prim.hypot",
        "prim.fmod", "prim.ldexp"}) {
    OpDef(name).set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                          DispatchFloatMath<DispatchMathExtern>);
  }

  OpDef("prim.nextafter").set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic", DispatchNextAfter);
  OpDef("prim.copysign")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::copysign, 2>);
  OpDef("prim.exp10").set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic", DispatchExp10);
  OpDef("prim.rsqrt")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic", DispatchFloatMath<DispatchRSqrt>);
  OpDef("prim.sigmoid")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic", DispatchFloatMath<DispatchSigmoid>);

  OpDef("tirx.prefetch")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMIntrin<::llvm::Intrinsic::prefetch, 4>);

  OpDef("prim.exp")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::exp, 1>);

  OpDef("prim.exp2")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::exp2, 1>);

  OpDef("prim.fma")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::fmuladd, 3>);

  OpDef("prim.log")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::log, 1>);

  OpDef("prim.log2")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::log2, 1>);

  OpDef("prim.log10")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::log10, 1>);

  OpDef("prim.sqrt")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::sqrt, 1>);

  OpDef("prim.floor")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::floor, 1>);

  OpDef("prim.ceil")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::ceil, 1>);

  OpDef("prim.trunc")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::trunc, 1>);

  OpDef("prim.fabs")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::fabs, 1>);

  OpDef("prim.round")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::nearbyint, 1>);

  OpDef("prim.nearbyint")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::nearbyint, 1>);

  OpDef("prim.pow")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::pow, 2>);

  OpDef("prim.popcount")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::ctpop, 1>);

  OpDef("prim.cos")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::cos, 1>);

  OpDef("prim.sin")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::sin, 1>);
}
}  // namespace intrin

namespace legalize {
using tirx::FLegalize;

TVM_FFI_STATIC_INIT_BLOCK() {
  // clang-format off
  OpDef("prim.clz")
      .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e, const Target& target) -> PrimExpr {
    const CallObj* call = e.as<CallObj>();
    TVM_FFI_ICHECK(call != nullptr);
    TVM_FFI_ICHECK_EQ(call->args.size(), 1);
    ffi::Array<PrimExpr> cargs;
    cargs.push_back(IntImm(PrimType::UInt(32), ::llvm::Intrinsic::ctlz));
    cargs.push_back(call->args[0].as_or_throw<PrimExpr>());
    cargs.push_back(IntImm(PrimType::Int(1), 1));  // is_zero_undef
    // LLVM requires that the return type must match the first argument type
    auto clz =
        Call(call->args[0]->ty.as_or_throw<PrimType>(), tirx::call_llvm_intrin_op(), cargs)
            .as_or_throw<PrimExpr>();
    return cast(clz, call->ty.as_or_throw<PrimType>());
  });
  // clang-format on
}

}  // namespace legalize
}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
