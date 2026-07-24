#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_builders.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace ftlpu::compiler::schedule {

struct TaskAllocation {
    mlir::DictionaryAttr address;
    mlir::DictionaryAttr placement;
    int64_t bytes;
};

struct PrimitiveFfnSchedulePlan {
    stream::ElementwiseTaskOp add;
    stream::MatmulTaskOp down0;
    stream::MatmulTaskOp down1;
    stream::RouteOp hidden0_route;
    stream::RouteOp hidden1_route;
    stream::ElementwiseTaskOp multiply;
    stream::SwishTaskOp swish;
    stream::MatmulTaskOp gate;
    stream::MatmulTaskOp up;
    stream::RouteOp activation_route;
    stream::RouteOp gate_route;
    stream::RouteOp up_route;
    stream::RouteOp down0_route;
    stream::RouteOp down1_route;
    TaskAllocation hidden0;
    TaskAllocation hidden1;
    TaskAllocation result;
    mlir::FloatAttr gate_scale;
    mlir::FloatAttr up_scale;
    mlir::FloatAttr down_rhs_scale;
    FfnTaskPlan task_plan;

    mlir::Value getActivation() { return activation_route.getOutput(); }
    mlir::Value getGateWeight() { return gate_route.getOutput(); }
    mlir::Value getUpWeight() { return up_route.getOutput(); }
    mlir::Value getDownWeight0() { return down0_route.getOutput(); }
    mlir::Value getDownWeight1() { return down1_route.getOutput(); }
    mlir::Value getResult() { return add.getResult(); }
    mlir::Operation* getOperation() { return add.getOperation(); }
    mlir::Location getLoc() { return add.getLoc(); }
    uint64_t getM() { return down0.getM(); }
    uint64_t getK() { return gate.getK(); }
    uint64_t getHidden() { return gate.getN(); }
    uint64_t getN() { return down0.getN(); }
    llvm::APFloat getGateScale() { return gate_scale.getValue(); }
    llvm::APFloat getUpScale() { return up_scale.getValue(); }
    llvm::APFloat getDownRhsScale() { return down_rhs_scale.getValue(); }
    mlir::DictionaryAttr getHidden0Address() { return hidden0.address; }
    mlir::DictionaryAttr getHidden0AddressAttr() { return hidden0.address; }
    mlir::DictionaryAttr getHidden0Placement() { return hidden0.placement; }
    mlir::DictionaryAttr getHidden1Address() { return hidden1.address; }
    mlir::DictionaryAttr getHidden1AddressAttr() { return hidden1.address; }
    mlir::DictionaryAttr getResultAddress() { return result.address; }
    mlir::DictionaryAttr getResultAddressAttr() { return result.address; }
    mlir::DictionaryAttr getResultPlacement() { return result.placement; }
};

mlir::FailureOr<llvm::SmallVector<PrimitiveFfnSchedulePlan, 2>>
collectPrimitiveFfnSchedulePlans(mlir::func::FuncOp function);

} // namespace ftlpu::compiler::schedule
