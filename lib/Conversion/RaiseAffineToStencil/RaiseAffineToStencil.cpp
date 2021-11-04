#include "Conversion/RaiseAffineToStencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <tuple>

using namespace mlir;
using namespace stencil;
using namespace scf;

namespace {

template <class T>
bool isOperationCalled(mlir::Operation *operation) {
  if (operation == nullptr) {
    return false;
  }
  return operation->getName().getStringRef().equals(T::getOperationName());
}

template <class T>
bool isDefinedByOperation(mlir::Value value) {
  if (value == nullptr) {
    return false;
  }
  return value.getDefiningOp()->getName().getStringRef().equals(
      T::getOperationName());
}

bool hasApplyOp(mlir::Block &block) {
  for (auto &op : block.without_terminator()) {
    if (isOperationCalled<stencil::ApplyOp>(&op)) {
      return true;
    }
  }

  return false;
}

SmallVector<Attribute> calculateOffsets(mlir::LoadOp loadOp) {
  SmallVector<Attribute> offsetAttr(loadOp.getIndices().size());

  for (size_t i = 0; i < loadOp.getIndices().size(); i++) {
    mlir::Value operand = loadOp.getIndices()[i];
    auto context = loadOp.getContext();

    if (isDefinedByOperation<mlir::AddIOp>(operand)) {
      auto addOp = operand.getDefiningOp<mlir::AddIOp>();

      if (isDefinedByOperation<mlir::ConstantOp>(addOp.lhs()) &&
          isDefinedByOperation<mlir::IndexCastOp>(addOp.rhs())) {
        offsetAttr[i] = IntegerAttr::get(IntegerType::get(context, 64),
                                         addOp.lhs()
                                             .getDefiningOp<mlir::ConstantOp>()
                                             .value()
                                             .cast<IntegerAttr>()
                                             .getInt());

        addOp.lhs().getDefiningOp()->erase();
      }

      if (isDefinedByOperation<mlir::IndexCastOp>(addOp.lhs()) &&
          isDefinedByOperation<mlir::ConstantOp>(addOp.rhs())) {
        offsetAttr[i] = IntegerAttr::get(IntegerType::get(context, 64),
                                         addOp.rhs()
                                             .getDefiningOp<mlir::ConstantOp>()
                                             .value()
                                             .cast<IntegerAttr>()
                                             .getInt());
        addOp.rhs().getDefiningOp()->erase();
      }

      addOp.erase();
    } else {
      offsetAttr[i] = IntegerAttr::get(IntegerType::get(context, 64), 0);
    }
  }
  return offsetAttr;
}

struct RaisingTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;
  RaisingTypeConverter(MLIRContext *context) {
    addConversion([&](mlir::MemRefType type) {
      return TempType::get(type.getElementType(), type.getShape());
    });
  }
};

class FuncOpRaising : public OpConversionPattern<FuncOp> {
public:
  using OpConversionPattern<FuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = funcOp.getLoc();

    auto context = funcOp->getContext();

    RaisingTypeConverter typeConverter(context);

    TypeConverter::SignatureConversion result(funcOp.getNumArguments());

    for (auto &en : llvm::enumerate(funcOp.getType().getInputs())) {

      if (en.value().isa<MemRefType>()) {
        mlir::MemRefType type = en.value().dyn_cast<mlir::MemRefType>();
        result.addInputs(en.index(),
                         TempType::get(type.getElementType(), type.getShape()));
      } else {
        result.addInputs(en.index(), en.value());
      }
    }

    auto funcResult = funcOp.getType().getResults();

    auto funcType = FunctionType::get(funcOp.getContext(),
                                      result.getConvertedTypes(), funcResult);

    auto newFuncOp =
        rewriter.create<FuncOp>(loc, "kernel", funcType, llvm::None);

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    rewriter.applySignatureConversion(&newFuncOp.getBody(), result);

    rewriter.eraseOp(funcOp);

    return success();
  }
};

class PopulateFuncWithStencilApply : public OpRewritePattern<FuncOp> {
public:
  PopulateFuncWithStencilApply(mlir::MLIRContext *context)
      : OpRewritePattern<FuncOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(FuncOp funcOp,
                                PatternRewriter &rewriter) const override {

    auto &entryBlock = funcOp.getBlocks().front();
    RaisingTypeConverter typeConverter(funcOp.getContext());

    // bail out if already populated with stencil.apply
    if (hasApplyOp(entryBlock)) {
      return success();
    }

    for (auto &arg : funcOp.getArguments()) {
      if (arg.use_empty()) {
        funcOp.eraseArgument(arg.getArgNumber());
      }
    }

    rewriter.setInsertionPointToStart(&entryBlock);

    SmallVector<mlir::Type> storeOpResultTypes;

    funcOp.walk([&](stencil::StoreResultOp storeOp) mutable {
      storeOpResultTypes.push_back(funcOp.getArgument(0).getType());
    });

    auto applyOp = rewriter.create<stencil::ApplyOp>(
        funcOp.getLoc(), storeOpResultTypes, funcOp.getArguments(), llvm::None,
        llvm::None);

    funcOp.walk([&](mlir::ReturnOp returnOp) { returnOp.erase(); });
    rewriter.setInsertionPointToEnd(&funcOp.getBlocks().front());

    rewriter.create<mlir::ReturnOp>(funcOp.getLoc(), applyOp.getResults());

    SmallVector<Operation *> opsToMove;

    for (auto &op : entryBlock.without_terminator()) {
      if (isOperationCalled<stencil::AccessOp>(&op)) {
        opsToMove.push_back(&op);
      }
    }

    for (auto &op : entryBlock.without_terminator()) {
      if (!isOperationCalled<stencil::AccessOp>(&op) &&
          !isOperationCalled<stencil::ApplyOp>(&op)) {
        opsToMove.push_back(&op);
      }
    }

    rewriter.setInsertionPointToStart(applyOp.getBody());
    SmallVector<Operation *> opsToErase;
    BlockAndValueMapping cloningMap;

    for (auto *op : opsToMove) {
      Operation *clone = rewriter.clone(*op, cloningMap);
      cloningMap.map(op->getResults(), clone->getResults());
      op->erase();
    }

    rewriter.setInsertionPointToEnd(applyOp.getBody());

    // auto returnType = ResultType::get(lastValue.getType());

    // auto storeResultOp = rewriter.create<stencil::StoreResultOp>(
    //     storeOp.getLoc(), returnType, storeOp.value());

    // rewriter.create<stencil::ReturnOp>(storeOp.getLoc(), llvm::None,

    // auto storeResultOp = rewriter.create<stencil::StoreResultOp>(
    //     applyOp.getLoc(), returnType, lastValue);

    // applyOp.walk([&](mlir::StoreOp storeOp) {
    //   auto returnType = ResultType::get(storeOp.value().getType());

    //   auto storeResultOp = rewriter.create<stencil::StoreResultOp>(
    //       storeOp.getLoc(), returnType, storeOp.value());

    //   storeOps.push_back(storeResultOp.getResult());

    //   storeOp.erase();
    // rewriter.create<stencil::ReturnOp>(storeOp.getLoc(), llvm::None,

    // });

    for (auto &arg : funcOp.getArguments()) {
      if (!arg.use_empty()) {
        arg.replaceUsesWithIf(
            applyOp.getBody()->getArgument(arg.getArgNumber()),
            [](mlir::OpOperand &op) {
              return !isOperationCalled<stencil::ApplyOp>(op.getOwner());
            });
      }
    }

    //
    SmallVector<mlir::Value> storeOps;

    applyOp.walk([&](stencil::StoreResultOp storeOp) mutable {
      storeOps.push_back(storeOp->getResult(0));
    });

    rewriter.create<stencil::ReturnOp>(applyOp.getLoc(), llvm::None,
                                       mlir::ValueRange(storeOps));

    auto funcType =
        FunctionType::get(funcOp.getContext(), funcOp.getArgumentTypes(),
                          applyOp.getResultTypes());
    funcOp.setType(funcType);

    return success();
  }
};

class LoadOpRaising : public OpConversionPattern<mlir::LoadOp> {
public:
  using OpConversionPattern<mlir::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::LoadOp loadOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto context = loadOp->getContext();
    RaisingTypeConverter typeConverter(context);

    auto convertedResultType =
        loadOp.getMemRefType().getElementType().cast<FloatType>();

    auto accessOp = rewriter.create<stencil::AccessOp>(
        loadOp.getLoc(), convertedResultType, loadOp.getOperand(0));

    auto offsetAttribute = ArrayAttr::get(calculateOffsets(loadOp), context);

    accessOp.offsetAttr(offsetAttribute);

    rewriter.replaceOp(loadOp, {accessOp});

    return success();
  }
};

class StoreOpRaising : public OpConversionPattern<mlir::StoreOp> {
public:
  using OpConversionPattern<mlir::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::StoreOp storeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto returnType = ResultType::get(storeOp.value().getType());
    rewriter.create<stencil::StoreResultOp>(storeOp.getLoc(), returnType,
                                            storeOp.value());

    // rewriter.create<stencil::ReturnOp>(storeOp.getLoc(), llvm::None,
    //                                    storeResultOp.getResult());

    rewriter.eraseOp(storeOp);

    return success();
  }
};

} // namespace

namespace {

class RaiseAffineToStencilPass
    : public RaiseAffineToStencilPassBase<RaiseAffineToStencilPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<mlir::stencil::StencilDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    for (auto attr : module.getAttrs()) {
      module.removeAttr(attr.first.strref());
    }

    module.walk([&](mlir::FuncOp funcOp) {
      auto builder =
          OpBuilder::atBlockBegin(&funcOp.getBody().getBlocks().front());

      auto zero = builder.create<ConstantIntOp>(funcOp.getLoc(), 0,
                                                builder.getI32Type());

      llvm::SmallVector<unsigned int> integerArgs;

      for (auto arg : funcOp.getArguments()) {
        if (arg.getType().isIntOrIndex()) {
          arg.replaceAllUsesWith(zero);
          integerArgs.push_back(arg.getArgNumber());
        }
      }

      funcOp.eraseArguments(integerArgs);
    });

    MLIRContext *context = &getContext();
    OwningRewritePatternList conversionPatterns;
    OwningRewritePatternList rewritePatterns;

    ConversionTarget target(*context);

    conversionPatterns.insert<FuncOpRaising>(module.getContext());
    conversionPatterns.insert<LoadOpRaising>(module.getContext());
    conversionPatterns.insert<StoreOpRaising>(module.getContext());

    rewritePatterns.insert<PopulateFuncWithStencilApply>(module.getContext());

    FrozenRewritePatternList frozenPatterns(std::move(rewritePatterns));

    target.addLegalDialect<StencilDialect>();
    target.addLegalDialect<mlir::StandardOpsDialect>();
    target.addDynamicallyLegalOp<FuncOp>([](FuncOp funcOp) {
      for (auto &en : llvm::enumerate(funcOp.getType().getInputs())) {
        if (en.value().isa<MemRefType>()) {
          return false;
        }
      }

      return true;
    });

    target.addIllegalOp<mlir::LoadOp>();
    target.addIllegalOp<mlir::StoreOp>();
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    if (failed(applyPartialConversion(module, target,
                                      std::move(conversionPatterns)))) {
      signalPassFailure();
    }

    module.walk([&frozenPatterns](FuncOp funcOp) {
      (void)applyOpPatternsAndFold(funcOp, frozenPatterns);
    });

    // if (failed(applyPatternsAndFoldGreedily(module,
    // std::move(rewritePatterns), 1))) {
    //   signalPassFailure();
    // }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createRaiseAffineToStencilPass() {
  return std::make_unique<RaiseAffineToStencilPass>();
}
