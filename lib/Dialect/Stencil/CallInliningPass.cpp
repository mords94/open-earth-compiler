#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;

namespace {

void inlineCalls(stencil::CallOp callOp) {
  ArrayRef<int64_t> callOffset = callOp.getOffset();
  FuncOp funcOp = callOp.getCallee().clone();
  OpBuilder builder(callOp);

  // Update offsets on function clone
  funcOp.walk([&](stencil::AccessOp accessOp) {
    ArrayRef<int64_t> current = accessOp.getOffset();

    ArrayAttr sum = builder.getI64ArrayAttr(llvm::makeArrayRef(
        {current[0] + callOffset[0], current[1] + callOffset[1],
         current[2] + callOffset[2]}));

    accessOp.setAttr(accessOp.getOffsetAttrName(), sum);
  });

  // Replace the arguments of the clone with the call op operands
  for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
    Value *argument = funcOp.getArgument(i);
    Value *replacement = callOp.getOperand(i);
    argument->replaceAllUsesWith(replacement);
  }

  // Insert the body of the function clone
  // TODO: currently we only support functions with a single region/block
  assert(funcOp.getOperation()->getNumRegions() == 1);
  assert(funcOp.getOperation()->getRegion(0).getBlocks().size() == 1);
  callOp.getOperation()->getBlock()->getOperations().splice(
      Block::iterator{callOp},
      funcOp.getOperation()->getRegion(0).front().getOperations());

  // Remove the return op
  // TODO: support multiple return values
  ReturnOp returnOp = cast<ReturnOp>(*std::prev(Block::iterator(callOp)));
  assert(returnOp.getNumOperands() == 1);
  Value *result = returnOp.getOperand(0);
  Value *old = callOp.getResult();
  old->replaceAllUsesWith(result);

  // Remove the call and the return operations
  returnOp.erase();
  callOp.erase();
}

struct CallInliningPass : public ModulePass<CallInliningPass> {
  void runOnModule() override;
};

void CallInliningPass::runOnModule() {
  ModuleOp moduleOp = getModule();
  
  // Walk all the functions of a module
  moduleOp.walk([](FuncOp funcOp) {
    // Walk the body of the function and inline all calls
    if (stencil::StencilDialect::isStencilFunction(funcOp))
      funcOp.walk([](stencil::CallOp callOp) { inlineCalls(callOp); });
  });

  // Walk all the apply ops
  moduleOp.walk([](stencil::ApplyOp applyOp) {
    // Walk the body of the apply operation and inline all calls
    applyOp.walk([](stencil::CallOp callOp) { inlineCalls(callOp); });
  });
}

} // namespace

static PassRegistration<CallInliningPass> pass("stencil-call-inlining",
                                               "Inline stencil function calls");
