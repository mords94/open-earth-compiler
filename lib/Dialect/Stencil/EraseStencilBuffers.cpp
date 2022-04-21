

#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "mlir/Pass/Pass.h"

#include <iostream>

using namespace mlir;
using namespace stencil;

namespace {

struct StencilBufferErasePass
    : public ::mlir::PassWrapper<StencilBufferErasePass,
                                 OperationPass<FuncOp>> {
  void runOnOperation() override {
    getOperation().walk([&](BufferOp bufferOp) {
      bufferOp.replaceAllUsesWith(bufferOp.getOperand());
      bufferOp.erase();
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createEraseStencilBuffersPass() {
  return std::make_unique<StencilBufferErasePass>();
}
