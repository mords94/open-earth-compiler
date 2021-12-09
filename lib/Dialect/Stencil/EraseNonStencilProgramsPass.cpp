

#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "mlir/Pass/Pass.h"

#include <iostream>

using namespace mlir;
using namespace stencil;

namespace {

struct NonStencilFuncEraserPass
    : public ::mlir::PassWrapper<NonStencilFuncEraserPass,
                                 OperationPass<ModuleOp>> {
  void runOnOperation() override {
    getOperation().walk([](FuncOp funcOp) {
      if (!StencilDialect::isStencilProgram(funcOp)) {
        funcOp.erase();
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createEraseNonStencilProgramsPass() {
  return std::make_unique<NonStencilFuncEraserPass>();
}
