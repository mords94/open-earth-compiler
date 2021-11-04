#ifndef CONVERSION_AFFINETOSTENCIL_AFFINETOSTENCIL
#define CONVERSION_AFFINETOSTENCIL_AFFINETOSTENCIL

include "mlir/Pass/PassBase.td"

def RaiseAffineToStencilPass : Pass<"raise-affine-to-stencil", "ModuleOp"> {
  let summary = "Raises affine to stencil";
  let constructor = "mlir::createRaiseAffineToStencilPass()";
}

#endif // CONVERSION_AFFINETOSTENCIL_AFFINETOSTENCIL
