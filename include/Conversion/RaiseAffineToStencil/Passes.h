#ifndef CONVERSION_AFFINETOSTENCIL_AFFINETOSTENCIL_PASSES_H
#define CONVERSION_AFFINETOSTENCIL_AFFINETOSTENCIL_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<Pass> createRaiseAffineToStencilPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Conversion/RaiseAffineToStencil/Passes.h.inc"

} // namespace mlir

#endif // CONVERSION_AFFINETOSTENCIL_AFFINETOSTENCIL_PASSES_H
