#ifndef CONVERSION_RAISEAFFINETOSTENCIL_PASSDETAIL_H_
#define CONVERSION_RAISEAFFINETOSTENCIL_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "Conversion/RaiseAffineToStencil/Passes.h.inc"

} // end namespace mlir

#endif // CONVERSION_RAISEAFFINETOSTENCIL_PASSDETAIL_H_
