#ifndef MLIR_DIALECT_STENCIL_STENCILOPS_H
#define MLIR_DIALECT_STENCIL_STENCILOPS_H

#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace stencil {

namespace {
/// Helper methods to convert between vectors and attributes
ArrayAttr convertSmallVectorToArrayAttr(ArrayRef<int64_t> vector,
                                        MLIRContext *context) {
  SmallVector<Attribute, 3> result;
  for (int64_t value : vector) {
    result.push_back(IntegerAttr::get(IntegerType::get(64, context), value));
  }
  return ArrayAttr::get(result, context);
}
SmallVector<int64_t, 3> convertArrayAttrToSmallVector(const ArrayAttr &array,
                                                      MLIRContext *context) {
  SmallVector<int64_t, 3> result;
  for (auto &attr : array) {
    result.push_back(attr.cast<IntegerAttr>().getValue().getSExtValue());
  }
  return result;
}
} // namespace

/// Retrieve the class declarations generated by TableGen
#define GET_OP_CLASSES
#include "Dialect/Stencil/StencilOps.h.inc"

} // namespace stencil
} // namespace mlir

#endif // MLIR_DIALECT_STENCIL_STENCILOPS_H
