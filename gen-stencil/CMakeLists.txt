get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRTestTransforms
  MLIRAnalysis
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRExecutionEngine
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTransforms
  
  Stencil
  StencilToStandard
  RaiseAffineToStencil
)


add_llvm_executable(gen-stencil gen-stencil.cpp)

llvm_update_compile_flags(gen-stencil)
target_link_libraries(gen-stencil PRIVATE ${LIBS})

