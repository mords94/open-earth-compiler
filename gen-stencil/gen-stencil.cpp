#include "Conversion/StencilToStandard/Passes.h"
#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include <rapidjson/istreamwrapper.h>

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

#define OPS_READ "OPS_READ"
#define OPS_WRITE "OPS_WRITE"

using namespace mlir::stencil;
using namespace mlir;
using namespace rapidjson;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<kernel mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string> jsonFileName(cl::Positional,
                                         cl::desc("<json file>"), cl::init("-"),
                                         cl::value_desc("json filename"));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

namespace {
enum Action {
  DumpKernelsOnly,
  DumpStencil,
  DumpStencilShapes,
  DumpStandard,
  DumpMLIRLLVM,
  DumpLLVMIR,
  RunJIT
};
}
static cl::opt<enum Action> emitAction(
    "emit", cl::init(DumpStencil),
    cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpKernelsOnly, "kernels-only",
                          "output the processed kernels in stencil MLIR")),
    cl::values(clEnumValN(DumpStencil, "mlir-stencil",
                          "output the stencil MLIR dump")),
    cl::values(clEnumValN(DumpStencilShapes, "mlir-stencil-shapes",
                          "output the stencil with shapes MLIR dump")),
    cl::values(clEnumValN(DumpStandard, "mlir-std",
                          "output the standard MLIR dump")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(
        clEnumValN(RunJIT, "jit",
                   "JIT the code and run it by invoking the main function")));

llvm::StringMap<mlir::FuncOp> functionMap;

mlir::MLIRContext context;

OpBuilder b(&context);

mlir::OwningModuleRef createModule() {
  auto theModule = mlir::ModuleOp::create(b.getUnknownLoc());

  if (failed(mlir::verify(theModule))) {
    theModule.emitError("module verification error");
    return nullptr;
  }

  return theModule;
}

int loadMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

void dumpJson(Document &document) {
  StringBuffer strbuf;
  strbuf.Clear();

  Writer<StringBuffer> writer(strbuf);
  document.Accept(writer);

  std::string ownShipRadarString = strbuf.GetString();
  std::cout << "**********************************************"
            << ownShipRadarString << std::endl;
}
namespace {
enum { DOUBLE = 0, FLOAT = 1 };
const char *types[] = {"double", "float"};

mlir::Type getDataType(llvm::StringRef typeName, MLIRContext *context) {
  if (typeName.equals(types[DOUBLE])) {
    return mlir::Float64Type::get(context);
  }

  if (typeName.equals(types[FLOAT])) {
    return mlir::Float32Type::get(context);
  }
}

/**
 * Creates an attribute of integere array
 * @param size - size of the array
 * @param value - initial value of the integer attr
 **/
ArrayAttr createI64ArrayAttr(int size, int value) {
  SmallVector<mlir::Attribute> attr(
      size, mlir::IntegerAttr::get(IntegerType::get(&context, 64), value));

  return ArrayAttr::get(attr, &context);
}

ArrayAttr createI64ArrayAttr(SmallVector<int64_t> elements) {
  SmallVector<mlir::Attribute> attr;

  for (auto &value : elements) {
    attr.push_back(IntegerAttr::get(IntegerType::get(&context, 64), value));
  }

  return ArrayAttr::get(attr, &context);
}

SmallVector<int64_t> createI64VectorFromJsonArray(
    rapidjson::GenericArray<true, rapidjson::Value> array) {

  SmallVector<int64_t> vector;

  for (auto &arg : array) {
    vector.push_back(arg.GetInt64());
  }

  return vector;
}

class KernelArgument {

private:
  llvm::StringRef name;
  mlir::Type type;
  mlir::stencil::FieldType fieldType;
  size_t idx;
  SmallVector<int64_t> begin;
  SmallVector<int64_t> end;

public:
  llvm::StringRef getName() { return this->name; }
  mlir::Type getType() { return this->type; }
  mlir::stencil::FieldType getFieldType() { return this->fieldType; }

  void setName(llvm::StringRef name) { this->name = name; }
  void setType(mlir::Type type) { this->type = type; }

  void setId(size_t id) { this->idx = id; }
  size_t getId() { return this->idx; }

  ArrayRef<int64_t> getBegin() { return this->begin; }
  void setBegin(SmallVector<int64_t> begin) { this->begin = begin; }

  ArrayRef<int64_t> getEnd() { return this->end; }
  void setEnd(SmallVector<int64_t> end) { this->end = end; }

  void calculateFieldType() {
    assert(this->begin.size() > 1 && "begin size > 1");

    SmallVector<int64_t> shape(this->begin.size(), -1);
    this->fieldType = mlir::stencil::FieldType::get(this->type, shape);
  }

  int64_t getBeginValue() { return this->begin.front(); }
  int64_t getEndValue() { return this->end.front(); }

  int64_t getFullRange() { return this->end.front() - this->begin.front(); }

  ArrayAttr getBeginAttr(MLIRContext &context) {
    SmallVector<mlir::Attribute> attr(
        this->begin.size(),
        mlir::IntegerAttr::get(IntegerType::get(&context, 64),
                               this->begin.front()));

    return ArrayAttr::get(attr, &context);
  }

  ArrayAttr getEndAttr(MLIRContext &context) {
    SmallVector<mlir::Attribute> attr(
        this->end.size(), mlir::IntegerAttr::get(IntegerType::get(&context, 64),
                                                 this->end.front()));

    return ArrayAttr::get(attr, &context);
  }

  mlir::stencil::FieldType getFullRangeFieldType() {
    SmallVector<int64_t> shape(this->begin.size(), getFullRange());
    return mlir::stencil::FieldType::get(this->type, shape);
  }

  KernelArgument() {}
  KernelArgument(llvm::StringRef name, mlir::Type type, size_t idx,
                 SmallVector<int64_t> begin, SmallVector<int64_t> end)
      : name(name), idx(idx), type(type), begin(begin), end(end) {
    calculateFieldType();
  }
};

class KernelApply {

private:
  llvm::StringRef name;
  mlir::FuncOp kernelCall;
  SmallVector<mlir::Value *> readArgs;
  SmallVector<mlir::Value *> writeArgs;

public:
  llvm::StringRef getName() { return this->name; }
  void setName(llvm::StringRef name) { this->name = name; }
};

} // namespace

llvm::StringMap<mlir::stencil::CastOp> castMap;
llvm::StringMap<mlir::Value> loadedStencilsMap;
SmallVector<KernelArgument> arguments;

Document parseJson() {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(jsonFileName);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open json file: " << EC.message() << "\n";
  }

  Document document;

  document.Parse(fileOrErr.get().get()->getBufferStart());

  if (document.HasParseError()) {
    std::cout << "Error  : " << document.GetParseError() << '\n'
              << "Offset : " << document.GetErrorOffset() << '\n';
  }

  assert(document.IsObject() && "Root is not an object");
  assert(document.HasMember("datasets") && "Root does not have datasets");
  assert(document.HasMember("loops") && "Root does not have loops");

  return document;
}

/**
 * Finds the the first block in the
 * first function of the module
 **/
Block *findMainBlock(OwningModuleRef &module) {
  mlir::FuncOp funcOp = *(module->getBodyRegion()
                              .getBlocks()
                              .front()
                              .getOps<mlir::FuncOp>()
                              .begin());

  return &funcOp.getBody().getBlocks().front();
}

mlir::CallOp lookupAndCallApplyKernel(llvm::StringRef name,
                                      SmallVector<mlir::Value> readArgs,
                                      OwningModuleRef &module,
                                      OwningModuleRef &kernelsModule) {

  auto mainBlock = findMainBlock(module);
  auto builder = OpBuilder::atBlockTerminator(mainBlock);

  auto kernelsFunctions = kernelsModule->getBody()->getOps<mlir::FuncOp>();

  mlir::FuncOp kernelToCall = nullptr;
  for (auto kf : kernelsFunctions) {
    if (kf.getName().contains(name)) {
      kernelToCall = kf;
      break;
    }
  }

  assert(kernelToCall != nullptr && "Kernel not found.");

  llvm::SmallVector<mlir::Value> args;

  auto callOp = builder.create<mlir::CallOp>(module->getLoc(), kernelToCall,
                                             ValueRange(readArgs));

  return callOp;
}

int processDatasets(Document &document, MLIRContext &context,
                    mlir::OwningModuleRef &module) {

  auto root = document.GetObject();

  const rapidjson::Value &datasets = document["datasets"];
  assert(datasets.IsArray() && "Datasets is not an array");

  llvm::SmallVector<mlir::Type> argTypes;

  for (auto &d : datasets.GetArray()) {

    KernelArgument kernelArgument;
    kernelArgument.setId(d["idx"].GetInt64());
    kernelArgument.setName(d["name"].GetString());
    kernelArgument.setType(getDataType(d["type"].GetString(), &context));

    kernelArgument.setBegin(
        createI64VectorFromJsonArray(d["begin"].GetArray()));
    kernelArgument.setEnd(createI64VectorFromJsonArray(d["end"].GetArray()));
    kernelArgument.calculateFieldType();

    arguments.push_back(kernelArgument);

    argTypes.push_back(kernelArgument.getFieldType());
  }

  auto builder = OpBuilder::atBlockBegin(
      &module.get().getBodyRegion().getBlocks().front());

  auto funcType = builder.getFunctionType(argTypes, llvm::None);

  auto funcOp = builder.create<mlir::FuncOp>(module.get().getLoc(), "jit",
                                             funcType, llvm::None);

  funcOp->setAttr(mlir::stencil::StencilDialect::getStencilProgramAttrName(),
                  builder.getBoolAttr(true));

  funcOp.addEntryBlock();

  builder.setInsertionPointToStart(&funcOp.getBlocks().front());

  for (auto &en : llvm::enumerate(arguments)) {
    auto castOp = builder.create<stencil::CastOp>(
        funcOp.getLoc(), en.value().getFullRangeFieldType(),
        funcOp.getArgument(en.index()), en.value().getBeginAttr(context),
        en.value().getEndAttr(context));

    auto resultTempType =
        castOp.getResult().getType().cast<stencil::FieldType>();

    SmallVector<int64_t> unrankedShape(resultTempType.getShape().size(), -1);

    auto unrankedResultTempType =
        stencil::TempType::get(resultTempType.getElementType(), unrankedShape);

    auto loadOp = builder.create<stencil::LoadOp>(
        castOp.getLoc(), unrankedResultTempType, castOp.getResult());

    castMap.insert({en.value().getName(), castOp});
    loadedStencilsMap.insert({en.value().getName(), loadOp.getResult()});
  }

  builder.create<mlir::ReturnOp>(funcOp.getLoc());

  return 0;
}

int processLoops(Document &document, MLIRContext &context,
                 mlir::OwningModuleRef &module,
                 mlir::OwningModuleRef &kernelsModule) {

  auto root = document.GetObject();

  const rapidjson::Value &loops = document["loops"];
  assert(loops.IsArray() && "Loops is not an array");

  llvm::StringMap<mlir::stencil::StoreOp> storeMap;

  for (auto &l : loops.GetArray()) {
    assert(l["name"].IsString() && "Name is not a string");
    llvm::StringRef name = l["name"].GetString();

    SmallVector<mlir::Value> readArgs;
    SmallVector<mlir::Value> writeArgs;
    SmallVector<llvm::StringRef> writeArgNames;
    SmallVector<llvm::StringRef> readArgNames;

    for (auto &arg : l["args"].GetArray()) {
      llvm::StringRef argName = arg["name"].GetString();
      llvm::StringRef acc = arg["acc"].GetString();

      if (llvm::StringRef(acc).equals(OPS_READ)) {
        readArgs.push_back(loadedStencilsMap[argName]);
        readArgNames.push_back(argName);
      } else {
        writeArgNames.push_back(argName);
        writeArgs.push_back(loadedStencilsMap[argName]);
      }
    }

    // TODO: Figure out unused
    // for (auto &argName : readArgNames) {
    //   if (storeMap.count(argName) == 1) {
    //     storeMap[argName].erase();
    //   }
    // }

    auto callOp =
        lookupAndCallApplyKernel(name, readArgs, module, kernelsModule);

    auto mainBlock = findMainBlock(module);
    auto builder = OpBuilder::atBlockTerminator(mainBlock);

    auto begin =
        createI64ArrayAttr(createI64VectorFromJsonArray(l["begin"].GetArray()));
    auto end =
        createI64ArrayAttr(createI64VectorFromJsonArray(l["end"].GetArray()));

    for (auto &argName : writeArgNames) {
      /*auto storeOp = */ builder.create<stencil::StoreOp>(
          callOp.getLoc(), callOp.getResult(0), castMap[argName], begin, end);

      // storeMap[argName] = storeOp;
      loadedStencilsMap[argName] = callOp.getResult(0);
    }
  }

  return 0;
}

int translateAndPrintLLVMIR(mlir::ModuleOp module) {
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::outs() << *llvmModule << "\n";
  return 0;
}

int main(int argc, char **argv) {

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "ops compiler\n");

  context.getOrLoadDialect<mlir::stencil::StencilDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<mlir::gpu::GPUDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  mlir::OwningModuleRef module;
  mlir::OwningModuleRef newModule = createModule();
  loadMLIR(context, module);

  Document document = parseJson();
  processDatasets(document, context, newModule);

  for (auto kernel : module->getBody()->getOps<mlir::FuncOp>()) {
    auto clonedKernel = kernel.clone();
    // clonedKernel.setPrivate();
    newModule->push_back(clonedKernel);
  }

  processLoops(document, context, newModule, module);

  mlir::PassManager pm(&context);

  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createEraseNonStencilProgramsPass());

  if (emitAction >= Action::DumpStencilShapes) {
    pm.addNestedPass<mlir::FuncOp>(mlir::createShapeInferencePass());
  }

  if (emitAction >= Action::DumpStandard) {
    pm.addPass(mlir::createConvertStencilToStandardPass());
    if (enableOpt) {
      pm.addPass(mlir::createCSEPass());
      pm.addPass(mlir::createCanonicalizerPass());
    }
    pm.addPass(mlir::createLowerAffinePass());
  }
  applyPassManagerCLOptions(pm);

  mlir::LowerToLLVMOptions lowerToLLVMOptions;

  lowerToLLVMOptions.emitCWrappers = true;

  if (emitAction >= Action::DumpMLIRLLVM) {
    pm.addPass(mlir::createLowerToCFGPass());
    pm.addPass(mlir::createLowerToLLVMPass(lowerToLLVMOptions));
  }

  if (mlir::failed(pm.run(*newModule))) {
    llvm::errs() << "Failed conversion passes";
    return -1;
  }

  if (emitAction <= DumpMLIRLLVM) {
    newModule->print(llvm::outs());
    return 0;
  } else if (emitAction == DumpLLVMIR) {
    return translateAndPrintLLVMIR(*module);
  }

  return 0;
}