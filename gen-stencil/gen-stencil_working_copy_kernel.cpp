#include "Conversion/LoopsToGPU/Passes.h"
#include "Conversion/RaiseAffineToStencil/Passes.h"
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
// #include "mlir/Transforms/InliningUtils.h"
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
#include <string>

#define OPS_READ 0
#define OPS_WRITE 1

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

static cl::opt<bool> enableAttrs("print-attr", cl::desc("Enable name attrs"),
                                 cl::init(false));

static cl::opt<bool> enableInline("inline", cl::desc("Enable stencil inlining"),
                                  cl::init(false));

static cl::opt<bool>
    enableEraseArgs("erase-args", cl::desc("Enable removing unused arguments"),
                    cl::init(false));

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
llvm::StringMap<bool> accMap;

mlir::MLIRContext context;

OpBuilder b(&context);

template <typename... Args>
std::string sstr(Args &&...args) {
  std::ostringstream sstr;
  (sstr << std::dec << ... << args);
  return sstr.str();
}

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
  std::string readableName;
  std::string name;
  mlir::Type type;
  mlir::stencil::FieldType fieldType;
  size_t idx;
  SmallVector<int64_t> begin;
  SmallVector<int64_t> end;
  SmallVector<int64_t> size;
  SmallVector<int64_t> base;

public:
  llvm::StringRef getReadableName() { return this->readableName; }
  void setReadableName(std::string name) { this->readableName = name; }

  llvm::StringRef getName() { return this->name; }
  mlir::Type getType() { return this->type; }
  mlir::stencil::FieldType getFieldType() { return this->fieldType; }

  void setName(std::string name) { this->name = name; }
  void setName(int64_t id) { setName(std::to_string(id)); }

  void setType(mlir::Type type) { this->type = type; }

  void setId(size_t id) { this->idx = id; }
  size_t getId() { return this->idx; }

  ArrayRef<int64_t> getBegin() { return this->begin; }
  void setBegin(SmallVector<int64_t> begin) { this->begin = begin; }

  ArrayRef<int64_t> getEnd() { return this->end; }
  void setEnd(SmallVector<int64_t> end) { this->end = end; }

  ArrayRef<int64_t> getBase() { return this->base; }
  void setBase(SmallVector<int64_t> end) { this->base = end; }

  ArrayRef<int64_t> getSize() { return this->size; }
  void setSize(SmallVector<int64_t> end) { this->size = end; }

  void calculateFieldType() {
    assert(this->begin.size() > 1 && "begin size > 1");

    SmallVector<int64_t> shape(this->begin.size(), -1);
    this->fieldType = mlir::stencil::FieldType::get(this->type, shape);
  }

  int64_t getBeginValue() { return this->begin.front(); }
  int64_t getEndValue() { return this->end.front(); }

  int64_t getFullRange() { return this->end.front() - this->begin.front(); }

  ArrayAttr getBeginAttr(MLIRContext &context) {
    SmallVector<mlir::Attribute> attr;

    for (auto &en : llvm::enumerate(this->begin)) {
      attr.push_back(
          mlir::IntegerAttr::get(IntegerType::get(&context, 64), en.value()));
    }

    return ArrayAttr::get(attr, &context);
  }

  ArrayAttr getEndAttr(MLIRContext &context) {
    SmallVector<mlir::Attribute> attr;

    for (auto &en : llvm::enumerate(this->size)) {
      attr.push_back(mlir::IntegerAttr::get(
          IntegerType::get(&context, 64), en.value() - this->end[en.index()]));
    }

    return ArrayAttr::get(attr, &context);
  }

  mlir::stencil::FieldType getSizeFieldType() {
    return mlir::stencil::FieldType::get(this->type, getSize());
  }

  mlir::stencil::FieldType getFullRangeFieldType() {
    return getSizeFieldType();
    // SmallVector<int64_t> shape(this->begin.size(), getFullRange());
    // return mlir::stencil::FieldType::get(this->type, shape);
  }

  KernelArgument() {}
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
std::vector<KernelArgument> arguments;
llvm::StringMap<std::string> argumentNameMap;

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
  assert(document.HasMember("dats") && "Root does not have datasets");
  assert(document.HasMember("kernels") && "Root does not have loops");

  return document;
}

mlir::FuncOp getMainFunc(OwningModuleRef &module) {
  return *(module->getBodyRegion()
               .getBlocks()
               .front()
               .getOps<mlir::FuncOp>()
               .begin());
}

/**
 * Finds the the first block in the
 * first function of the module
 **/
Block *findMainBlock(OwningModuleRef &module) {
  mlir::FuncOp funcOp = getMainFunc(module);

  return &funcOp.getBody().getBlocks().front();
}

mlir::OpBuilder mainBuilder(OwningModuleRef &module) {
  auto mainBlock = findMainBlock(module);
  return OpBuilder::atBlockTerminator(mainBlock);
}

void dumpLoadedStencilsMap() {
  for (auto &en : llvm::enumerate(loadedStencilsMap)) {
    llvm::errs() << " [arg: " + en.value().getKey() + " ]\n";
    en.value().getValue().dump();
    llvm::errs() << " [\\arg]\n";
  }
}

mlir::stencil::ApplyOp createCopyKernel(mlir::Value from, mlir::Value to,
                                        mlir::ArrayAttr lb, mlir::ArrayAttr ub,
                                        llvm::StringRef toName,
                                        OwningModuleRef &module) {

  auto builder = mainBuilder(module);

  auto resultTempType = to.getType().cast<stencil::FieldType>();

  SmallVector<int64_t> unrankedShape(resultTempType.getShape().size(), -1);

  auto unrankedResultTempType =
      stencil::TempType::get(resultTempType.getElementType(), unrankedShape);

  auto applyOp = builder.create<stencil::ApplyOp>(
      module->getLoc(), unrankedResultTempType, from, llvm::None, llvm::None);

  if (enableAttrs) {
    applyOp->setAttr(std::string("name"),
                     builder.getStringAttr(std::string("copy_kernel")));
  }

  builder.setInsertionPointToStart(applyOp.getBody());

  llvm::SmallVector<int64_t, 3> tempShape(from.getDefiningOp()
                                              ->getResult(0)
                                              .getType()
                                              .cast<stencil::TempType>()
                                              .getMemRefShape()
                                              .size(),
                                          0);

  auto accessOp = builder.create<stencil::AccessOp>(
      module->getLoc(), applyOp.getBody()->getArgument(0), tempShape);

  auto stencilStoreOp = builder.create<stencil::StoreResultOp>(
      module->getLoc(), accessOp.getResult());

  builder.create<stencil::ReturnOp>(applyOp.getLoc(), llvm::None,
                                    stencilStoreOp.getResult());

  builder.setInsertionPointAfter(applyOp);
  auto storeOp = builder.create<stencil::StoreOp>(
      module->getLoc(), applyOp->getResult(0), to, lb, ub);
  if (enableAttrs) {
    storeOp->setAttr(std::string("from"), builder.getStringAttr("copy_kernel"));
    storeOp->setAttr(std::string("to"), builder.getStringAttr(toName));
  }
}

mlir::stencil::ApplyOp lookupAndCallApplyKernel(
    llvm::StringRef name, SmallVector<mlir::Value> readArgs,
    OwningModuleRef &module, OwningModuleRef &kernelsModule) {

  auto builder = mainBuilder(module);

  auto kernelsFunctions = kernelsModule->getBody()->getOps<mlir::FuncOp>();
  mlir::FuncOp kernelToCall = nullptr;
  for (auto kf : kernelsFunctions) {
    if (kf.getName().contains(name)) {
      kernelToCall = kf;
      break;
    }
  }

  if (kernelToCall == nullptr) {
    llvm::errs() << "Kernel: " << name << "\n";
    assert(kernelToCall != nullptr && "Kernel not found.");
  }

  for (auto &ra : readArgs) {
    assert(ra != nullptr && "One of the input argument is null");
  }

  /*
    inline apply op from kernel function
    assuming that it contains only an apply op
  */
  SmallVector<mlir::stencil::ApplyOp, 2> applyOps(
      kernelToCall.getBody()
          .getBlocks()
          .front()
          .getOps<mlir::stencil::ApplyOp>());

  mlir::stencil::ApplyOp *applyOp = applyOps.begin();

  mlir::Operation *clonedOp = builder.clone(*applyOp->getOperation());
  if (enableAttrs) {
    clonedOp->setAttr(std::string("name"), builder.getStringAttr(name));
  }
  clonedOp->setLoc(OpaqueLoc::get<void *>(nullptr, module->getLoc()));

  assert(
      kernelToCall.getNumArguments() ==
          cast<mlir::stencil::ApplyOp>(clonedOp).getBody()->getNumArguments() &&
      "Invalid number of operands");
  clonedOp->setOperands(ValueRange(readArgs));

  return cast<mlir::stencil::ApplyOp>(clonedOp);
}

int processDatasets(Document &document, MLIRContext &context,
                    mlir::OwningModuleRef &module) {

  auto root = document.GetObject();

  const rapidjson::Value &datasets = document["dats"];
  assert(datasets.IsArray() && "Dats is not an array");

  llvm::SmallVector<mlir::Type> argTypes;

  for (auto &d : datasets.GetArray()) {
    KernelArgument kernelArgument;
    kernelArgument.setId(d["idx"].GetInt64());
    kernelArgument.setReadableName(d["name"].GetString());
    kernelArgument.setName(d["idx"].GetInt64());
    kernelArgument.setType(getDataType(d["type"].GetString(), &context));

    kernelArgument.setBegin(createI64VectorFromJsonArray(d["d_m"].GetArray()));

    kernelArgument.setEnd(createI64VectorFromJsonArray(d["d_p"].GetArray()));
    kernelArgument.setBase(createI64VectorFromJsonArray(d["base"].GetArray()));
    kernelArgument.setSize(createI64VectorFromJsonArray(d["size"].GetArray()));

    kernelArgument.calculateFieldType();

    arguments.push_back(kernelArgument);

    argTypes.push_back(kernelArgument.getFieldType());

    argumentNameMap[kernelArgument.getName()] =
        kernelArgument.getReadableName().str();
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

    if (enableAttrs) {
      castOp->setAttr(std::string("name"),
                      builder.getStringAttr(en.value().getReadableName()));
    }

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

  const rapidjson::Value &kernels = document["kernels"];
  assert(kernels.IsArray() && "Kernels is not an array");

  llvm::StringMap<mlir::stencil::StoreOp> storeMap;

  for (auto &l : kernels.GetArray()) {
    assert(l["name"].IsString() && "Name is not a string");
    llvm::StringRef name = l["name"].GetString();

    SmallVector<mlir::Value> readArgs;
    SmallVector<mlir::Value> writeArgs;
    SmallVector<llvm::StringRef> writeArgNames;
    SmallVector<llvm::StringRef> readArgNames;

    assert(l.HasMember("args") && "no args");
    assert(l["args"].IsArray() && "args is not an array");

    for (auto &arg : l["args"].GetArray()) {
      llvm::StringRef argName = std::to_string(arg["datidx"].GetInt64());
      int64_t acc = arg["acc"].GetInt64();

      if (acc == OPS_READ) {
        readArgs.push_back(loadedStencilsMap[argName]);
        readArgNames.push_back(argName);
      } else {
        writeArgNames.push_back(argName);
        writeArgs.push_back(loadedStencilsMap[argName]);
      }
    }

    auto callOp =
        lookupAndCallApplyKernel(name, readArgs, module, kernelsModule);

    auto mainBlock = findMainBlock(module);

    auto builder = OpBuilder::atBlockTerminator(mainBlock);

    assert(l.HasMember("range") && "no range on kernel");
    auto range = createI64VectorFromJsonArray(l["range"].GetArray());

    SmallVector<int64_t> beginValues;
    SmallVector<int64_t> endValues;

    for (auto &en : llvm::enumerate(range)) {
      if (en.index() % 2 == 0) {
        beginValues.push_back(en.value());
      } else {
        endValues.push_back(en.value());
      }
    }

    auto begin = createI64ArrayAttr(beginValues);
    auto end = createI64ArrayAttr(endValues);

    for (auto &en : llvm::enumerate(writeArgNames)) {
      auto argName = en.value();
      auto index = en.index();
      auto storeOp = builder.create<stencil::StoreOp>(
          callOp->getLoc(), callOp->getResult(index), castMap[argName], begin,
          end);

      if (enableAttrs) {
        storeOp->setAttr(
            std::string("from"),
            builder.getStringAttr(
                callOp->getAttr("name").cast<mlir::StringAttr>().getValue()));

        storeOp->setAttr(std::string("to"),
                         builder.getStringAttr(argumentNameMap[argName]));
      }

      storeMap[argName] = storeOp;
      loadedStencilsMap[argName] = callOp->getResult(index);
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

  // add kernels to the new module
  for (auto kernel : module->getBody()->getOps<mlir::FuncOp>()) {
    auto clonedKernel = kernel.clone();
    newModule->push_back(clonedKernel);
  }

  processLoops(document, context, newModule, module);

  mlir::PassManager pm(&context);

  // erase non-main functions
  mlir::PassManager pmTidy(&context);
  pmTidy.addPass(mlir::createEraseNonStencilProgramsPass());
  pmTidy.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(pmTidy.run(*newModule))) {
    llvm::errs() << "Failed to run clean up passes";
    return -1;
  }

  // remove unnsecessary stores for intermediary stencils results
  if (enableInline) {
    newModule->walk([&](mlir::stencil::StoreOp storeOp) {
      if (!storeOp.temp().hasOneUse()) {
        storeOp.erase();
      }
    });
  }

  newModule->walk([&](mlir::stencil::StoreOp storeOp) {
    // storeOp.dump();
    bool isStoreOverlappingInput =
        llvm::any_of(storeOp.field().getUses(), [&](mlir::OpOperand &op) {
          return isa<mlir::stencil::LoadOp>(op.getOwner());
        });

    if (isStoreOverlappingInput && enableInline) {
      auto builder = mainBuilder(newModule);
      auto originalCastOp = storeOp.field().getDefiningOp();

      auto funcOp = getMainFunc(newModule);

      bool found = false;
      for (auto &en : llvm::enumerate(funcOp.getArguments())) {
        auto arg = en.value();
        auto argIndex = en.index();

        if (arg.use_empty()) {

          auto castOp = builder.clone(*originalCastOp);

          castOp->setOperand(0, arg);
          auto toName = argumentNameMap[std::to_string(argIndex)];

          if (enableAttrs) {
            castOp->setAttr(std::string("name"), builder.getStringAttr(toName));
          }

          auto bufferOp = builder.create<mlir::stencil::BufferOp>(
              newModule->getLoc(), storeOp.temp());
          createCopyKernel(bufferOp.getResult(), castOp->getResult(0),
                           storeOp.lbAttr(), storeOp.ubAttr(), toName,
                           newModule);

          found = true;
          break;
        }
      }

      if (!found) {
        storeOp.emitError(
            "There are no unused arguments for creating copy kernel");
        mlir::failure(true);
      } else {
        storeOp.erase();
      }
    }
  });

  // erase func unused arguments from main func
  SmallVector<unsigned> argumentIndexesToDelete;
  mlir::FuncOp funcOp = getMainFunc(newModule);

  for (auto &arg : funcOp.getArguments()) {
    if (arg.use_empty()) {
      argumentIndexesToDelete.push_back(arg.getArgNumber());
    }
  }

  if (enableEraseArgs) {
    funcOp.eraseArguments(argumentIndexesToDelete);
  }

  // verify all applyOps has store op
  newModule->walk([&](mlir::stencil::ApplyOp applyOp) {
    auto hasUnusedResult = false;
    for (auto res : applyOp.getResults()) {
      if (res.use_empty()) {
        hasUnusedResult = true;
        break;
      }
    }

    if (hasUnusedResult) {
      applyOp.emitOpError("expected to have usage (storage)");
    }
  });

  if (enableInline) {
    mlir::PassManager pmInline(&context);
    pmInline.addNestedPass<mlir::FuncOp>(mlir::createStencilInliningPass());
    pmInline.addPass(mlir::createCSEPass());
    pmInline.addPass(mlir::createCanonicalizerPass());

    if (mlir::failed(pmInline.run(*newModule))) {
      llvm::errs() << "Failed to run stencil inlining";
      return -1;
    }
  }

  if (emitAction >= Action::DumpStencilShapes) {
    pm.addNestedPass<mlir::FuncOp>(mlir::createShapeInferencePass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }

  if (emitAction >= Action::DumpStandard) {
    pm.addPass(mlir::createConvertStencilToStandardPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }

  mlir::LowerToLLVMOptions lowerToLLVMOptions;

  lowerToLLVMOptions.emitCWrappers = true;

  if (emitAction >= Action::DumpMLIRLLVM) {
    pm.addPass(mlir::createLowerToCFGPass());
    pm.addPass(mlir::createLowerToLLVMPass(lowerToLLVMOptions));
  }

  applyPassManagerCLOptions(pm);

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