#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <numeric>

using namespace cv;
using namespace std;

struct TRTDestroy
{
    template< class T >
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template< class T >
using TRTUniquePtr = std::unique_ptr< T, TRTDestroy >;
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override {
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << "n";
        }
    }
} gLogger;

void preprocessImage(const string& img_path, float* gpu_input, nvinfer1::Dims& dims);
size_t getSizeByDim(const nvinfer1::Dims& dims);
vector< std::string > getClassNames(const std::string& imagenet_classes);
void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                    TRTUniquePtr< nvinfer1::IExecutionContext >& context);
void postprocessResults(float* gpu_output, const nvinfer1::Dims &dims, int batch_size);


// https://learnopencv.com/how-to-run-inference-using-tensorrt-c-api/


int main(int argc, char* argv[]) {
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " model.onnx image.jpgn\n";
        return -1;
    }
    std::string model_path(argv[1]);
    std::string image_path(argv[2]);
    int batch_size = 1;


    TRTUniquePtr< nvinfer1::ICudaEngine > engine{nullptr};
    TRTUniquePtr< nvinfer1::IExecutionContext > context{nullptr};
    parseOnnxModel(model_path, engine, context);

    std::vector< nvinfer1::Dims > input_dims; // we expect only one input
    std::vector< nvinfer1::Dims > output_dims; // and one output
    std::vector< void* > buffers(engine->getNbBindings()); // buffers for input and output data
    for (size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);
        cudaMalloc(&buffers[i], binding_size);
        if (engine->bindingIsInput(i))
        {
            input_dims.emplace_back(engine->getBindingDimensions(i));
        }
        else
        {
            output_dims.emplace_back(engine->getBindingDimensions(i));
        }
    }
    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Expect at least one input and one output for networkn\n";
        return -1;
    }

    // preprocess input data
    preprocessImage(image_path, (float*)buffers[0], input_dims[0]);
    // inference
    context->enqueue(batch_size, buffers.data(), 0, nullptr);
    // post-process results
    postprocessResults((float *) buffers[1], output_dims[0], batch_size);

    for (void* buf : buffers)
    {
        cudaFree(buf);
    }
    return 0;
}

void preprocessImage(const string& img_path, float* gpu_input, nvinfer1::Dims& dims){
    Mat img = imread(img_path);
    if (img.empty()){
        cerr << "Input image " << img_path << " load failed.\n";
    }

    // Upload to GPU
    cuda::GpuMat gpu_frame;
    gpu_frame.upload(img);

    auto input_width = dims.d[2];
    auto input_height = dims.d[1];
    auto channels = dims.d[0];

    auto input_size = Size(input_width, input_height);

    cuda::GpuMat resized;
    cuda::resize(gpu_frame, resized, input_size, 0, 0, INTER_LINEAR);

    //rar normalize kode
    cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

    // To tensor
    vector<cuda::GpuMat> chw;
    for (size_t i = 0; i < channels; i++){
        chw.emplace_back(cuda::GpuMat(input_size, CV_32FC1, gpu_input + i*input_width*input_width));
    }
    cuda::split(flt_image, chw);
}

void postprocessResults(float* gpu_output, const nvinfer1::Dims &dims, int batch_size){
    auto classes = getClassNames("../classes.txt");

    vector<float> cpu_output(getSizeByDim(dims) * batch_size);

    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cout << "test1\n";
    // calculate softmax
    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) {return std::exp(val);});
    auto sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0.0);
    // find top classes predicted by the model
    std::vector< int > indices(getSizeByDim(dims) * batch_size);
    // generate sequence 0, 1, 2, 3, ..., 999
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2) {return cpu_output[i1] > cpu_output[i2];});
    // print results
    int i = 0;
    cout << "test2\n";
    while (cpu_output[indices[i]] / sum > 0.005)
    {
        if (classes.size() > indices[i])
        {
            std::cout << "class: " << classes[indices[i]] << " | ";
        }
        std::cout << "confidence: " << 100 * cpu_output[indices[i]] / sum << "% | index: " << indices[i] << "n";
        ++i;
    }
    cout << "test3\n";
}

size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

vector< std::string > getClassNames(const std::string& imagenet_classes)
{
    std::ifstream classes_file(imagenet_classes);
    std::vector< std::string > classes;
    if (!classes_file.good())
    {
        std::cerr << "ERROR: can't read file with class names";
        return classes;
    }
    std::string class_name;
    cout << "Loading class: ";
    while (std::getline(classes_file, class_name))
    {
        classes.push_back(class_name);
        cout << class_name << ", ";
    }
    cout << std::endl;
    return classes;
}
void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                    TRTUniquePtr< nvinfer1::IExecutionContext >& context) {
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast< int >(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }

    TRTUniquePtr< nvinfer1::IBuilderConfig > config{builder->createBuilderConfig()};
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);
    // use FP16 mode if possible
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // we have only one image in batch
    builder->setMaxBatchSize(1);

    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());
}