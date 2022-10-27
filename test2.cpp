#include <iostream>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <numeric>
#include "tensorrt_utils.h"

using namespace std;
using namespace nvinfer1;

static TestLogger logger;

// Good example code - https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/TensorRT-introduction/simpleOnnx.cpp


void uploadToBuffer(cv::Mat &img, float *gpu_input, const Dims dims, bool showImg);
void postprocessAndDisplay(cv::Mat &img, float *gpu_output, const Dims dims, float treshold);

// Usage - ./TensorFlowTest {engine file}.engine {image file}.jpg
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " model.engine image.jpgn\n";
        return -1;
    }

    std::string model_path(argv[1]);
    std::string image_path(argv[2]);

    std::ifstream opened_file_stream;
    opened_file_stream.open(model_path);

    std::printf("Loading network from %s\n", model_path.c_str());


    // Check netowrk size
    opened_file_stream.seekg(0, std::ios::end);
    const int model_size = opened_file_stream.tellg();
    opened_file_stream.seekg(0, std::ios::beg);

    std::printf("Model size is %d\n", model_size);

    void* model_mem = malloc(model_size);
    if (!model_mem){
        std::printf("Failed to allocate %i bytes to deserialize model\n", model_size);
    }
    opened_file_stream.read((char*)model_mem, model_size);


    //SECTION 1 - Creating CUDA Engine and loading .engine file | Works

    ICudaEngine* engine = nullptr;
    IRuntime* runtime = createInferRuntime(logger);

    engine = runtime->deserializeCudaEngine(model_mem, model_size);

    if (!engine){
        cout << "Engine creation failed\n";
        return 1;
    }

    printf("Engine takes in %i tensors.\n", engine->getNbBindings());

    // Free memory
    delete runtime;
    free(model_mem);

    // SECTION 2 - Create dimension data

    IExecutionContext *context = engine->createExecutionContext();

    if (engine->getNbBindings() != 2){
        printf("Engine seems to have more than two tensors - exiting");
        return 1;
    }

    int32_t inputIndex = engine->getBindingIndex("images");
    int32_t outputIndex = engine->getBindingIndex("output0");

    // Found indexes using function.
    // Input - 'images'
    // Output - 'output0'

    void* buffers[2];

    for (int i = 0; i < engine->getNbBindings(); i++){
        Dims tensor = engine->getBindingDimensions(i);
        size_t size = accumulate(tensor.d+1, tensor.d+ tensor.nbDims, 1, multiplies<size_t>());
        cudaMalloc(&buffers[i], size * sizeof(float));


        bool input = engine->bindingIsInput(i);
        printf("Found tensor: %i, input: %i, name: %s\n", i, input, engine->getBindingName(i));
    }

    cv::Mat img = cv::imread(image_path);
    if (img.empty()){
        printf("Image load failed. Exiting...");
        return 1;
    }

    Dims input = engine->getBindingDimensions(inputIndex);
    Dims output = engine->getBindingDimensions(outputIndex);



    uploadToBuffer(img, (float*)buffers[0], input, true);

    // PART 3 - DO INFERENCE
    bool inferenceSuccessful = context->executeV2(buffers);
    printf("Done inference. Success: %i\n", inferenceSuccessful);
    printf("Output dimensions: %ix%ix%i, %i dimensions\n", output.d[0], output.d[1], output.d[2], output.nbDims);

    // PART 3 - CHECK OUTPUT


    postprocessAndDisplay(img, (float*)buffers[1], output, 0.5);

    cudaFree(buffers);

}

void uploadToBuffer(cv::Mat &img, float *gpu_input, const Dims dims, bool showImg){
    // Do scaling on GPU
    cv::cuda::GpuMat gpu_frame;
    gpu_frame.upload(img);

    int32_t input_width = dims.d[3];
    int32_t input_height = dims.d[2];
    int32_t input_channels = dims.d[1];
//    cv::Size_<int> input_size = cv::Size(input_width, input_height);

    printf("Loading and resizing image. Tensor dimensions = %ix%i, channels: %i\n", input_width, input_height, input_channels);
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, cv::Size(input_width, input_height),
                     0, 0, cv::INTER_NEAREST);

    cv::cuda::GpuMat flt_img;
    resized.convertTo(flt_img, CV_32FC3, 1.f/255.f);
    /* Normalize with std dev and offset from YOLO
     * Note OpenCV uses BGR format not RGB
     * yolov5/utils/augmentations.py
     * IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
     * IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation
     */
    cv::cuda::subtract(flt_img, cv::Scalar(0.406f, 0.456f, 0.485f), flt_img, cv::noArray(), -1);
    cv::cuda::divide(flt_img, cv::Scalar(0.225f, 0.224f, 0.229f), flt_img, 1, -1);

    // To tensor
    vector<cv::cuda::GpuMat> channels;
    for (int i = 0; i < input_channels; i++){
        cv::cuda::GpuMat split = cv::cuda::GpuMat(cv::Size(input_width, input_height), CV_32FC1, gpu_input + i* input_width * input_height);
        channels.emplace_back(split);
    }
    cv::cuda::split(flt_img, channels);

}

// https://github.com/enazoe/yolo-tensorrt/blob/master/modules/trt_utils.cpp
// https://github.com/enazoe/yolo-tensorrt/blob/master/modules/yolo.h
// PLS HJELP

void postprocessAndDisplay(cv::Mat &img, float *gpu_output, const Dims dims, float treshold){
    // Copy to CPU
    size_t dimsSize = accumulate(dims.d+1, dims.d+dims.nbDims, 1, multiplies<size_t>());
    vector<float> cpu_output (dimsSize);

    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size()*sizeof(float), cudaMemcpyDeviceToHost);

    vector<int> classIds, indices;
    vector<cv::Rect> boxes, boxesNMS;
    vector<float> confidences;

    int img_width = img.cols;
    int img_height = img.rows;

    int n_boxes = dims.d[1], n_classes = dims.d[2];

//    printf("Image size: %i x %i, n_boxes: %i, n_classes: %i\n", img_width, img_height, n_boxes, n_classes);

    for (int i = 0; i < n_boxes; i++){

        uint32_t maxClass = 0;
        float maxScore = -1000.0f;

        for (int j = 1; j < n_classes; j++){ // Starte paa 1 sia 0 er himmelen???
            float score = cpu_output[i * n_classes + j];

//            printf("Confidence found %f\n", score);

            if (score < treshold)continue;

            if (score > maxScore){
                maxScore = score;
                maxClass = j;
            }
        }

//        printf("Max score for %i, class %i: %f\n", i, maxClass , maxScore);
        if (maxScore > treshold){
            float left_raw = (cpu_output[4*i]);
            float top_raw = (cpu_output[4*i + 1]);
            float right_raw = (cpu_output[4*i + 2]);
            float bottom_raw = (cpu_output[4*i + 3]);
//            int width = right - left + 1;
//            int height = bottom - top + 1;
//
//            cv::rectangle(img, cv::Rect(left, top, width, height), cv::Scalar(255, 0, 0), 1);

//            printf("Drawing rectangle at: %f %f %f %f\n", left_raw, top_raw, right_raw, bottom_raw);

            //printf("Found class %i\n", maxClass);
        }
    }


    cv::resize(img, img, cv::Size(1000, 1000));
//    cv::imshow("Test", img);
//    cv::waitKey(0);
}