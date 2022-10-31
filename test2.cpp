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
vector< std::string > getClassNames(const std::string& imagenet_classes);

// Usage - ./TensorFlowTest {engine file}.engine {image file}.jpg
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " model.engine image.jpgn\n";
        return -1;
    }

    std::string model_path(argv[1]);
    std::string image_path(argv[2]);
    // Remember to load as binary
    std::ifstream opened_file_stream(model_path, std::ios::in | std::ios::binary);
//    opened_file_stream.open(model_path);

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

void postprocessAndDisplay(cv::Mat &img, float *gpu_output, const Dims dims, float threshold){
    // Copy to CPU
    size_t dimsSize = accumulate(dims.d+1, dims.d+dims.nbDims, 1, multiplies<size_t>());
    vector<float> cpu_output (dimsSize);

    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size()*sizeof(float), cudaMemcpyDeviceToHost);

    vector<int> classIds;
    vector<cv::Rect> boxes, boxesNMS;
    vector<float> confidences;

    /*
     * YOLOv5 Network outputs a 1x25200x85 tensor
     *
     * 25200 are all the box guesses (with probably shitty confidence for most of them)
     * 85 are the following:
     *  - x
     *  - y
     *  - width
     *  - height
     *  - scores for indiviudal classes (in test model - 80 for COCO)
     *
     * Thanks to https://stackoverflow.com/questions/74226690/tensorrt-finding-boundin-box-data-after-inference/74233108#74233108
     */

    int n_boxes = dims.d[1];
    int n_data = dims.d[2];

    int img_w = img.cols;
    int img_h = img.rows;

    vector<string> class_names = getClassNames("../classes.txt"); // TODO - Placeholder

    for (int i = 0; i < n_boxes; i++){
        int offset = i * n_data;

        const float w = cpu_output[offset + 2];
        const float h = cpu_output[offset + 3];
        // Go from x-mid to x, y-mid to y
        const float x = cpu_output[offset + 0] - w/2;
        const float y = cpu_output[offset + 1] - h/2;

        const float confidence = cpu_output[offset + 4];

        if (confidence < threshold)continue;

        int max_class = 1;
        float max_class_score = -1.0f;

        for (int j = 0; j < n_data-5; j++){
            float class_score = cpu_output[offset + 5 + j];

            if (class_score > max_class_score){
                max_class_score = class_score;
                max_class = j;
            }
        }

        float multiplier_norm_x = ((float)img_w)/640;
        float multiplier_norm_y = ((float)img_h)/640;
        float x_norm = x * multiplier_norm_x;
        float y_norm = y * multiplier_norm_y;
        float w_norm = w * multiplier_norm_x;
        float h_norm = h * multiplier_norm_y;

        printf("Found box of %s (%i) with conf=%f at %f %f %f %f\n", class_names[max_class].c_str() , max_class, confidence, x, y, w, h);



        cv::Rect box = cv::Rect((int)(x_norm), (int)(y_norm), (int)(w_norm), (int)(h_norm));


        boxes.emplace_back(box);
        confidences.emplace_back(confidence);
    }

    std::vector<int> filtered;
    cv::dnn::NMSBoxes(boxes, confidences, threshold, threshold, filtered);

    vector<float> confidencesNMS(filtered.size());
    for (int i = 0; i < filtered.size(); i++){
        confidencesNMS.emplace_back(confidences[filtered[i]]);
        boxesNMS.emplace_back(boxes[filtered[i]]);

        cv::rectangle(img, boxes[filtered[i]],cv::Scalar(0, 0, 255));
    }

//    cv::resize(img, img, cv::Size(1000, 1000));
    cv::imshow("Test", img);
    cv::waitKey(0);
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