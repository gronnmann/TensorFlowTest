#include <iostream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <numeric>
#include "tensorrt_utils.h"
#include <chrono>

using namespace std;
using namespace nvinfer1;

static TestLogger logger;

// Good example code - https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/TensorRT-introduction/simpleOnnx.cpp


void uploadToBuffer(cv::Mat &img, float *gpu_input, const Dims dims);
void uploadToBuffer(cv::Mat &img, float *gpu_input, int32_t input_width, int32_t input_height, int32_t input_channels);

void postprocessAndDisplay(cv::Mat &img, float *gpu_output, const Dims dims, vector<string> class_names, float threshold, float nms_threshold);
bool loadEngine(ICudaEngine** engine, string& model_name);
bool loadONNX(ICudaEngine** engine, string& onnx_path, string saved_name);


vector< std::string > getClassNames(const std::string& imagenet_classes);

cv::Scalar diffColors[] = {
        cv::Scalar(255, 0, 0),
        cv::Scalar(255, 255, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 255, 255),
        cv::Scalar(255, 0, 255),
};


// Usage - ./TensorFlowTest {engine file}.engine {image file}.jpg
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " model.engine/onnx image.jpg useOnnx\n";
        return -1;
    }

    std::string model_path(argv[1]);
    std::string image_path(argv[2]);
    std::string useOnnx(argv[3]);

//    // Remember to load as binary
//    std::ifstream opened_file_stream(model_path, std::ios::in | std::ios::binary);
//    opened_file_stream.open(model_path);
//
//    std::printf("Loading network from %s\n", model_path.c_str());
//
//
//    // Check netowrk size
//    opened_file_stream.seekg(0, std::ios::end);
//    const int model_size = opened_file_stream.tellg();
//    opened_file_stream.seekg(0, std::ios::beg);
//
//    std::printf("Model size is %d\n", model_size);
//
//    void* model_mem = malloc(model_size);
//    if (!model_mem){
//        std::printf("Failed to allocate %i bytes to deserialize model\n", model_size);
//    }
//    opened_file_stream.read((char*)model_mem, model_size);


    //SECTION 1 - Creating CUDA Engine and loading .engine file | Works

    ICudaEngine* engine = nullptr;

//    string saved_name = "parsed_yolov5l.engine";
//    bool loadModel = loadONNX(engine, model_path, saved_name);

    bool loadModel = false;
    if (useOnnx == "true"){
        cout << "Parsing onnx model\n";
        loadModel = loadONNX(&engine, model_path, "auto.engine");
    }else{
        loadModel = loadEngine(&engine, model_path);
    }

    if (!loadModel){
        cout << "Engine creation failed\n";
        return 1;
    }
    printf("Engine takes in %i tensors.\n", engine->getNbBindings());

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

    vector<string> class_names = getClassNames("./classes.txt"); // TODO - Placeholder

    int32_t input_width = input.d[3];
    int32_t input_height = input.d[2];
    int32_t input_channels = input.d[1];


    auto point1 = chrono::high_resolution_clock::now();
    uploadToBuffer(img, (float*)buffers[0], input_width, input_height, input_channels);

    auto point2 = chrono::high_resolution_clock::now();

    // PART 3 - DO INFERENCE
    bool inferenceSuccessful = context->executeV2(buffers);
    auto point3 = chrono::high_resolution_clock::now();

    // PART 3 - CHECK OUTPUT

    postprocessAndDisplay(img, (float*)buffers[1], output, class_names, 0.25, 0.45);
    auto point4 = chrono::high_resolution_clock::now();

    printf("Done inference. Success: %i\n", inferenceSuccessful);
    printf("Output dimensions: %ix%ix%i, %i dimensions\n", output.d[0], output.d[1], output.d[2], output.nbDims);


    chrono::milliseconds durationPreprocess = chrono::duration_cast<chrono::milliseconds>(point2 - point1);
    chrono::milliseconds durationInference = chrono::duration_cast<chrono::milliseconds>(point3 - point2);
    chrono::milliseconds durationPostprocess = chrono::duration_cast<chrono::milliseconds>(point4 - point3);
    chrono::milliseconds durationTotal = chrono::duration_cast<chrono::milliseconds>(point4 - point1);
    cout << "Timings:\n preprocessing " << durationPreprocess.count() << " ms\n inference " << durationInference.count() << " ms\n postprocess " << durationPostprocess.count() << " ms \n"
        << "total " << durationTotal.count() << " ms\n";

    cudaFree(buffers);

}

bool loadEngine(ICudaEngine** engine, string& model_path){
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

    IRuntime* runtime = createInferRuntime(logger);


    //// Crashes here for some reason
    *engine = runtime->deserializeCudaEngine(model_mem, model_size);

    if (!*engine){
        cout << "Engine creation failed\n";
        return false;
    }


    delete runtime;
    free(model_mem);

    return true;
}
bool loadONNX(ICudaEngine** engine, string& onnx_path, string saved_name){
    IBuilder* builder = createInferBuilder(logger);

    //idk what it does but is in nvidia
    uint32_t flag = 1U <<static_cast<uint32_t>
    (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    INetworkDefinition* network = builder->createNetworkV2(flag);

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

    bool success = parser->parseFromFile(onnx_path.c_str(),
                                         static_cast<int32_t>(ILogger::Severity::kINFO));

    if (!success){
        cout << "Failed to parse onnx from file " << onnx_path << "\n";
        return false;
    }

    IBuilderConfig* config = builder->createBuilderConfig();
    // 4 GB
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, (4 * 1U) << 20);

//    IOptimizationProfile* profile = builder->createOptimizationProfile();
//    //Model 1x 3 channels x size
//    profile->setDimensions("input", OptProfileSelector::kMAX, Dims4(1, 3, 640, 640));
//    profile->setDimensions("input", OptProfileSelector::kMIN, Dims4(1, 3, 640, 640));
//    profile->setDimensions("input", OptProfileSelector::kOPT, Dims4(1, 3, 640, 640));
//    config->addOptimizationProfile(profile);

    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);

    ofstream write_out(saved_name, ios::binary);
    write_out.write((const char*) serializedModel->data(), serializedModel->size());
    write_out.close();


    IRuntime* runtime = createInferRuntime(logger);
    *engine = runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());
    if (!*engine){
        cout << "Engine file creation failed\n";
        return false;
    }

    delete parser;
    delete network;
    delete config;
    delete builder;
    delete runtime;

    return true;
}

void uploadToBuffer(cv::Mat &img, float *gpu_input, const Dims dims){
    int32_t input_width = dims.d[3];
    int32_t input_height = dims.d[2];
    int32_t input_channels = dims.d[1];

    uploadToBuffer(img, gpu_input, input_width, input_height, input_channels);
}

void uploadToBuffer(cv::Mat &img, float *gpu_input, int32_t input_width, int32_t input_height, int32_t input_channels){
    // Do scaling on GPU
    cv::cuda::GpuMat gpu_frame;

    gpu_frame.upload(img);

//    cv::Size_<int> input_size = cv::Size(input_width, input_height);

    printf("Loading and resizing image. Tensor dimensions = %ix%i, channels: %i\n", input_width, input_height, input_channels);
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, cv::Size(input_width, input_height),
                     0, 0, cv::INTER_LINEAR);
    // BGR to RGB
    cv::cuda::cvtColor(resized, resized, cv::COLOR_BGR2RGB);



    cv::cuda::GpuMat flt_img;
    resized.convertTo(flt_img, CV_32FC3, 1.f/255.f);
    /* Normalize with std dev and offset from YOLO
     * Note OpenCV uses BGR format not RGB
     * yolov5/utils/augmentations.py
     * IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
     * IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation
     */
//    cv::cuda::subtract(flt_img, cv::Scalar(0.406f, 0.456f, 0.485f), flt_img, cv::noArray(), -1);
//    cv::cuda::divide(flt_img, cv::Scalar(0.225f, 0.224f, 0.229f), flt_img, 1, -1);

    // To tensor
    vector<cv::cuda::GpuMat> channels;
    for (int i = 0; i < input_channels; i++){
        cv::cuda::GpuMat split = cv::cuda::GpuMat(cv::Size(input_width, input_height), CV_32FC1, gpu_input + i * input_width * input_height);
        channels.emplace_back(split);
    }
    cv::cuda::split(flt_img, channels);

}

// https://github.com/enazoe/yolo-tensorrt/blob/master/modules/trt_utils.cpp
// https://github.com/enazoe/yolo-tensorrt/blob/master/modules/yolo.h
// PLS HJELP

void postprocessAndDisplay(cv::Mat &img, float *gpu_output, const Dims dims, vector<string> class_names, float threshold, float nms_threshold){
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

        cv::rectangle(img, box ,cv::Scalar(255, 255, 255), 1);
//        cv::Point text_point = cv::Point(box.x + 15,box.y + 50);
//        cv::putText(img, "car (" + to_string(confidence) + ")", text_point,
//                    cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 00255));

        boxes.emplace_back(box);
        confidences.emplace_back(confidence);
        classIds.emplace_back(max_class);
    }

    std::vector<int> filtered;
    cv::dnn::NMSBoxes(boxes, confidences, threshold, nms_threshold, filtered);
    vector<float> confidencesNMS(filtered.size());
    for (int i = 0; i < filtered.size(); i++){
        confidencesNMS.emplace_back(confidences[filtered[i]]);

        cv::Rect box = boxes[filtered[i]];
        boxesNMS.emplace_back(box);

        cv::Scalar color_to_use = diffColors[ classIds[filtered[i]] % (sizeof(diffColors)/sizeof(diffColors[0]))];

        cv::rectangle(img, box,color_to_use, 10);
        cv::Point text_point = cv::Point(box.x,box.y - 20);
        cv::putText(img, class_names[classIds[filtered[i]]] + " (" + to_string(confidences[filtered[i]]) + ")", text_point,
                    cv::FONT_HERSHEY_TRIPLEX, 1.5, color_to_use);
    }
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::imwrite("inferenced.jpg", img);

//    cv::resize(img, img, cv::Size(1000, 1000));
//    cv::imshow("Test", img);
//    cv::waitKey(0);

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
