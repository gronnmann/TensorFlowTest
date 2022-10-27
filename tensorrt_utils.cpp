//
// Created by bartosz on 25.10.22.
//

#include "tensorrt_utils.h"


void TestLogger::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept {
    std::cout << msg << "\n";
}
