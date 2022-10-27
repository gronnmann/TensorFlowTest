//
// Created by bartosz on 25.10.22.
//

#include <NvInfer.h>
#include <iostream>

#ifndef TENSORFLOWTEST_LOGGERFILE_H
#define TENSORFLOWTEST_LOGGERFILE_H

class TestLogger : public nvinfer1::ILogger{
    void log(Severity severity, const char* msg) noexcept override;
};


#endif //TENSORFLOWTEST_LOGGERFILE_H
