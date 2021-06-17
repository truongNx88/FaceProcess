//
//  FaceDetector.hpp
//
//  Created by Nguyen Xuan Truong on 13/06/2021.
//  Copyright © 2021 Nguyen Xuan Truong. All rights reserved.
//
#ifndef FaceDetector_hpp
#define FaceDetector_hpp

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include <math.h>
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <regex>
#include <opencv2/opencv.hpp>

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

class FaceDetector {
private:
    bool init();
    std::unique_ptr<tensorflow::Session> session;
    tensorflow::TensorShape shapeInput;

    std::vector<Tensor> inputs;
    Status loadGraph(const string &graph_file_name);
    Status readTensorFromMat(const cv::Mat &mat, tensorflow::Tensor& tensor);

    std::vector<std::string> inputLayer;
    std::vector<std::string> outputLayer;
    std::vector<float> thresholds;
public:
    FaceDetector();
    ~FaceDetector();
    virtual bool detector(cv::Mat frame, std::vector<cv::Rect>& boxes, std::vector<std::vector<cv::Point>>& landmarks);
};

#endif