//
//  FaceRecognitor.hpp
//
//  Created by Nguyen Xuan Truong on 18/06/2021.
//  Copyright © 2021 Nguyen Xuan Truong. All rights reserved.
//

#ifndef FaceRecognitor_hpp
#define FaceRecognitor_hpp

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


#include <vector>
#include <iostream>
#include <math.h>
#include <regex>
#include <opencv2/opencv.hpp>

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

class FaceRecognitor{
private:
    std::unique_ptr<tensorflow::Session> session;
    tensorflow::TensorShape shapeImg;
    Tensor imgTensor;
    std::vector<Tensor> dropoutTensors;
    std::vector<std::string> inputLayers;
    // std::string inputLayers;
    std::vector<std::string> outputLayers;
    int m_width;
    int m_height;
    
    Status loadGraph(const string &graph_file_name);
    Status readTensorFromMat(const cv::Mat &mat, tensorflow::Tensor& tensor);
    void align(cv::Mat& img, std::vector<cv::Point>& landmark, cv::Mat& ret);
    void getAffineMatrix(float* src_5pts, const float* dst_5pts, float* M);
    bool init();
public:
    FaceRecognitor();
    ~FaceRecognitor();

    virtual bool recognize(cv::Mat frame, std::vector<cv::Point> landmarks, std::vector<float>& embeddings);
};

#endif