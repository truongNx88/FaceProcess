//
//  FaceDetector.cpp
//
//  Created by Nguyen Xuan Truong on 13/06/2021.
//  Copyright © 2021 Nguyen Xuan Truong. All rights reserved.
//
#include "FaceDetector.hpp"

FaceDetector::FaceDetector() {
    this->thresholds = {0.6, 0.7, 0.7};
    init();
}

FaceDetector::~FaceDetector() {
}

bool FaceDetector::init() {
    // Set dirs variables   
    std::string ROOTDIR = "../";
    std::string GRAPH = "model/mtcnn.pb";

    // Set input & output nodes names
    this->inputLayer = {"input:0", "min_size:0", "thresholds:0", "factor:0"};
    this->outputLayer = {"prob:0", "landmarks:0", "box:0"};

    std::string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
    Status loadGraphStatus = loadGraph(graphPath);
    if (!loadGraphStatus.ok()) {
        LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
        return false;
    } 
    else {
        LOG(INFO) << "loadGraph(): frozen graph loaded" << std::endl;
    }

    //setting frame shape
    this->inputs.push_back(Tensor(tensorflow::DT_FLOAT, this->shapeInput));
    
    this->inputs.push_back(Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({})));
    
    this->inputs.push_back(Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({3})));
    
    this->inputs.push_back(Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({})));

    this->inputs[1].flat<float>()(0) = 40.00;

    float *p = this->inputs[2].flat<float>().data();
    std::copy( p, p + 1, this->thresholds.begin() );

    this->inputs[3].flat<float>()(0) = 0.709;

    return true;
}

/** Read a model graph definition (xxx.pb) from disk, and creates a session object you can use to run it.
 */
Status FaceDetector::loadGraph(const string &graph_file_name) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    this->session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = this->session->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}

Status FaceDetector::readTensorFromMat(const cv::Mat &mat, tensorflow::Tensor& tensor) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;

    // Trick from https://github.com/tensorflow/tensorflow/issues/8033
    float *p = tensor.flat<float>().data();
    cv::Mat fakeMat(mat.rows, mat.cols, CV_32FC3, p);
    mat.convertTo(fakeMat, CV_32FC3);

    auto input_tensor = Placeholder(root.WithOpName("input"), tensorflow::DT_FLOAT);
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {{"input", tensor}};
    auto uint8Caster = Cast(root.WithOpName("uint8_Cast"), tensor, tensorflow::DT_FLOAT);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output outTensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::vector<Tensor> outTensors;
    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"uint8_Cast"}, {}, &outTensors));

    tensor = outTensors.at(0);
    return Status::OK();
}

bool FaceDetector::detector(cv::Mat frame, std::vector<cv::Rect>& boxes, std::vector<std::vector<cv::Point>>& landmarks) {
    this->shapeInput = tensorflow::TensorShape();
    this->shapeInput.AddDim(frame.rows);
    this->shapeInput.AddDim(frame.cols);
    this->shapeInput.AddDim(3);
    this->inputs[0] = Tensor(tensorflow::DT_FLOAT, this->shapeInput);
    Status readTensorStatus = readTensorFromMat(frame, this->inputs[0]);
    if (!readTensorStatus.ok()) {
        LOG(ERROR) << "Mat->Tensor conversion failed: " << readTensorStatus;
        return false;
    }

    std::vector<std::pair<std::string, Tensor>> tensorInputs;

    for (size_t i = 0; i < this->inputs.size(); i++) {
        tensorInputs.push_back(std::make_pair(this->inputLayer[i], this->inputs[i]));
    }
    
    std::vector<Tensor> outputs;
    Status runStatus = this->session->Run(tensorInputs, this->outputLayer, {}, &outputs);
    if (!runStatus.ok()) {
        LOG(ERROR) << "Running model failed: " << runStatus;
        return false;
    }
    double thresholdIOU = 0.9999;
    tensorflow::TTypes<float>::Flat probs = outputs[0].flat<float>();
    tensorflow::TTypes<float>::Flat landmarksTs = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat boxesTs = outputs[2].flat<float>();
    std::vector<cv::Point> landmark;
    for (size_t i = 0; i < probs.size(); i++) {
        if (probs(i) > thresholdIOU) {
            int x_tl = (int) (boxesTs(4*i + 1) );
            int y_tl = (int) (boxesTs(4*i));
            int x_br = (int) (boxesTs(4*i + 3));
            int y_br = (int) (boxesTs(4*i + 2));
            cv::Rect box = cv::Rect(cv::Point(x_tl, y_tl), cv::Point(x_br, y_br));
            landmark.push_back(cv::Point( (int) landmarksTs(4*(i+1)), (int) landmarksTs(4*i)));
            landmark.push_back(cv::Point( (int) landmarksTs(4*(i+1) + 1), (int) landmarksTs(4*i + 1)));
            landmark.push_back(cv::Point( (int) landmarksTs(4*(i+1) + 2) , (int) landmarksTs(4*i + 2)));
            landmark.push_back(cv::Point( (int) landmarksTs(4*(i+1) + 3) , (int) landmarksTs(4*i + 3)));

            // std::cout << probs(i) << std::endl;
            boxes.push_back(box);
            landmarks.push_back(landmark);
        }        
    }
    return true;
}
