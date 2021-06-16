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
    this->shapeThresholds = tensorflow::TensorShape();
    this->shapeThresholds.AddDim(3);

    this->inputs.push_back(Tensor(tensorflow::DT_FLOAT, this->shapeInput));
    
    this->inputs.push_back(Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({})));
    
    this->inputs.push_back(Tensor(tensorflow::DT_FLOAT, this->shapeThresholds));
    
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

std::vector<cv::Rect> FaceDetector::NMS(std::vector<std::vector<int>> box, double threshold) {
    size_t count = box.size();
    std::vector<std::pair<size_t, float>> order(count);
    for (size_t i = 0; i < count; ++i) {
        order[i].first = i;
        order[i].second = box[i][4];
    }

    sort(order.begin(), order.end(), [](const std::pair<int, float> &ls, const std::pair<int, float> &rs) {
        return ls.second > rs.second;
    });

    std::vector<int> keep;
    std::vector<bool> exist_box(count, true);
    for (size_t _i = 0; _i < count; ++_i) {
        size_t i = order[_i].first;
        float x1, y1, x2, y2, w, h, iarea, jarea, inter, ovr;
        if (!exist_box[i]) continue;
        keep.push_back(i);
        for (size_t _j = _i + 1; _j < count; ++_j) {
            size_t j = order[_j].first;
            if (!exist_box[j]) continue;
            x1 = std::max(box[i][0], box[j][0]);
            y1 = std::max(box[i][1], box[j][1]);
            x2 = std::min(box[i][2], box[j][2]);
            y2 = std::min(box[i][3], box[j][3]);
            w = std::max(float(0.0), x2 - x1 + 1);
            h = std::max(float(0.0), y2 - y1 + 1);
            iarea = (box[i][2] - box[i][0] + 1) * (box[i][3] - box[i][1] + 1);
            jarea = (box[j][2] - box[j][0] + 1) * (box[j][3] - box[j][1] + 1);
            inter = w * h;
            ovr = inter / (iarea + jarea - inter);
            if (ovr >= threshold) exist_box[j] = false;
        }
    }

    std::vector<cv::Rect> result;
    result.reserve(keep.size());
    for (size_t i = 0; i < keep.size(); ++i) {
        int x1 = box[keep[i]].at(0);
        int y1 = box[keep[i]].at(1);
        int x2 = box[keep[i]].at(2);
        int y2 = box[keep[i]].at(3);
        cv::Rect NMSbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
        result.push_back(NMSbox);
    }

    return result;
}

bool FaceDetector::detector(cv::Mat frame, std::vector<cv::Rect>& boxes) {
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
    int thresholdIOU = 0.709;
    tensorflow::TTypes<float>::Flat probs = outputs[0].flat<float>();
    tensorflow::TTypes<float>::Flat landmarks = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat boxesTs = outputs[2].flat<float>();
    std::vector<std::vector<int>> totalBox;
    for (size_t i = 0; i < probs.size(); i++) {
        if (probs(i) > thresholdIOU) {
            std::vector<int> boxNMS;
            boxNMS.push_back((int) (boxesTs(4*i + 1))); 
            boxNMS.push_back((int) (boxesTs(4*i)));
            boxNMS.push_back((int) (boxesTs(4*i + 3)));
            boxNMS.push_back((int) (boxesTs(4*i + 2)));
            totalBox.push_back(boxNMS);
        }
    }
    
    boxes = NMS(totalBox, 0.2);

    // std::vector<int> goodIdxs = NMS(probs, boxesTs, thresholdIOU);
    // for (size_t i = 0; i < goodIdxs.size(); i++) {
    //     int x_tl = (int) (boxesTs(4*goodIdxs.at(i) + 1) * frame.cols);
    //     int y_tl = (int) (boxesTs(4*goodIdxs.at(i)) * frame.rows);

    //     int x_br = (int) (boxesTs(4*goodIdxs.at(i) + 3) * frame.cols);
    //     int y_br = (int) (boxesTs(4*goodIdxs.at(i) + 2) * frame.rows);

    //     cv::Rect box = cv::Rect(cv::Point(x_tl, y_tl), cv::Point(x_br, y_br));
    //     boxes.push_back(box);   
    // }
    return true;
}
