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

std::vector<cv::Rect> FaceDetector::NMS(std::vector<cv::Rect> box, std::vector<float> scores, double threshold) {
    size_t count = box.size();
    std::vector<std::pair<size_t, float>> order(count);
    for (size_t i = 0; i < count; ++i) {
        order[i].first = i;
        order[i].second = scores[i];
        // std::cout << "Score: " <<scores[i] << std::endl;
    }

    sort(order.begin(), order.end(), [](const std::pair<int, float> &ls, const std::pair<int, float> &rs) {
        std::cout << ls.second << ", " << rs.second << " : " << (ls.second > rs.second)  << std::endl;
        return ls.second > rs.second;
    });

    std::vector<int> keep;
    std::vector<bool> exist_box(count, true);
    for (size_t _i = 0; _i < count; ++_i) {
        size_t i = order[_i].first;
        // std::cout << "i: " << i << std::endl;

        float x1, y1, x2, y2, w, h, iarea, jarea, inter, ovr;
        if (!exist_box[i]) {
            continue;
        }        
        keep.push_back(i);
        for (size_t _j = _i + 1; _j < count; ++_j) {
            size_t j = order[_j].first;
            if (!exist_box[j]) {
                continue;
            }
            x1 = std::max(box[i].tl().x, box[j].tl().x);
            y1 = std::max(box[i].tl().y, box[j].tl().y);
            x2 = std::min(box[i].br().x, box[j].tl().x);
            y2 = std::min(box[i].br().y, box[j].br().y);
            w = std::max(float(0.0), x2 - x1 + 1);
            h = std::max(float(0.0), y2 - y1 + 1);
            iarea = (box[i].br().x - box[i].tl().x + 1) * (box[i].br().y - box[i].tl().y + 1);
            jarea = (box[j].br().x - box[j].tl().x + 1) * (box[j].br().y - box[j].tl().y + 1);
            // iarea = box[i].area();
            // jarea = box[j].area();
            inter = w * h;
            // ovr = inter / (iarea + jarea - inter);
            ovr = inter / (iarea < jarea ? iarea : jarea);
            // std::cout << "ovr: " << ovr << std::endl;
            if (ovr >= threshold) {
                exist_box[j] = false;
            }
        }
    }

    std::vector<cv::Rect> result;
    result.reserve(keep.size());
    for (size_t i = 0; i < keep.size(); ++i) {
        result.push_back(box[keep[i]]);
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
    double thresholdIOU = 0.9999;
    tensorflow::TTypes<float>::Flat probs = outputs[0].flat<float>();
    tensorflow::TTypes<float>::Flat landmarks = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat boxesTs = outputs[2].flat<float>();

    std::vector<cv::Rect> nmsBoxes;
    std::vector<float> scores;
    for (size_t i = 0; i < probs.size(); i++) {
        if (probs(i) > thresholdIOU) {
            int x_tl = (int) (boxesTs(4*i + 1) );
            int y_tl = (int) (boxesTs(4*i));
            int x_br = (int) (boxesTs(4*i + 3));
            int y_br = (int) (boxesTs(4*i + 2));
            cv::Rect box = cv::Rect(cv::Point(x_tl, y_tl), cv::Point(x_br, y_br));
            scores.push_back(probs(i));
            // std::cout << probs(i) << std::endl;
            boxes.push_back(box);
        }        
    }
    // boxes = NMS(nmsBoxes, scores,0.1);
    return true;
}
