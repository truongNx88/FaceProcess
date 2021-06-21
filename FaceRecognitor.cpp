//
//  FaceRecognitor.cpp
//
//  Created by Nguyen Xuan Truong on 18/06/2021.
//  Copyright © 2021 Nguyen Xuan Truong. All rights reserved.
//

#include "FaceRecognitor.hpp"

FaceRecognitor::FaceRecognitor() {
    this->m_height = this->m_width = 112;
    init();
}

FaceRecognitor::~FaceRecognitor() {
}

/** Read a model graph definition (xxx.pb) from disk, and creates a session object you can use to run it.
 */
Status FaceRecognitor::loadGraph(const string &graph_file_name) {
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

Status FaceRecognitor::readTensorFromMat(const cv::Mat &mat, tensorflow::Tensor& tensor) {
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

bool FaceRecognitor::init () {
    // Set dirs variables   
    std::string ROOTDIR = "../";
    std::string GRAPH = "model/insightface.pb";

    // Set input & output nodes names
    this->inputLayers = {"img_inputs:0", "dropout_rate:0"};
    this->outputLayers = {"E_BN2/Identity:0"};

    std::string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
    Status loadGraphStatus = loadGraph(graphPath);
    if (!loadGraphStatus.ok()) {
        LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
        return false;
    } 
    else {
        LOG(INFO) << "loadGraph(): frozen graph loaded" << std::endl;
    }

    this->dropoutTensors.push_back(Tensor(tensorflow::DT_FLOAT));
    this->dropoutTensors.push_back(Tensor(tensorflow::DT_FLOAT));

    return true;
}

void FaceRecognitor::align(cv::Mat& img, std::vector<cv::Point>& landmark, cv::Mat& ret) {
    float dst[10] = {38.2946, 73.5318, 56.0252, 41.5493, 70.7299,
                     51.6963, 51.5014, 71.7366, 92.3655, 92.2041};
    float src[10];
    for (int i = 0; i < 5; ++i) {
      src[i] = landmark[i].x;
      src[i+5] = landmark[i].y;
    }
    float M[6];
    getAffineMatrix(src, dst, M);
    cv::Mat m(2, 3, CV_32F);
    for (int i = 0; i < 6; ++i) {
      m.at<float>(i) = M[i];
    }
    cv::warpAffine(img, ret, m, cv::Size(m_width, m_height));
}

void FaceRecognitor::getAffineMatrix(float* src_5pts, const float* dst_5pts, float* M) {
    float src[10], dst[10];
    memcpy(src, src_5pts, sizeof(float)*10);
    memcpy(dst, dst_5pts, sizeof(float)*10);

    float ptmp[2];
    ptmp[0] = ptmp[1] = 0;
    for (int i = 0; i < 5; ++i) {
        ptmp[0] += src[i];
        ptmp[1] += src[5+i];
    }
    ptmp[0] /= 5;
    ptmp[1] /= 5;
    for (int i = 0; i < 5; ++i) {
        src[i] -= ptmp[0];
        src[5+i] -= ptmp[1];
        dst[i] -= ptmp[0];
        dst[5+i] -= ptmp[1];
    }

    float dst_x = (dst[3]+dst[4]-dst[0]-dst[1])/2, dst_y = (dst[8]+dst[9]-dst[5]-dst[6])/2;
    float src_x = (src[3]+src[4]-src[0]-src[1])/2, src_y = (src[8]+src[9]-src[5]-src[6])/2;
    float theta = atan2(dst_x, dst_y) - atan2(src_x, src_y);

    float scale = sqrt(pow(dst_x, 2) + pow(dst_y, 2)) / sqrt(pow(src_x, 2) + pow(src_y, 2));
    float pts1[10];
    float pts0[2];
    float _a = sin(theta), _b = cos(theta);
    pts0[0] = pts0[1] = 0;
    for (int i = 0; i < 5; ++i) {
        pts1[i] = scale*(src[i]*_b + src[i+5]*_a);
        pts1[i+5] = scale*(-src[i]*_a + src[i+5]*_b);
        pts0[0] += (dst[i] - pts1[i]);
        pts0[1] += (dst[i+5] - pts1[i+5]);
    }
    pts0[0] /= 5;
    pts0[1] /= 5;

    float sqloss = 0;
    for (int i = 0; i < 5; ++i) {
        sqloss += ((pts0[0]+pts1[i]-dst[i])*(pts0[0]+pts1[i]-dst[i])
                + (pts0[1]+pts1[i+5]-dst[i+5])*(pts0[1]+pts1[i+5]-dst[i+5]));
    }

    float square_sum = 0;
    for (int i = 0; i < 10; ++i) {
        square_sum += src[i]*src[i];
    }
    for (int t = 0; t < 200; ++t) {
        _a = 0;
        _b = 0;
        for (int i = 0; i < 5; ++i) {
            _a += ((pts0[0]-dst[i])*src[i+5] - (pts0[1]-dst[i+5])*src[i]);
            _b += ((pts0[0]-dst[i])*src[i] + (pts0[1]-dst[i+5])*src[i+5]);
        }
        if (_b < 0) {
            _b = -_b;
            _a = -_a;
        }
        float _s = sqrt(_a*_a + _b*_b);
        _b /= _s;
        _a /= _s;

        for (int i = 0; i < 5; ++i) {
            pts1[i] = scale*(src[i]*_b + src[i+5]*_a);
            pts1[i+5] = scale*(-src[i]*_a + src[i+5]*_b);
        }

        float _scale = 0;
        for (int i = 0; i < 5; ++i) {
            _scale += ((dst[i]-pts0[0])*pts1[i] + (dst[i+5]-pts0[1])*pts1[i+5]);
        }
        _scale /= (square_sum*scale);
        for (int i = 0; i < 10; ++i) {
            pts1[i] *= (_scale / scale);
        }
        scale = _scale;

        pts0[0] = pts0[1] = 0;
        for (int i = 0; i < 5; ++i) {
            pts0[0] += (dst[i] - pts1[i]);
            pts0[1] += (dst[i+5] - pts1[i+5]);
        }
        pts0[0] /= 5;
        pts0[1] /= 5;

        float _sqloss = 0;
        for (int i = 0; i < 5; ++i) {
            _sqloss += ((pts0[0]+pts1[i]-dst[i])*(pts0[0]+pts1[i]-dst[i])
                    + (pts0[1]+pts1[i+5]-dst[i+5])*(pts0[1]+pts1[i+5]-dst[i+5]));
        }
        if (abs(_sqloss - sqloss) < 1e-2) {
            break;
        }
        sqloss = _sqloss;
    }

    for (int i = 0; i < 5; ++i) {
        pts1[i] += (pts0[0] + ptmp[0]);
        pts1[i+5] += (pts0[1] + ptmp[1]);
    }

    M[0] = _b*scale;
    M[1] = _a*scale;
    M[3] = -_a*scale;
    M[4] = _b*scale;
    M[2] = pts0[0] + ptmp[0] - scale*(ptmp[0]*_b + ptmp[1]*_a);
    M[5] = pts0[1] + ptmp[1] - scale*(-ptmp[0]*_a + ptmp[1]*_b);
}

bool FaceRecognitor::recognize(cv::Mat frame, std::vector<cv::Point> landmarks, std::vector<float>& embeddings) {
    this->shapeImg = tensorflow::TensorShape();
    this->shapeImg.AddDim(1);
    this->shapeImg.AddDim(112);
    this->shapeImg.AddDim(112);
    this->shapeImg.AddDim(3);

    // tensorflow::TensorShape dropOutShape = tensorflow::TensorShape(Tensor());
    // dropOutShape.AddDim();
    // dropOutShape.AddDim(Tensor());
    this->imgTensor = Tensor(tensorflow::DT_FLOAT, this->shapeImg);
    cv::Mat aligned;
    align(frame, landmarks, aligned);
    Status readTensorStatus = readTensorFromMat(aligned, this->imgTensor);
    if (!readTensorStatus.ok()) {
        LOG(ERROR) << "Mat->Tensor conversion failed: " << readTensorStatus;
        return false;
    }
    std::vector<std::pair<std::string, Tensor>> inputs;
    const Tensor& dropoutTensor = this->dropoutTensors[0] ;
    inputs.push_back(std::make_pair(this->inputLayers[0], this->imgTensor));
    inputs.push_back(std::make_pair(this->inputLayers[1], dropoutTensor));
    // for (size_t i = 0; i < this->inputTensors.size(); i++) {
    //     inputs.push_back(std::make_pair(this->inputLayers[i], this->inputTensors[i]));
    // }


    std::vector<Tensor> outputs;
    Status runStatus = this->session->Run(inputs, this->outputLayers, {}, &outputs);
    if (!runStatus.ok()) {
        LOG(ERROR) << "Running model failed: " << runStatus;
        return false;
    }
    return true;
}