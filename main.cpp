
#include "FaceDetector.hpp"
#include "FaceRecognitor.hpp"


int main(int argc, char const *argv[]) {
    FaceDetector* detector = new FaceDetector();
    FaceRecognitor* recognitor = new FaceRecognitor();
    // cv::VideoCapture cap;
    // cap.open(0);

    // cv::VideoCapture cap("/home/truongnxd/Desktop/aiview_auto_recorder_20210125084412.mp4", cv::CAP_FFMPEG);
    // setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp", 1);
    // cv::VideoCapture cap("rtsp://admin:Bkav@2020@10.2.64.78:554/live", cv::CAP_FFMPEG);
    // if (!cap.isOpened()) {
    //     std::cout << "Error open camera" << std::endl;
    // }
    

    cv::Mat frame = cv::imread("../test.jpg");
    // cv::Mat frame;
    // while (1) {
        // cap.read(frame);
        std::vector<cv::Rect> boxes;
        std::vector<std::vector<cv::Point>> landmarks;
        if (detector->detector(frame, boxes, landmarks)) {
            if (boxes.size() > 0) {
                std::vector<cv::Mat> faces;
                for (int i = 0; i < boxes.size(); i++) {
                    cv::rectangle(frame, boxes[i], cv::Scalar(0, 255, 255), 3);
                    for (auto landmark : landmarks[i]) {
                        cv::circle( frame, landmark, 0.1, cv::Scalar( 255, 0, 0 ), 3 );
                    }
                    std::vector<float> embeddings;
                    cv::Mat face = frame(boxes[i]);
                    recognitor->recognize(face, landmarks[i], embeddings);
                }  
            }
        }
        else {
            std::cout << "Error" << std::endl;
            return 1;
        }
        
        cv::namedWindow("stream", cv::WINDOW_NORMAL);
        cv::imshow("stream", frame);
        cv::waitKey(0);
    // }
    
    return 0;
}
