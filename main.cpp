
#include "FaceDetector.hpp"


int main(int argc, char const *argv[]) {
    FaceDetector* detector = new FaceDetector();
    // cv::VideoCapture cap;
    // cap.open(0);

    cv::VideoCapture cap("/home/truongnxd/Desktop/aiview_auto_recorder_20210125084412.mp4", cv::CAP_FFMPEG);
    // setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp", 1);
    // cv::VideoCapture cap("rtsp://admin:Bkav@2020@10.2.64.78:554/live", cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cout << "Error open camera" << std::endl;
    }
    
    // cv::Mat frame = cv::imread("../test_image.png");
    cv::Mat frame;
    while (1) {
        cap.read(frame);
        std::vector<cv::Rect> boxes;
        if (detector->detector(frame, boxes)) {
            if (boxes.size() > 0) {
                for (auto box : boxes) {
                    cv::rectangle(frame, box, cv::Scalar(0, 255, 255), 1);
                }  
            }
        }
        else {
            std::cout << "Error" << std::endl;
            return 1;
        }
        
        cv::namedWindow("stream", cv::WINDOW_NORMAL);
        cv::imshow("stream", frame);
        cv::waitKey(1);
    }
    
    return 0;
}
