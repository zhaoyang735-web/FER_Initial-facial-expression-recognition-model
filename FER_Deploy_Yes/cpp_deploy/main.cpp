#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <opencv2/objdetect.hpp> // 引入人脸检测模块

// 简单的日志记录器
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

const std::string ENGINE_FILE = "mobilefacenet_fp16.engine"; // 直接使用我们生成的引擎
const int INPUT_H = 112;
const int INPUT_W = 112;
const int NUM_CLASSES = 7;
const std::vector<std::string> EMOTIONS = {"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"};

int main() {
    // 1. 读取已经生成的 Engine 文件
    std::ifstream file(ENGINE_FILE, std::ios::binary);
    if (!file.good()) {
        std::cerr << "找不到引擎文件: " << ENGINE_FILE << " (请确认文件是否在当前目录下!)" << std::endl;
        return -1;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    // 2. 初始化 TensorRT 10 运行时
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    // TRT 10 接口：只传数据和大小，不再需要 nullptr 插件工厂
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size); 
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // 3. 分配显存
    void* d_input;
    void* d_output;
    cudaMalloc(&d_input, 1 * 3 * INPUT_H * INPUT_W * sizeof(float));
    cudaMalloc(&d_output, 1 * NUM_CLASSES * sizeof(float));

    // TRT 10 接口：使用 setTensorAddress 绑定输入输出
    context->setTensorAddress("input", d_input);
    context->setTensorAddress("output", d_output);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

// 4. 打开摄像头 (绕过 GStreamer，强制使用 V4L2 底层驱动)
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头！请检查 USB 摄像头是否插入。" << std::endl;
        return -1;
    }
    
    // 【秘密武器】设置 MJPG 视频流格式，这能让 USB 摄像头极速传输图像！
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    // 设置为非常舒适的 640x480 分辨率
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame, img;
    std::vector<float> inputData(3 * INPUT_H * INPUT_W);
    std::vector<float> outputData(NUM_CLASSES);

    std::cout << "引擎加载成功，开始实时推理..." << std::endl;

// --- 新增：加载 OpenCV 自带的人脸检测器 ---
    cv::CascadeClassifier face_cascade;
    // Jetson/Ubuntu OpenCV 的默认路径，如果报错找不到文件，请告诉我
    if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        std::cerr << "警告：找不到人脸检测模型文件！" << std::endl;
    }

    std::cout << "引擎加载成功，开始实时推理..." << std::endl;

    // 5. 推理循环
    while (true) {
        auto start = std::chrono::high_resolution_clock::now();
        
        cap >> frame;
        if (frame.empty()) break;

        // --- 新增：人脸检测步骤 ---
        std::vector<cv::Rect> faces;
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        // 执行检测
        face_cascade.detectMultiScale(gray, faces, 1.1, 4);

        int maxIdx = -1; // -1 表示没检测到脸
        float fps = 0.0f;

        // 如果画面中检测到了至少一张人脸
        if (faces.size() > 0) {
            // 我们暂时只处理画面中的第一张脸
            cv::Rect faceBox = faces[0];
            
            // 确保框没有超出图像边界
            faceBox &= cv::Rect(0, 0, frame.cols, frame.rows); 
            
            // 【核心修正】：从大图中把人脸抠出来！
            cv::Mat faceImg = frame(faceBox); 

            // 在原图上画个人脸框，方便你看
            cv::rectangle(frame, faceBox, cv::Scalar(255, 0, 0), 2);

            // 预处理（现在只对抠出来的脸进行 Resize ！！！）
            cv::resize(faceImg, img, cv::Size(INPUT_W, INPUT_H));
            img.convertTo(img, CV_32FC3, 1.0/127.5, -1.0); 
            int vol = INPUT_W * INPUT_H;
            for (int c = 0; c < 3; ++c) {
                for (int j = 0; j < vol; ++j) {
                    inputData[c * vol + j] = ((float*)img.data)[j * 3 + c];
                }
            }

            // 内存拷贝并执行 TRT 10 的 enqueueV3
            cudaMemcpyAsync(d_input, inputData.data(), inputData.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
            context->enqueueV3(stream);
            cudaMemcpyAsync(outputData.data(), d_output, outputData.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            // 后处理 (找最大值)
            float maxVal = -1000.0f;
            for (int i = 0; i < NUM_CLASSES; ++i) {
                if (outputData[i] > maxVal) {
                    maxVal = outputData[i];
                    maxIdx = i;
                }
            }
            
            // 把表情文字写在人脸框的上方
            cv::putText(frame, EMOTIONS[maxIdx], cv::Point(faceBox.x, faceBox.y - 10), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
        }

        // 绘制整体 FPS
        auto end = std::chrono::high_resolution_clock::now();
        fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        cv::putText(frame, "FPS: " + std::to_string(int(fps)), cv::Point(20, 40), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        
        cv::imshow("Jetson Orin FER", frame);
        if (cv::waitKey(1) == 27) break; // 按 ESC 退出
    }

    // 6. 清理内存 (TRT 10 直接使用 delete 即可，不再需要 destroy())
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
    delete context;
    delete engine;
    delete runtime;
    
    return 0;
}
