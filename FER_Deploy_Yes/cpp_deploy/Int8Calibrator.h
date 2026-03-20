/**
 * Int8Calibrator.h
 * 对应技术报告 3.3 节：C++校准器的深度实现
 * 功能：读取校准数据集，为 TensorRT 提供 INT8 量化所需的统计数据
 */

#ifndef INT8_CALIBRATOR_H
#define INT8_CALIBRATOR_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iterator>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

// 继承自 TensorRT 的熵校准器接口
class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator2(int batchSize, int inputW, int inputH, const std::string& calibDataDir, const std::string& calibTableName, const std::string& inputBlobName)
        : mBatchSize(batchSize), mInputW(inputW), mInputH(inputH), mImgDir(calibDataDir), mCalibTableName(calibTableName), mInputBlobName(inputBlobName) 
    {
        // 1. 获取校准数据集中的所有图片路径
        std::vector<cv::String> fileNames;
        cv::glob(mImgDir + "/*.jpg", fileNames, false);
        mFileNames = fileNames;
        
        // 随机打乱或截取前 500 张用于校准 (报告建议 500-1000 张)
        if (mFileNames.size() > 500) {
            mFileNames.resize(500);
        }

        mInputCount = mBatchSize * 3 * mInputH * mInputW;
        mCurBatchIdx = 0;
        
        // 分配 GPU 显存用于存放一个 Batch 的数据
        cudaMalloc(&mDeviceInput, mInputCount * sizeof(float));
    }

    virtual ~Int8EntropyCalibrator2() {
        cudaFree(mDeviceInput);
    }

    // TensorRT 会循环调用此函数获取 Batch 数据
    int getBatchSize() const noexcept override {
        return mBatchSize;
    }

    // 核心函数：读取图片 -> 预处理 -> 拷贝到 GPU
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
        if (mCurBatchIdx + mBatchSize > mFileNames.size())
            return false; // 数据读完了

        std::vector<float> batchData(mInputCount);
        
        // 读取一个 Batch 的图片
        for (int i = 0; i < mBatchSize; i++) {
            std::string fileName = mFileNames[mCurBatchIdx + i];
            cv::Mat img = cv::imread(fileName);
            
            // --- 必须与训练时的预处理完全一致！---
            // 1. Resize 112x112
            cv::resize(img, img, cv::Size(mInputW, mInputH));
            // 2. BGR 转 RGB
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            // 3. 归一化 (x - 127.5) / 127.5  ==> (x/255 - 0.5)/0.5
            img.convertTo(img, CV_32FC3, 1.0/127.5, -1.0);

            // 4. HWC 转 CHW (OpenCV 是 HWC，TensorRT 需要 CHW)
            // 这是一个很多初学者容易写错的地方，导致量化失败
            int volImg = mInputW * mInputH;
            for (int c = 0; c < 3; ++c) {
                for (int j = 0; j < volImg; ++j) {
                    batchData[i * 3 * volImg + c * volImg + j] = ((float*)img.data)[j * 3 + c];
                }
            }
        }

        // 拷贝到 GPU
        cudaMemcpy(mDeviceInput, batchData.data(), mInputCount * sizeof(float), cudaMemcpyHostToDevice);
        bindings[0] = mDeviceInput; // 绑定输入节点

        std::cout << "Calibrating batch " << mCurBatchIdx << " to " << mCurBatchIdx + mBatchSize << std::endl;
        mCurBatchIdx += mBatchSize;
        return true;
    }

    // 读取缓存表 (避免每次运行都重新校准)
    const void* readCalibrationCache(size_t& length) noexcept override {
        mCalibrationCache.clear();
        std::ifstream input(mCalibTableName, std::ios::binary);
        input >> std::noskipws;
        if (input.good()) {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    // 写入缓存表
    void writeCalibrationCache(const void* cache, size_t length) noexcept override {
        std::ofstream output(mCalibTableName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    int mBatchSize;
    int mInputW, mInputH;
    int mInputCount;
    int mCurBatchIdx;
    std::string mImgDir;
    std::string mCalibTableName;
    std::string mInputBlobName;
    std::vector<std::string> mFileNames;
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache;
};

#endif // INT8_CALIBRATOR_H