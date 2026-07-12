// ============================================================================
// 文件：general_calculation.cpp
// 功能：离线处理一系列深度图像（PGM 格式），提取目标区域中心，
//       估算目标在三维空间中的运动速度，并生成带标注的伪彩色视频。
// 说明：本程序实际上应该集成到视觉感知环节中，用于通用物体（无blob和apriltag）的相关计算
// 输入：深度图像序列（16-bit PGM），命名格式 depth_frame_%04d.pgm。
// 输出：AVI 视频文件，每帧叠加检测框、中心点、深度值与速度信息。
// ============================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdio>

// ======================== 参数配置区 ============================

// 示例图像
const std::string IMG1 = "depth_t1.pgm";
const std::string IMG2 = "depth_t2.pgm";

// 深度阈值（毫米），用于滤除背景噪声
const ushort    DEPTH_MIN = 5;            // 最小有效深度（mm）
const ushort    DEPTH_MAX = 200;          // 最大有效深度（mm）
const int       MARGIN = 3;               // 检测框外扩像素数
const double    DT = 0.1;                 // 前后帧时间间隔（秒）
const double    f = 221.7;                // 相机焦距（像素），用于像素速度转米/秒

// ================================================================

// 功能：获取深度图中有效区域的外接矩形及中心点
// 输入：d16 - 16-bit 深度图 (CV_16U)
// 输出：center - 有效区域中心像素坐标
// 返回：cv::Rect - 有效区域的外接矩形（已外扩并裁剪）
cv::Rect getRectAndCenter(const cv::Mat& d16, cv::Point& center)
{
    // 1. 根据深度阈值创建掩码
    cv::Mat mask;
    cv::inRange(d16, DEPTH_MIN, DEPTH_MAX, mask);

    // 2. 查找轮廓（仅外部轮廓）
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(mask, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (cs.empty()) {
        center = cv::Point(-1, -1);
        return cv::Rect();
    }

    // 3. 合并所有轮廓点，计算包围盒
    std::vector<cv::Point> all;
    for (auto& c : cs) all.insert(all.end(), c.begin(), c.end());
    cv::Rect r = cv::boundingRect(all);

    // 4. 外扩并裁剪至图像边界
    r.x = std::max(0, r.x - MARGIN);
    r.y = std::max(0, r.y - MARGIN);
    r.width = std::min(d16.cols - r.x, r.width + 2 * MARGIN);
    r.height = std::min(d16.rows - r.y, r.height + 2 * MARGIN);

    // 5. 计算中心点
    center = cv::Point(r.x + r.width / 2, r.y + r.height / 2);
    return r;
}

int main()
{
    // ---- 输出视频配置 ----
    const int FRAME_NUM = 25;              // 要处理的帧数（1~25）
    const double fps = 8.0;                // 输出视频帧率
    const std::string outVideo = "depth_result.avi";

    std::vector<cv::Mat> frames;           // 存放每一帧生成的彩色图像
    cv::Size frameSize;                    // 记录帧尺寸

    // ---- 循环处理每一帧 ----
    for (int i = 1; i <= FRAME_NUM; ++i)
    {
        // 1. 构造文件名并读取深度图（16-bit）
        char fname[256];
        sprintf_s(fname, "depth_frame_0002\\depth_frame_%04d.pgm", i);
        cv::Mat d = cv::imread(fname, cv::IMREAD_ANYDEPTH);
        if (d.empty()) {
            std::cout << "读取失败: " << fname << "\n";
            continue;
        }

        /*---- 2. 检测有效区域，获取中心点 ----*/
        cv::Point c2;
        cv::Rect  r2 = getRectAndCenter(d, c2);
        if (c2.x < 0) {
            std::cout << "未检测到目标在 " << fname << "\n";
            continue;
        }

        /*---- 3. 获取中心点深度（灰度值） ----*/
        // 注意：深度图实际为 16-bit，但此处强制转 uchar 读取，只能得到低 8 位，可能丢失精度
        // 更合理的方式应使用 d.at<ushort>(c2.y, c2.x) 获取真实毫米值
        uchar gray = d.at<uchar>(c2.y, c2.x);

        /*---- 4. 计算沿光轴速度 vz ----*/
        static double depthPrev = 0;                  // 上一帧深度（米）
        // 将灰度值映射到 [0.3, 1.3] 米，仅作示例，非真实深度转换
        double depthM = 0.3 + gray / 255.0 * 1.0;     // 当前帧深度（米）
        std::cout << depthM << '\n';
        double vz_mps = (i == 1) ? 0.0 : (depthPrev - depthM) / DT; // 注意：符号约定 (前-后)/dt
        depthPrev = depthM;

        /*---- 5. 计算水平方向速度 vx, vy ----*/
        static cv::Point c_prev = c2;
        double vx_px = (i == 1) ? 0.0 : (c2.x - c_prev.x) / DT;
        double vy_px = (i == 1) ? 0.0 : (c2.y - c_prev.y) / DT;
        c_prev = c2;
        // 利用针孔模型：v_mps = v_px * depth / f
        double vx_mps = vx_px * depthM / f;
        double vy_mps = vy_px * depthM / f;

        /*---- 6. 生成伪彩色深度图 ----*/
        // 使用掩码获取有效区域的最小/最大深度
        double minVal, maxVal;
        cv::Mat mask;
        cv::inRange(d, DEPTH_MIN, DEPTH_MAX, mask);
        cv::minMaxIdx(d, &minVal, &maxVal, nullptr, nullptr, mask);
        // 将有效区域线性映射到 0~255
        cv::Mat obj8;
        d.convertTo(obj8, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        cv::Mat show8 = cv::Mat::zeros(d.size(), CV_8UC1);
        obj8.copyTo(show8, mask);
        // 应用伪彩色映射
        cv::Mat color;
        cv::applyColorMap(show8, color, cv::COLORMAP_JET);
        // 将背景置为黑色
        cv::Mat bgColor(color.size(), CV_8UC3, cv::Vec3b(0, 0, 0));
        color.copyTo(bgColor, mask);
        color = bgColor;

        /*---- 7. 绘制检测框和中心点 ----*/
        cv::rectangle(color, r2, cv::Scalar(0, 255, 0), 2);
        cv::circle(color, c2, 4, cv::Scalar(0, 0, 255), -1);

        /*---- 8. 叠加文字信息 ----*/
        cv::putText(color, cv::format("frame %04d", i),
            cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
        cv::putText(color, cv::format("center: (%d,%d)  depth: %.3f m", c2.x, c2.y, depthM),
            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
        cv::putText(color, cv::format("vx = %.3f m/s  vy = %.3f m/s", vx_mps, vy_mps),
            cv::Point(10, 45), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
        cv::putText(color, cv::format("vz = %.3f m/s", vz_mps),
            cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);

        /*---- 9. 保存当前帧 ----*/
        frames.push_back(color.clone());
        if (frameSize.empty()) frameSize = color.size();
    }

    /*======== 写出 AVI 视频 ========*/
    if (frames.empty()) {
        std::cout << "无有效帧！\n";
        return -1;
    }

    cv::VideoWriter writer(outVideo, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frameSize);
    for (const auto& fr : frames) writer.write(fr);
    writer.release();
    std::cout << "视频已保存: " << outVideo << "\n";
    return 0;
}
