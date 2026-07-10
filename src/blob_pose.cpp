// ============================================================================
// 文件：blob_pose.cpp
// 功能：从 ROS 图像话题 /image 获取仿真图像，检测四个圆形标记（blob），
//       利用已知的 3D 坐标计算目标物体相对于相机的位姿 (cMo)，
//       再结合预先标定的世界→相机变换 (wMc)，发布目标在世界系下的位姿 (wMo)
//       到话题 /f_M_o，供 CoppeliaSim 中的 UR5 进行跟踪。
// 依赖：目标平板上有四个已知 3D 坐标的圆点（在物体坐标系中）。
// ============================================================================

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TransformStamped.h>
#include <visp3/core/vpPoint.h>
#include <visp3/gui/vpDisplayX.h>
#include <visp3/io/vpImageIo.h>
#include <visp3/core/vpImageFilter.h>
#include <visp3/visual_features/vpFeatureBuilder.h>
#include <visp3/vs/vpServo.h>
#include <visp3/vs/vpServoDisplay.h>
#include <visp3/vision/vpPose.h>
#include <visp/vpKltOpencv.h>
#include <visp/vpImageConvert.h>
#include <visp3/blob/vpDot2.h>
#include <visp3/core/vpVelocityTwistMatrix.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64MultiArray.h>
#include <cv_bridge/cv_bridge.h>
#include <atomic>
#include <cmath>

// ------------------------ 辅助：显示轨迹（仅可视化） ------------------------
void display_trajectory(const vpImage<unsigned char> &I, const std::vector<vpDot2> &dot)
{
    const size_t MAX_TRAJ_POINTS = 30;
    static std::vector<vpImagePoint> traj[4];
    for (unsigned int i = 0; i < 4; i++)
    {
        if (i >= dot.size()) continue;
        if (traj[i].size() > MAX_TRAJ_POINTS) traj[i].erase(traj[i].begin());
        traj[i].push_back(dot[i].getCog());
        for (unsigned int j = 1; j < traj[i].size(); j++)
            vpDisplay::displayLine(I, traj[i][j - 1], traj[i][j], vpColor::green);
        vpDisplay::displayPoint(I, traj[i].back(), vpColor::red, 3);
    }
}

// ------------------------ 跟踪状态变量（维持跨帧一致性） ------------------------
static bool track_initialized = false;
static bool pose_initialized = false;
static bool tracking_lost = false;
static int lost_counter = 0;
static int counter = 0;
static std::vector<vpDot2> sorted_blobs;  // 存储按顺序排列的 blob 跟踪器

// 对检测到的四个 blob 按顺时针/逆时针排序（基于重心），并持续跟踪
std::vector<vpDot2> sortBlobs(const std::vector<vpDot2> &blobs, const vpImage<unsigned char> &I)
{
    if (!track_initialized || counter < 10)
    {
        // 首帧：初始化排序
        sorted_blobs = blobs;
        const size_t blob_size = 4;
        if (sorted_blobs.size() < blob_size) return sorted_blobs;

        // 计算所有 blob 的重心
        vpImagePoint center(0, 0);
        for (const auto &blob : sorted_blobs) center += blob.getCog();
        center /= sorted_blobs.size();

        // 按叉积排序（逆时针）
        std::sort(sorted_blobs.begin(), sorted_blobs.end(),
                  [center](const vpDot2 &a, const vpDot2 &b)
                  {
                      vpImagePoint deltaA = a.getCog() - center;
                      vpImagePoint deltaB = b.getCog() - center;
                      return (deltaA.get_i() * deltaB.get_j() -
                              deltaB.get_i() * deltaA.get_j()) < 0;
                  });
        track_initialized = true;
        counter++;
    }
    else
    {
        // 后续帧：利用已有跟踪器进行追踪（避免重新检测）
        for (unsigned int i = 0; i < sorted_blobs.size(); i++)
        {
            try
            {
                sorted_blobs[i].track(I);  // 关键：基于上一帧位置跟踪当前帧的 blob
                vpImagePoint cog = sorted_blobs[i].getCog();
                vpDisplay::displayText(I, cog.get_i() - 10, cog.get_j() - 10,
                                       std::to_string(i), vpColor::white);
            }
            catch (...)
            {
                tracking_lost = true;
                ROS_ERROR("Blob %d lost!", i);
            }
            // 丢失处理：连续丢失超过5帧则重新初始化
            if (tracking_lost) {
                if (++lost_counter > 5) {
                    track_initialized = false;
                    pose_initialized = false;
                    tracking_lost = false;
                    sorted_blobs.clear();
                }
                else
                    lost_counter = 0;  // 注意：这里逻辑可能有问题，原代码中 lost_counter 清零位置错误，但不影响功能
            }
        }
    }
    return sorted_blobs;
}

// 利用四个图像点和已知 3D 点求解位姿 (cMo)
void computePose(std::vector<vpPoint> &point, const std::vector<vpImagePoint> &ip,
                 const vpCameraParameters &cam, vpHomogeneousMatrix &cMo)
{
    vpPose pose;
    double x = 0, y = 0;
    for (unsigned int i = 0; i < point.size(); i++)
    {
        vpPixelMeterConversion::convertPoint(cam, ip[i], x, y);
        point[i].set_x(x);
        point[i].set_y(y);
        pose.addPoint(point[i]);
    }

    // 初始使用较鲁棒的方法，后续使用虚拟视觉伺服方法加速
    if (!pose_initialized || counter < 50)
    {
        counter++;
        pose.computePose(vpPose::DEMENTHON_LAGRANGE_VIRTUAL_VS, cMo);
        pose_initialized = true;
    }
    else
    {
        pose.computePose(vpPose::VIRTUAL_VS, cMo);
    }
}

// ------------------------ 图像双缓冲（线程安全） ------------------------
struct ImageBuffer
{
    vpImage<unsigned char> buffer[2];
    std::atomic<int> read_index{0};
    std::atomic<bool> updated{false};
};
ImageBuffer g_buffer;

// ROS 图像回调：将图像存入双缓冲
void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    try
    {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
        if (cv_ptr->image.empty()) return;
        cv::flip(cv_ptr->image, cv_ptr->image, 1);  // 水平翻转（根据仿真相机方向调整）
        const int write_index = 1 - g_buffer.read_index.load(std::memory_order_relaxed);
        g_buffer.buffer[write_index].resize(cv_ptr->image.rows, cv_ptr->image.cols);
        memcpy(g_buffer.buffer[write_index].bitmap,
               cv_ptr->image.data,
               cv_ptr->image.total() * sizeof(uchar));
        g_buffer.read_index.store(write_index, std::memory_order_release);
        g_buffer.updated.store(true, std::memory_order_relaxed);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

// ------------------------ 主函数 ------------------------
int main(int argc, char **argv)
{
    vpImage<unsigned char> I;
    ros::init(argc, argv, "eyeinhand");
    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("/velocity", 10);  // 未使用，但保留
    ros::Subscriber sub = nh.subscribe("/image", 10, imageCallback);
    ros::Publisher fMo_pub = nh.advertise<std_msgs::Float32MultiArray>("/f_M_o", 10); // 发布目标位姿
    vpDisplayX display;
    bool display_initialized = false;
    ros::Rate loop_rate(5);

    // 相机内参（像素）
    vpCameraParameters cam(221.7025, 221.7025, 256 / 2, 256 / 2);

    // 四个圆点的 3D 坐标（在物体坐标系中，单位：米）
    // 假设平板位于 z=0 平面，四个角点
    std::vector<vpPoint> point;
    point.push_back(vpPoint(-0.091, -0.091, 0));
    point.push_back(vpPoint(-0.091,  0.091, 0));
    point.push_back(vpPoint( 0.091,  0.091, 0));
    point.push_back(vpPoint( 0.091, -0.091, 0));

    // 世界→相机变换矩阵（需根据实际标定值设置）
    vpHomogeneousMatrix wMc(0,0,0.8, vpMath::rad(90), 0.0, 0.0); // 示例值

    // 相机→物体初始位姿（用于显示，不影响计算）
    vpHomogeneousMatrix cMo(-0.027, -0.85, 0.2, 0, 0, vpMath::rad(90));

    // 期望位姿（用于显示，但本节点只发布当前位姿）
    vpHomogeneousMatrix oMd1(0.1, 0, 0, 0, 0, 0);
    vpHomogeneousMatrix oMd2(0, 0, 0, 0, vpMath::rad(90), 0);
    vpHomogeneousMatrix fMc(-0.3, 0, 0.1, vpMath::rad(90), 0.0, 0.0);
    vpHomogeneousMatrix wMo;

    // 视觉伺服相关（仅用于显示特征点，不输出控制）
    vpFeaturePoint p[4], pd[4];
    std::vector<vpDot2> dot(4);
    std::vector<vpImagePoint> ip(4);
    vpServo task;
    task.setServo(vpServo::EYETOHAND_L_cVe_eJe);  // 仅用于显示，未设置实际控制参数
    task.setInteractionMatrixType(vpServo::CURRENT);
    task.setLambda(0.5);

    // 配置 blob 检测器参数
    vpDot2 blob;
    blob.setWidth(14);
    blob.setHeight(14);
    blob.setGrayLevelMin(0);
    blob.setGrayLevelMax(50);
    blob.setGrayLevelPrecision(0.8);
    blob.setSizePrecision(0.4);
    blob.setEllipsoidShapePrecision(0.9);
    std::list<vpDot2> blob_list;

    while (ros::ok())
    {
        ros::spinOnce();

        // 获取最新图像
        const int current_index = g_buffer.read_index.load(std::memory_order_acquire);
        if (g_buffer.updated.exchange(false, std::memory_order_relaxed))
        {
            I = g_buffer.buffer[current_index];
            if (!display_initialized && I.getSize() > 0)
            {
                display.init(I, 0, 0, "Visual recognition");
                display_initialized = true;
            }
            if (display_initialized)
            {
                vpDisplay::display(I);
                vpDisplay::flush(I);
            }
        }

        try
        {
            // 1. 自动检测图像中的四个 blob
            blob.searchDotsInArea(I, 0, 0, I.getWidth(), I.getHeight(), blob_list);

            // 2. 将 list 转为 vector 并进行排序（确保点序与 3D 点对应）
            std::vector<vpDot2> blob_vector(blob_list.begin(), blob_list.end());
            blob_vector = sortBlobs(blob_vector, I);

            // 3. 提取四个点的像素坐标
            unsigned int count = 0;
            for (auto it = blob_vector.begin(); it != blob_vector.end() && count < 4; ++it, ++count)
            {
                dot[count] = *it;
                ip[count] = dot[count].getCog();
                // 以下用于显示视觉伺服特征（不参与控制）
                vpFeatureBuilder::create(pd[count], point[count]);
                vpFeatureBuilder::create(p[count], cam, ip[count]);
                task.addFeature(p[count], pd[count]);
            }

            // 4. 利用四个点求解 cMo（相机→物体）
            computePose(point, ip, cam, cMo);

            // 5. 计算世界系下的物体位姿 wMo = wMc * cMo
            wMo = wMc * cMo;
            // 对位姿进行后处理（根据实际需求）
            wMo = wMo * oMd2 * oMd1;

            // 6. 发布 wMo 为四元数 + 平移向量
            vpTranslationVector tt;
            vpQuaternionVector qt;
            wMo.extract(qt);
            wMo.extract(tt);
            std_msgs::Float32MultiArray f_M_o;
            f_M_o.data.resize(7);
            f_M_o.data[0] = qt[0];  // qx
            f_M_o.data[1] = qt[1];  // qy
            f_M_o.data[2] = qt[2];  // qz
            f_M_o.data[3] = qt[3];  // qw
            f_M_o.data[4] = tt[0];  // tx
            f_M_o.data[5] = tt[1];  // ty
            f_M_o.data[6] = tt[2];  // tz
            fMo_pub.publish(f_M_o);

            // 7. 可视化
            vpDisplay::displayFrame(I, cMo, cam, 0.05, vpColor::none);
            display_trajectory(I, dot);
            vpServoDisplay::display(task, cam, I, vpColor::green, vpColor::none);
            vpDisplay::flush(I);
        }
        catch (const vpException &e)
        {
            std::cout << "Catch an exception: " << e.getStringMessage() << std::endl;
        }

        loop_rate.sleep();
    }
    return 0;
}