// ============================================================================
// 文件：apriltag_servo.cpp
// 说明：
//       它实现了“眼在手外”（eye-to-hand）构型下的视觉伺服控制，包括：
//         1. 订阅 /image 检测 AprilTag；
//         2. 订阅 /joint_states 获取 UR5 关节角；
//         3. 计算世界系下的雅可比矩阵，转换为末端系；
//         4. 使用 IBVS（基于图像的视觉伺服）计算关节速度命令；
//         5. 发布 /joint_velocity_cmd 驱动机械臂。
//       因为 CoppeliaSim 侧已采用 IK 跟踪，无需直接速度控制。
//       本文件仅供理解原始控制思路，不可直接使用。
// ============================================================================

#include <ros/ros.h>
 #include <std_msgs/Float64MultiArray.h>
 #include <sensor_msgs/Image.h>
 #include <sensor_msgs/JointState.h>

 #include <visp3/core/vpPoint.h>
 #include <visp3/core/vpHomogeneousMatrix.h>
 #include <visp3/core/vpCameraParameters.h>
 #include <visp3/core/vpQuaternionVector.h>
 #include <visp3/core/vpTranslationVector.h>
 #include <visp3/core/vpVelocityTwistMatrix.h>
 #include <visp3/visual_features/vpFeatureBuilder.h>
 #include <visp3/vs/vpServo.h>
 #include <visp3/vs/vpServoDisplay.h>
 #include <visp3/detection/vpDetectorAprilTag.h>
 #include <visp/vpImageConvert.h>
 #include <visp3/gui/vpDisplayX.h>

 #include <cv_bridge/cv_bridge.h>
 #include <atomic>
 #include <vector>
 #include <cstring>
 #include <iostream>
 #include <unordered_map>
 #include <algorithm>

 // ===== 参数 =====
 static const double APRILTAG_SIZE = 0.1; 
 static const int    DESIRED_TAG_ID = -1;   // >=0时只用此ID；-1使用第一个检测到的
 static const size_t DOF = 6;               // UR5 六关节

 // —— UR5 DH 参数（UR 官方常用一组）——
 static const double a[DOF]     = { 0.0,   -0.425,  -0.39225, 0.0,   0.0,    0.0    };
 static const double d[DOF]     = { 0.089159, 0.0,   0.0,     0.10915, 0.09465, 0.0823 };
 static const double alpha[DOF] = { M_PI/2, 0.0,    0.0,     M_PI/2, -M_PI/2, 0.0   };
 // 若你的零位与此不一致，可在此加 joint offset
 static const double theta_offset[DOF] = {0,0,0,0,0,0};

 // —— 轨迹可视化（AprilTag 四角像素点）——
 void display_trajectory(const vpImage<unsigned char> &I, const std::vector<vpImagePoint> &poly)
 {
   const size_t MAX_TRAJ_POINTS = 30;
   static std::vector<vpImagePoint> traj[4];
   for (size_t i = 0; i < 4 && i < poly.size(); ++i) {
     if (traj[i].size() >= MAX_TRAJ_POINTS) traj[i].erase(traj[i].begin());
     traj[i].push_back(poly[i]);
     for (size_t j = 1; j < traj[i].size(); ++j)
       vpDisplay::displayLine(I, traj[i][j - 1], traj[i][j], vpColor::green);
     vpDisplay::displayPoint(I, traj[i].back(), vpColor::red, 3);
   }
 }

 // —— 双缓冲图像 —— 
 struct ImageBuffer {
   vpImage<unsigned char> buffer[2];
   std::atomic<int>  read_index{0};
   std::atomic<bool> updated{false};
 };
 ImageBuffer g_buffer;

 void imageCallback(const sensor_msgs::ImageConstPtr &msg)
 {
   try {
     cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
     if (cv_ptr->image.empty()) return;
     // 与你原流程一致：水平镜像
     cv::flip(cv_ptr->image, cv_ptr->image, 1);

     const int write_index = 1 - g_buffer.read_index.load(std::memory_order_relaxed);
     g_buffer.buffer[write_index].resize(cv_ptr->image.rows, cv_ptr->image.cols);
     std::memcpy(g_buffer.buffer[write_index].bitmap,
                 cv_ptr->image.data,
                 cv_ptr->image.total() * sizeof(uchar));
     g_buffer.read_index.store(write_index, std::memory_order_release);
     g_buffer.updated.store(true, std::memory_order_relaxed);
   } catch (cv_bridge::Exception &e) {
     ROS_ERROR("cv_bridge exception: %s", e.what());
   }
 }

 // —— 关节状态缓存（从 /joint_states 获取）——
 struct JointStateCache {
   std::vector<std::string> desired_names = {
     "joint1","joint2",
     "joint3",
     "joint4",
     "joint5",
     "joint6"
   }; 
   std::unordered_map<std::string, double> pos_map;
   std::vector<double> q{0,0,0,0,0,0};
   bool ready = false;

   void fromMsg(const sensor_msgs::JointState &js) {
     for (size_t i=0;i<js.name.size() && i<js.position.size();++i)
       pos_map[js.name[i]] = js.position[i];

     // 尝试按 desired_names 顺序取出
     std::vector<double> qtmp(DOF, 0.0);
     bool all_found = true;
     for (size_t i=0;i<DOF;++i) {
       auto it = pos_map.find(desired_names[i]);
       if (it == pos_map.end()) { all_found = false; break; }
       qtmp[i] = it->second;
     }

     if (!all_found) {
       // 退化：若条目足够，取前6个
       if (pos_map.size() >= DOF) {
         size_t k=0;
         for (auto &kv: pos_map) { if (k<DOF) { qtmp[k]=kv.second; ++k; } }
         all_found = true;
       }
     }

     if (all_found) { q = qtmp; ready = true; }
   }
 } g_js;

 void jointStateCallback(const sensor_msgs::JointState::ConstPtr &msg) {
   g_js.fromMsg(*msg);
 }

 // —— DH 单节变换 —— 
 static inline vpHomogeneousMatrix dh(double a_i, double alpha_i, double d_i, double theta_i)
 {
   double cth = std::cos(theta_i), sth = std::sin(theta_i);
   double ca  = std::cos(alpha_i), sa = std::sin(alpha_i);
   vpHomogeneousMatrix T;
   T[0][0] =  cth;   T[0][1] = -sth*ca;  T[0][2] =  sth*sa;  T[0][3] = a_i*cth;
   T[1][0] =  sth;   T[1][1] =  cth*ca;  T[1][2] = -cth*sa;  T[1][3] = a_i*sth;
   T[2][0] =  0.0;   T[2][1] =  sa;      T[2][2] =  ca;      T[2][3] = d_i;
   T[3][0] =  0.0;   T[3][1] =  0.0;     T[3][2] =  0.0;     T[3][3] = 1.0;
   return T;
 }

 // —— 计算 T0i（i=0..6）、末端位姿 wMe、以及世界系雅可比 wJ（6x6）——
 struct FKJac {
   vpHomogeneousMatrix T0i[DOF+1]; // T00=I, T06=wMe
   vpMatrix wJ; // 6x6
 };

 static inline vpColVector cross3(const vpColVector &a, const vpColVector &b) {
   vpColVector c(3); c[0]=a[1]*b[2]-a[2]*b[1]; c[1]=a[2]*b[0]-a[0]*b[2]; c[2]=a[0]*b[1]-a[1]*b[0]; return c;
 }

 FKJac computeFKJac_world(const std::vector<double> &q_in)
 {
   FKJac out;
   // 1) 逐关节累乘，得到 T0i
   out.T0i[0].buildFrom(0,0,0,0,0,0); // identity
   for (size_t i=0;i<DOF;++i) {
     const double th = q_in[i] + theta_offset[i];
     vpHomogeneousMatrix A = dh(a[i], alpha[i], d[i], th);
     out.T0i[i+1] = out.T0i[i] * A;
   }
   // 2) 取各关节坐标系 z_i、o_i（i=0..5），以及末端 on
   std::vector<vpColVector> z(DOF), o(DOF+1);
   for (size_t i=0;i<=DOF;++i) {
     o[i].resize(3); 
     o[i][0]=out.T0i[i][0][3];
     o[i][1]=out.T0i[i][1][3];
     o[i][2]=out.T0i[i][2][3];
   }
   for (size_t i=0;i<DOF;++i) {
     z[i].resize(3);
     z[i][0]=out.T0i[i][0][2]; // 第三列是Z轴
     z[i][1]=out.T0i[i][1][2];
     z[i][2]=out.T0i[i][2][2];
   }
   vpColVector on = o[DOF];

   // 3) 世界系雅可比 wJ（6x6），每列： [ z_i × (on - o_i) ; z_i ]
   out.wJ.resize(6,6);
   for (size_t i=0;i<DOF;++i) {
     vpColVector pi = on - o[i];
     vpColVector Ji_lin = cross3(z[i], pi); // 3x1
     // 填充
     out.wJ[0][i] = Ji_lin[0]; out.wJ[1][i] = Ji_lin[1]; out.wJ[2][i] = Ji_lin[2];
     out.wJ[3][i] = z[i][0];   out.wJ[4][i] = z[i][1];   out.wJ[5][i] = z[i][2];
   }
   return out;
 }

 int main(int argc, char **argv)
 {
   ros::init(argc, argv, "eyetohand2");
   ros::NodeHandle nh("~"); // 私有命名空间，便于读参数

   // 可通过参数覆盖期望的 Joint 名称数组
   std::vector<std::string> joint_names_param;
   if (nh.getParam("joint_names", joint_names_param) && joint_names_param.size()==DOF) {
     g_js.desired_names = joint_names_param;
   }

   ros::NodeHandle nh_pub;
   ros::Publisher  qdot_pub = nh_pub.advertise<std_msgs::Float64MultiArray>("/joint_velocity_cmd", 10);

   ros::NodeHandle nh_sub;
   ros::Subscriber img_sub  = nh_sub.subscribe("/image", 10, imageCallback);
   ros::Subscriber js_sub   = nh_sub.subscribe("/joint_states", 50, jointStateCallback);

   vpImage<unsigned char> I;
   vpDisplayX display;
   bool display_initialized = false;
   ros::Rate loop_rate(30); // 视觉回路频率可适当提升

   // 相机内参（像素）
   vpCameraParameters cam(221.7025, 221.7025, 256/2, 256/2);

   // Tag 模型点（中心为原点，z=0）
   std::vector<vpPoint> model_pts{
     vpPoint(-APRILTAG_SIZE/2, -APRILTAG_SIZE/2, 0),
     vpPoint(-APRILTAG_SIZE/2,  APRILTAG_SIZE/2, 0),
     vpPoint( APRILTAG_SIZE/2,  APRILTAG_SIZE/2, 0),
     vpPoint( APRILTAG_SIZE/2, -APRILTAG_SIZE/2, 0)
   };

   // 世界到相机（外参）——需来自你的标定
   vpHomogeneousMatrix wMc;
   wMc[0][0] = -1; wMc[0][1] = 0;  wMc[0][2] = 0;   wMc[0][3] = 0;
   wMc[1][0] = 0;  wMc[1][1] = 0;  wMc[1][2] = 1;  wMc[1][3] = 0;
   wMc[2][0] = 0;  wMc[2][1] = 1;  wMc[2][2] = 0;   wMc[2][3] = 0.8;
   wMc[3][0] = 0;  wMc[3][1] = 0;  wMc[3][2] = 0;   wMc[3][3] = 1;
   vpHomogeneousMatrix cMw = wMc.inverse();

   // 视觉伺服（Eye-to-Hand）
   vpServo task;
   task.setServo(vpServo::EYETOHAND_L_cVe_eJe);
   task.setInteractionMatrixType(vpServo::CURRENT);
   task.setLambda(0.2); // [MOD-4] 调小增益，更接近示例、提高稳定性

   // 特征
   vpFeaturePoint p[4], pd[4];

   // AprilTag
   vpDetectorAprilTag detector(vpDetectorAprilTag::TAG_36h11);
   // [MOD-1] 对齐检测器关键参数（估计方法、抽稀、轴对齐、是否显示）
   detector.setAprilTagPoseEstimationMethod(vpDetectorAprilTag::HOMOGRAPHY_VIRTUAL_VS);
   detector.setAprilTagQuadDecimate(2);
   detector.setZAlignedWithCameraAxis(false);
   detector.setDisplayTag(true);

   while (ros::ok())
   {
     ros::spinOnce();

     // 图像
     const int current_index = g_buffer.read_index.load(std::memory_order_acquire);
     if (g_buffer.updated.exchange(false, std::memory_order_relaxed)) {
       I = g_buffer.buffer[current_index];
       if (!display_initialized && I.getSize() > 0) {
         display.init(I, 0, 0, "AprilTag Eye-to-Hand UR5 qdot");
         display_initialized = true;
       }
     }
     if (!display_initialized ) { loop_rate.sleep(); continue; }

     vpDisplay::display(I);

     try {
       // 1) AprilTag → cMo & 顶点 poly
       std::vector<vpHomogeneousMatrix> cMo_vec;
       const bool found = detector.detect(I, APRILTAG_SIZE, cam, cMo_vec);
       if (!found || cMo_vec.empty()) {
         vpDisplay::flush(I); loop_rate.sleep(); continue;
       }
       const std::vector<int> &ids = detector.getTagsId();
       int chosen = 0;
       if (DESIRED_TAG_ID >= 0) {
         chosen = -1;
         for (size_t i=0;i<ids.size();++i) if (ids[i]==DESIRED_TAG_ID) { chosen=(int)i; break; }
         if (chosen < 0) { vpDisplay::flush(I); loop_rate.sleep(); continue; }
       }
       const vpHomogeneousMatrix &cMo = cMo_vec[chosen];
       const std::vector<vpImagePoint> poly = detector.getPolygon(chosen);
       if (poly.size() < 4) { vpDisplay::flush(I); loop_rate.sleep(); continue; }

       // 2) 构造特征（4点）
       task.kill();
       for (int k=0;k<4;++k) {
         vpFeatureBuilder::create(pd[k], model_pts[k]);  // 期望（保持你原来的方式）
         vpFeatureBuilder::create(p[k],  cam, poly[k]);  // 当前（像素）
         // [MOD-2] 用 cMo 计算并补充当前特征 Z（对 IBVS 的交互矩阵至关重要）
         vpColVector cP;
         model_pts[k].changeFrame(cMo, cP);
         p[k].set_Z(cP[2]);

         task.addFeature(p[k], pd[k]);               
       }

       // 3) 由关节角 q 计算 wMe、wJ、eJe、cVe（保持不变）
       std::vector<double> q = g_js.q; // 弧度
       FKJac fk = computeFKJac_world(q);              // 含 T0i & wJ
       vpHomogeneousMatrix wMe = fk.T0i[DOF];         // 世界到末端
       vpHomogeneousMatrix cMe = cMw * wMe;           // 相机到末端
       vpVelocityTwistMatrix cVe(cMe);
       task.set_cVe(cVe);
       vpHomogeneousMatrix eMw = wMe.inverse();
       vpVelocityTwistMatrix eVw(eMw);
       vpMatrix eJe = eVw * fk.wJ; // 6x6
       task.set_eJe(eJe);

       // 4) 控制律 → 关节速度
       vpColVector qdot = task.computeControlLaw();
       std::cout<<"关节速度"<<qdot<<std::endl;

       // 5) 发布 qdot
       std_msgs::Float64MultiArray cmd;
       cmd.data.resize(qdot.size());
       for (unsigned int i=0;i<qdot.size();++i) cmd.data[i] = qdot[i];
       qdot_pub.publish(cmd);

       // 6) 可视化
       vpDisplay::displayFrame(I, cMo, cam, 0.05, vpColor::none);
       vpServoDisplay::display(task, cam, I, vpColor::green, vpColor::none);
       display_trajectory(I, poly);

       // [MOD-5] 显示误差值，方便调参与对齐示例行为
       double err = task.getError().sumSquare();
       std::ostringstream oss; oss << "error: " << err;
       vpDisplay::displayText(I, 20, std::max(0,(int)I.getWidth()-180), oss.str(), vpColor::red);

     } catch (const vpException &e) {
       ROS_ERROR_STREAM("ViSP exception: " << e.getStringMessage());
     }

     vpDisplay::flush(I);
     loop_rate.sleep();
   }
   return 0;
 }
 ======= tag_vs_follow.cpp =======
 目标：把检测到的 AprilTag 作为“期望目标”，标签动则机械臂跟随，保持末端-目标的固定相对位姿 e_M_o（首次学习）。
 输出：/joint_velocity_cmd (std_msgs/Float64MultiArray)，单位 rad/s

 #include <ros/ros.h>
 #include <sensor_msgs/Image.h>
 #include <sensor_msgs/image_encodings.h>
 #include <sensor_msgs/JointState.h>
 #include <std_msgs/Float64MultiArray.h>
 #include <cv_bridge/cv_bridge.h>

 #include <visp3/core/vpCameraParameters.h>
 #include <visp3/core/vpHomogeneousMatrix.h>
 #include <visp3/core/vpVelocityTwistMatrix.h>
 #include <visp3/core/vpPoint.h>
 #include <visp3/gui/vpDisplayX.h>
 #include <visp3/detection/vpDetectorAprilTag.h>
 #include <visp3/visual_features/vpFeatureBuilder.h>
 #include <visp3/vs/vpServo.h>
 #include <visp3/vs/vpServoDisplay.h>

 #include <opencv2/imgproc.hpp>
 #include <atomic>
 #include <iomanip>
 #include <vector>
 #include <string>
 #include <unordered_map>
 #include <algorithm>
 #include <cstring>
 #include <cmath>

 #ifndef M_PI
 #define M_PI 3.14159265358979323846
 #endif
 bool flip_vertical = true;
 //================ 参数 =================
 struct Params {
   double tag_size = 0.5;                // [m] AprilTag 边长
   int    desired_id = -1;               // -1: 首次自动锁定；>=0: 只能该ID
   double desired_Z =1.75;               // 期望工作距离（相机->目标）
   bool   flip_horizontal = true;       // 是否水平镜像

   // 相机内参（像素）
   double fx = 221.7025, fy = 221.7025, u0 = 256/2.0, v0 = 256/2.0;

   // AprilTag 检测参数
   int    family = (int)vpDetectorAprilTag::TAG_36h11;
   int    pose_method = (int)vpDetectorAprilTag::HOMOGRAPHY_VIRTUAL_VS;
   int    quad_decimate = 1;
   bool   z_aligned = false;
   bool   display_tag_edges = true;

   // 伺服增益与限幅
   double lambda = 0.12;                 // IBVS 增益（稳一点）
   double vel_limit = 1.0;               // 关节速度限幅 [rad/s]
   double qdot_alpha = 0.25;             // 低通滤波系数(0..1)，越小越平滑
 } P;

 //================ 图像缓冲 ================
 struct ImageBuffer {
   vpImage<unsigned char> img[2];
   std::atomic<int> read_idx{0};
   std::atomic<bool> updated{false};
   std::atomic<int> w{0}, h{0};
   std::atomic<bool> got_first{false};
   std::string enc{"?"};
 } gbuf;

 static inline void toGray(const sensor_msgs::ImageConstPtr& msg, cv::Mat &gray)
 {
   cv_bridge::CvImageConstPtr cv_ptr;
   try { cv_ptr = cv_bridge::toCvShare(msg); }
   catch (...) { cv_ptr = cv_bridge::toCvCopy(msg); }

   const std::string &enc = msg->encoding;
   if (enc == sensor_msgs::image_encodings::MONO8 || enc == "8UC1" || enc == "mono8") {
     gray = cv_ptr->image;
   } else if (enc == sensor_msgs::image_encodings::BGR8) {
     cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);
   } else if (enc == sensor_msgs::image_encodings::RGB8) {
     cv::cvtColor(cv_ptr->image, gray, cv::COLOR_RGB2GRAY);
   } else if (enc == sensor_msgs::image_encodings::BGRA8) {
     cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGRA2GRAY);
   } else if (enc == sensor_msgs::image_encodings::RGBA8) {
     cv::cvtColor(cv_ptr->image, gray, cv::COLOR_RGBA2GRAY);
   } else {
     cv_bridge::CvImagePtr mono = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
     gray = mono->image;
   }
 }

 // 回调里：
 void imageCb(const sensor_msgs::ImageConstPtr &msg)
 {
   cv::Mat gray;
   toGray(msg, gray);
   if (gray.empty()) return;

   if (P.flip_horizontal) cv::flip(gray, gray, 1); // 可选：水平镜像
   if (flip_vertical)      cv::flip(gray, gray, 0); // 关键：垂直翻转（把OpenGL底->上改成OpenCV上->下）

   // 再拷贝到 ViSP 缓冲…
   const int write_idx = 1 - gbuf.read_idx.load(std::memory_order_relaxed);
   gbuf.img[write_idx].resize(gray.rows, gray.cols);
   std::memcpy(gbuf.img[write_idx].bitmap, gray.data, gray.total());
   gbuf.read_idx.store(write_idx, std::memory_order_release);
   gbuf.updated.store(true, std::memory_order_relaxed);
   gbuf.w = gray.cols; gbuf.h = gray.rows;
   if (!gbuf.got_first) gbuf.got_first = true;
 }

 //================ 关节状态缓存 ================
 static const size_t DOF = 6; // UR5

 struct JointStateCache {
   std::vector<std::string> desired_names = {
     "joint1","joint2","joint3","joint4","joint5","joint6"
   };
   std::unordered_map<std::string, double> pos_map;
   std::vector<double> q{0,0,0,0,0,0};
   bool ready = false;

   void fromMsg(const sensor_msgs::JointState &js) {
     for (size_t i=0;i<js.name.size() && i<js.position.size();++i)
       pos_map[js.name[i]] = js.position[i];

     std::vector<double> qtmp(DOF, 0.0);
     bool all_found = true;
     for (size_t i=0;i<DOF;++i) {
       auto it = pos_map.find(desired_names[i]);
       if (it == pos_map.end()) { all_found = false; break; }
       qtmp[i] = it->second;
     }
     if (!all_found && pos_map.size() >= DOF) {
       size_t k=0;
       for (auto &kv: pos_map) { if (k<DOF) { qtmp[k]=kv.second; ++k; } }
       all_found = true;
     }
     if (all_found) { q = qtmp; ready = true; }
   }
 } g_js;

 void jointStateCb(const sensor_msgs::JointState::ConstPtr &msg) {
   g_js.fromMsg(*msg);
 }

 //================ UR5 DH、正解与雅可比（世界系） ================
 static const double a_[DOF]     = { 0.0,   -0.425,   -0.39225, 0.0,    0.0,     0.0    };
 static const double d_[DOF]     = { 0.089159, 0.0,   0.0,      0.10915,0.09465, 0.0823  };
 static const double alpha_[DOF] = { M_PI/2, 0.0,     0.0,      M_PI/2, -M_PI/2, 0.0    };
 static const double theta_off_[DOF] = {0,0,0,0,0,0};

 static inline vpHomogeneousMatrix dh(double a, double alpha, double d, double th)
 {
   double cth = std::cos(th), sth = std::sin(th);
   double ca  = std::cos(alpha), sa = std::sin(alpha);
   vpHomogeneousMatrix T;
   T[0][0]=cth;   T[0][1]=-sth*ca; T[0][2]= sth*sa; T[0][3]= a*cth;
   T[1][0]=sth;   T[1][1]= cth*ca; T[1][2]=-cth*sa; T[1][3]= a*sth;
   T[2][0]=0.0;   T[2][1]= sa;     T[2][2]= ca;     T[2][3]= d;
   T[3][0]=0.0;   T[3][1]= 0.0;    T[3][2]= 0.0;    T[3][3]= 1.0;
   return T;
 }

 struct FKJac {
   vpHomogeneousMatrix T0i[DOF+1]; // T00=I, T06=wMe
   vpMatrix wJ; // 6x6
 };

 static inline vpColVector cross3(const vpColVector &a, const vpColVector &b) {
   vpColVector c(3); c[0]=a[1]*b[2]-a[2]*b[1]; c[1]=a[2]*b[0]-a[0]*b[2]; c[2]=a[0]*b[1]-a[1]*b[0]; return c;
 }

 FKJac computeFKJac_world(const std::vector<double> &q_in)
 {
   FKJac out;
   out.T0i[0].buildFrom(0,0,0,0,0,0); // identity
   for (size_t i=0;i<DOF;++i) {
     double th = q_in[i] + theta_off_[i];
     vpHomogeneousMatrix A = dh(a_[i], alpha_[i], d_[i], th);
     out.T0i[i+1] = out.T0i[i] * A;
   }
   std::vector<vpColVector> z(DOF), o(DOF+1);
   for (size_t i=0;i<=DOF;++i) {
     o[i].resize(3);
     o[i][0]=out.T0i[i][0][3];
     o[i][1]=out.T0i[i][1][3];
     o[i][2]=out.T0i[i][2][3];
   }
   for (size_t i=0;i<DOF;++i) {
     z[i].resize(3);
     z[i][0]=out.T0i[i][0][2];
     z[i][1]=out.T0i[i][1][2];
     z[i][2]=out.T0i[i][2][2];
   }
   vpColVector on = o[DOF];

   out.wJ.resize(6,6);
   for (size_t i=0;i<DOF;++i) {
     vpColVector pi = on - o[i];
     vpColVector Ji_lin = cross3(z[i], pi);
     out.wJ[0][i]=Ji_lin[0]; out.wJ[1][i]=Ji_lin[1]; out.wJ[2][i]=Ji_lin[2];
     out.wJ[3][i]=z[i][0];   out.wJ[4][i]=z[i][1];   out.wJ[5][i]=z[i][2];
   }
   return out;
 }

 //================ 主程序 =================
 int main(int argc, char **argv)
 {
   ros::init(argc, argv, "tag_vs_ibvs_follow", ros::init_options::AnonymousName);
   ros::NodeHandle nh("~");

   //—— 参数 ——//
   nh.param("tag_size", P.tag_size, P.tag_size);
   nh.param("desired_id", P.desired_id, P.desired_id);
   nh.param("desired_Z", P.desired_Z, P.desired_Z);
   nh.param("flip_horizontal", P.flip_horizontal, P.flip_horizontal);
   nh.param("fx", P.fx, P.fx); nh.param("fy", P.fy, P.fy);
   nh.param("u0", P.u0, P.u0); nh.param("v0", P.v0, P.v0);
   nh.param("family", P.family, P.family);
   nh.param("pose_method", P.pose_method, P.pose_method);
   nh.param("quad_decimate", P.quad_decimate, P.quad_decimate);
   nh.param("z_aligned", P.z_aligned, P.z_aligned);
   nh.param("display_tag_edges", P.display_tag_edges, P.display_tag_edges);
   nh.param("lambda", P.lambda, P.lambda);
   nh.param("vel_limit", P.vel_limit, P.vel_limit);
   nh.param("qdot_alpha", P.qdot_alpha, P.qdot_alpha);

   // 覆盖关节名
   std::vector<std::string> joint_names_param;
   if (nh.getParam("joint_names", joint_names_param) && joint_names_param.size()==DOF)
     g_js.desired_names = joint_names_param;

   // 订阅/发布
   ros::Subscriber subImg = nh.subscribe<sensor_msgs::Image>("/image", 10, imageCb);
   ros::Subscriber subJS  = nh.subscribe<sensor_msgs::JointState>("/joint_states", 50, jointStateCb);
   ros::Publisher  pub_qdot = nh.advertise<std_msgs::Float64MultiArray>("/joint_velocity_cmd", 10);

   // 显示
   vpImage<unsigned char> I;
   vpDisplayX display;
   bool display_ready = false;

   // 相机内参
   vpCameraParameters cam; cam.initPersProjWithoutDistortion(P.fx, P.fy, P.u0, P.v0);

   // AprilTag
   vpDetectorAprilTag detector((vpDetectorAprilTag::vpAprilTagFamily)P.family);
   detector.setAprilTagPoseEstimationMethod((vpDetectorAprilTag::vpPoseEstimationMethod)P.pose_method);
   detector.setAprilTagQuadDecimate(P.quad_decimate);
   detector.setZAlignedWithCameraAxis(P.z_aligned);
   detector.setDisplayTag(P.display_tag_edges);

   // —— 外参：世界->相机（wMc） —— //
   vpHomogeneousMatrix wMc;
   wMc[0][0] = 0; wMc[0][1] = 0;  wMc[0][2] = -1;    wMc[0][3] = -0.475;
   wMc[1][0] = 0;  wMc[1][1] = 1;  wMc[1][2] = 0;    wMc[1][3] = -0.25;
   wMc[2][0] = 1;  wMc[2][1] = 0;  wMc[2][2] = 0;    wMc[2][3] = 0.7;
   wMc[3][3] = 1;

   // —— IBVS 任务 —— 
   vpServo task;
   task.setServo(vpServo::EYETOHAND_L_cVe_eJe);
   task.setInteractionMatrixType(vpServo::CURRENT);
   task.setLambda(P.lambda);

   // 模型四点（标签平面 z=0，中心原点，注意点序与 detector polygon 对应）
   std::vector<vpPoint> model_pts{
     vpPoint(-P.tag_size/2, -P.tag_size/2, 0),
     vpPoint(-P.tag_size/2,  P.tag_size/2, 0),
     vpPoint( P.tag_size/2,  P.tag_size/2, 0),
     vpPoint( P.tag_size/2, -P.tag_size/2, 0)
   };
   vpFeaturePoint p[4], pd[4];

   // 跟随状态
   int  tracked_id = -1;
   bool have_eMo   = false;
   vpHomogeneousMatrix e_M_o; // 末端->目标（保持相对）

   // 期望相机->目标（正对 + 固定距离）
   vpHomogeneousMatrix cd_M_o_des; // R=I, t=(0,0,desired_Z)
   cd_M_o_des[0][0]=1; cd_M_o_des[1][1]=1; cd_M_o_des[2][2]=1; cd_M_o_des[3][3]=1;
   cd_M_o_des[2][3]=P.desired_Z;

   vpColVector qdot_last(DOF); qdot_last = 0;

   ros::Rate rate(30);
   int log_tick = 0;

   while (ros::ok()) {
     ros::spinOnce();

     // 图像更新
     const int idx = gbuf.read_idx.load(std::memory_order_acquire);
     if (gbuf.updated.exchange(false)) {
       I = gbuf.img[idx];
       if (!display_ready && I.getSize() > 0) {
         display.init(I, 50, 50, "AprilTag IBVS Follow");
         display_ready = true;
       }
     }
     if (!display_ready) { rate.sleep(); continue; }

     vpDisplay::display(I);

     // 检测
     std::vector<vpHomogeneousMatrix> cMo_vec;
     bool found = detector.detect(I, P.tag_size, cam, cMo_vec);

     // UI & 丢失处理
     {
       std::ostringstream os; os << "tags: " << (found ? detector.getNbObjects() : 0)
                                 << " | js_ready: " << (g_js.ready?"Y":"N");
       vpDisplay::displayText(I, 10, 10, os.str(), found ? vpColor::green : vpColor::red);
       if ((log_tick++ % 60) == 0) {
         ROS_INFO_STREAM("[apriltag] detect=" << (found?"true":"false")
                         << " count=" << (found ? (int)detector.getNbObjects() : 0)
                         << " js_ready=" << (g_js.ready?"true":"false"));
       }
     }
     if (!(found && detector.getNbObjects() > 0) || !g_js.ready) {
       std_msgs::Float64MultiArray cmd; cmd.data.assign(DOF, 0.0);
       pub_qdot.publish(cmd);
       vpDisplay::flush(I); rate.sleep(); continue;
     }

     // 选择目标 ID（锁定）
     int chosen = -1;
     const std::vector<int> ids = detector.getTagsId();
     if (P.desired_id >= 0) {
       for (size_t i=0;i<ids.size();++i) if (ids[i]==P.desired_id) { chosen=(int)i; break; }
     } else {
       if (tracked_id < 0) { tracked_id = ids[0]; chosen = 0; }
       else { for (size_t i=0;i<ids.size();++i) if (ids[i]==tracked_id) { chosen=(int)i; break; } }
     }
     if (chosen < 0) {
       std_msgs::Float64MultiArray cmd; cmd.data.assign(DOF, 0.0);
       pub_qdot.publish(cmd);
       vpDisplay::flush(I); rate.sleep(); continue;
     }

     // 当前相机->目标
     const vpHomogeneousMatrix &c_M_o = cMo_vec[chosen];
     if (P.display_tag_edges) vpDisplay::displayFrame(I, c_M_o, cam, P.tag_size*0.5, vpColor::none, 3);

     // 世界姿态、雅可比
     FKJac fk_now = computeFKJac_world(g_js.q);     // 世界->末端
     vpHomogeneousMatrix w_M_e = fk_now.T0i[DOF];
     vpHomogeneousMatrix w_M_c = wMc;               // 世界->相机
     vpHomogeneousMatrix w_M_o = w_M_c * c_M_o;     // 世界->目标

     // 学习 e_M_o（保持末端-目标相对）
     if (!have_eMo) {
       e_M_o = w_M_e.inverse() * w_M_o;
       have_eMo = true;
       ROS_INFO_STREAM("[init] learned e_M_o:\n" << e_M_o);
     }

     // === IBVS 特征构造 ===
     task.kill();
     task.setServo(vpServo::EYETOHAND_L_cVe_eJe);
     task.setInteractionMatrixType(vpServo::CURRENT);
     task.setLambda(P.lambda);

     // cVe：用当前 c_M_e = c_M_o * (e_M_o)^-1
     vpHomogeneousMatrix c_M_e = c_M_o * e_M_o.inverse();
     vpVelocityTwistMatrix cVe(c_M_e);
     task.set_cVe(cVe);

     // eJe：wJ 转到末端系
     vpHomogeneousMatrix e_M_w = w_M_e.inverse();
     vpVelocityTwistMatrix eVw(e_M_w);
     vpMatrix eJe = eVw * fk_now.wJ;
     task.set_eJe(eJe);

     // 角点 polygon（像素坐标）
     const auto poly = detector.getPolygon(chosen);
     if (poly.size() < 4) {
       std_msgs::Float64MultiArray cmd; cmd.data.assign(DOF, 0.0);
       pub_qdot.publish(cmd);
       vpDisplay::flush(I); rate.sleep(); continue;
     }

     // 当前特征 p：由像素构建，并为每个点设置当前深度 Z（由 c_M_o + 模型点获得）
     for (int k=0;k<4;++k) {
       vpFeatureBuilder::create(p[k], cam, poly[k]);       // 用像素得到 (x,y)
       vpColVector cP; model_pts[k].changeFrame(c_M_o, cP); // 当前 c 系下深度
       p[k].set_Z(cP[2]);

       // 期望特征 pd：把模型点放在期望相机位姿 cd_M_o_des 下，得到 (x*,y*,Z*)
       vpColVector cdP; model_pts[k].changeFrame(cd_M_o_des, cdP);
       const double xstar = cdP[0]/cdP[2];
       const double ystar = cdP[1]/cdP[2];
       pd[k].set_x(xstar);
       pd[k].set_y(ystar);
       pd[k].set_Z(cdP[2]); // = desired_Z

       task.addFeature(p[k], pd[k]);
     }

     // 控制律
     vpColVector qdot = task.computeControlLaw();

     // 低通 + 限幅
     for (unsigned int i=0;i<qdot.size();++i) {
       qdot[i] = P.qdot_alpha*qdot[i] + (1.0-P.qdot_alpha)*qdot_last[i];
       qdot[i] = std::max(-P.vel_limit, std::min(P.vel_limit, qdot[i]));
     }
     qdot_last = qdot;

     // 发布
     std_msgs::Float64MultiArray cmd;
     cmd.data.resize(qdot.size());
     for (unsigned int i=0;i<qdot.size();++i) cmd.data[i] = qdot[i];
     pub_qdot.publish(cmd);

     // 可视化
     vpServoDisplay::display(task, cam, I, vpColor::green, vpColor::none);
     for (size_t k=0;k<poly.size();++k) {
       vpDisplay::displayCross(I, poly[k], 10, vpColor::red, 2);
     }
     double err = task.getError().sumSquare();
     std::ostringstream es; es << "IBVS err: " << err << " | id: " << (P.desired_id>=0?P.desired_id:tracked_id);
     vpDisplay::displayText(I, 30, 10, es.str(), vpColor::yellow);

     vpDisplay::flush(I);
     rate.sleep();
   }
   return 0;
 }
