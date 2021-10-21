#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>
using namespace std;
using namespace cv;
using namespace Eigen;

// camera instrinstc
const double camera_factor = 1;
const double camera_cx = 695.5;
const double camera_cy = 360.5;
const double camera_fx = 700;
const double camera_fy = 700;
typedef Matrix<double,6,1> Vector6d;

vector<Vector6d, Eigen::aligned_allocator<Vector6d>> compute_depth(cv::Mat &disparity, cv::Mat &color_image, int div=64)
{
    const double fx = 700;
    const double baseline = 0.5;

    Mat depth(disparity.rows, disparity.cols, CV_16S);  //深度图
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;

    cout << depth.type() << endl;
    //视差图转深度图
    for (int row = 0; row < depth.rows; row+=2)
    {
        for (int col = 0; col < depth.cols; col+=2)
        {
            short d = disparity.ptr<uchar>(row)[col];

            if (d < 0.1)
                depth.ptr<short>(row)[col] = 35.0;

            depth.ptr<short>(row)[col] = fx * baseline / d;

            double depth_num = depth.ptr<short>(row)[col];

            double z = double (depth_num) / camera_factor;
            double x = (col - camera_cx) * z / camera_fx;
            double y = (row - camera_cy) * z / camera_fy;

            // read bgr
//            double b = (color_image.ptr<short>(row)[col])[0];
//            double g = color_image.at<cv::Vec3b>(row, col)[2]/div*div + div/2;
//            double r = color_image.at<cv::Vec3b>(row, col)[3]/div*div + div/2;

            double b = color_image.at<cv::Vec4b>(row, col)[2]/div*div + div/2;
            double g = color_image.at<cv::Vec4b>(row, col)[1]/div*div + div/2;
            double r = color_image.at<cv::Vec4b>(row, col)[0]/div*div + div/2;

            Vector6d point(6);
            point << x, //X
                     y, //Y
                     z, //Z
                     b, //R
                     g, //G
                     r; //B
            pointcloud.push_back(point);//将点-颜色存储到容器中

        }
    }

    return pointcloud;
}

//vector<Vector6d, Eigen::aligned_allocator<Vector6d>> compute_pointcloud(cv::Mat &image, cv::Mat &color_image, int div=64)
//{
//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
//    int nr= image.rows; // number of rows
//    int nc= image.cols; // number of columns
//    for (int j=0; j<nr; j++) {
//        for (int i=0; i<nc; i++) {
//
//            double data = image.ptr<uchar>(nr)[nc];
//            cout << "data: " << data << endl;
//            if ( !(data > 0.3 && data < 30) )
//                continue;
//
////            double z = double (data) / camera_factor;
////            double x = (nc - camera_cx) * z / camera_fx;
////            double y = (nr - camera_cy) * z / camera_fy;
////
////            // read bgr
////            double b = color_image.at<cv::Vec3b>(j,i)[0]/div*div + div/2;
////            double g = color_image.at<cv::Vec3b>(j,i)[1]/div*div + div/2;
////            double r = color_image.at<cv::Vec3b>(j,i)[2]/div*div + div/2;
////
////            Vector6d point(6);
////            point << x, //X
////                     y, //Y
////                     z, //Z
////                     r, //R
////                     g, //G
////                     b; //B
////            pointcloud.push_back(point);//将点-颜色存储到容器中
//        }
//    }
//
////    return pointcloud;
//}

void showPointCloud( const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud ) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);//创建窗口并确定尺寸
    glEnable(GL_DEPTH_TEST);//3D可视化时开启，只绘制朝向摄像头一侧的图像
    glEnable(GL_BLEND);//启用颜色混合
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);//颜色混合方式
    //创建一个相机的观察视角
    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),//相机视角的尺寸，内参，最近和最远可视距离
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)//设置相机的外参：相机位置，相机朝向(俯仰，左右)，相机机轴方向(相机平面的旋转)——>（0.0, -1.0, 0.0）代表-y方向
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()//创建视图
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)//视图在视窗中的范围，以及视图的长宽比
            .SetHandler(new pangolin::Handler3D(s_cam));//显示s_cam所拍摄的内容

    while (pangolin::ShouldQuit() == false) {//若不关闭openGL窗口
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);//清空颜色和深度缓存，防止前后帧之间存在干扰

        d_cam.Activate(s_cam);//激活显示并设置状态矩阵
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);//刷新缓冲区颜色，防止帧间干扰
        glPointSize(0.5);//所绘点的大小
        /////// 真正的绘图部分 //////////
        glBegin(GL_POINTS);//点设置的开始
        for (auto &p: pointcloud) {//auto根据后面的变量值自行判断变量类型，继承点云
            glColor3d(p[3]/255.0, p[4]/255.0, p[5]/255.0);//RGB三分量相等即是灰度图像
            glVertex3d(p[0], p[1], p[2]);//确定点坐标
        }
        glEnd();//点设置的结束
        ///////////////////////////////
        pangolin::FinishFrame();//开始执行后期渲染，事件处理以及帧交换
        usleep(5000);   // sleep 5 ms
    }
    return;
}

int main() {

    Mat disparity_image = imread("/home/zed/Desktop/depth.jpg", -1);
    Mat origin_image = imread("/home/zed/Desktop/origin.png", -1);
    origin_image.resize((1390, 721));
    cout << "origin_image.channels(): " << origin_image.channels() << endl;
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
    pointcloud = compute_depth(disparity_image, origin_image);

    showPointCloud(pointcloud);




}