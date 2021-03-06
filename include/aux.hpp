/*=============================================================================
#  Author:           Nicolas Queiroga - https://github.com/NicolasQueiroga/
#  Email:            n.macielqueiroga@gmail.com
#  FileName:         aux.hpp
#  Description:      This file contais prototype info for aux.cpp
#  Version:          0.0.1
=============================================================================*/

#ifndef AUX_HPP
#define AUX_HPP

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>

/* Function Deffinitions */
std::vector<int> colorPicker(std::string path = "");
cv::Mat getMask(cv::Mat bgr, std::vector<int> hsvRanges, bool kernel = true);
cv::Mat getEdges(cv::Mat img, bool isMask = true);
void getAllContours(cv::Mat mask, std::vector<std::vector<cv::Point>> *contours, std::vector<cv::Vec4i> *hierarchy);
int getMaxAreaContourId(std::vector<std::vector<cv::Point>> contours);
std::vector<cv::Point> getMaxAreaContour(cv::Mat mask);
cv::Point getContourCenter(std::vector<cv::Point> contour);
void crossHair(cv::Mat img, cv::Point point, int size = 20, cv::Scalar color = {0, 0, 255});
std::vector<cv::Vec3f> findCircles(cv::Mat img, bool isMask = true);
std::vector<cv::Vec4i> findLines(cv::Mat img, bool isMask = true);
cv::Point getVanishingPoint(cv::Mat img, std::vector<cv::Vec4i> lines);
double getAngleWithVertical(double m);

#endif