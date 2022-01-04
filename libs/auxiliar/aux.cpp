/*=============================================================================
#  Author:           Nicolas Queiroga - https://github.com/NicolasQueiroga/
#  Email:            n.macielqueiroga@gmail.com
#  FileName:         aux.cpp
#  Description:      file that contains auxiliary functions to use with opencv
#  Version:          0.0.1
=============================================================================*/

#include "aux.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <math.h>

std::vector<int> colorPicker(std::string path)
{
    cv::Mat bgr, hsv, mask;

    int hmin = 0, smin = 50, vmin = 50;
    int hmax = 255, smax = 255, vmax = 255;

    cv::namedWindow("Trackbars");
    cv::createTrackbar("Hue Min", "Trackbars", &hmin, 255);
    cv::createTrackbar("Hue Max", "Trackbars", &hmax, 255);
    cv::createTrackbar("Sat Min", "Trackbars", &smin, 255);
    cv::createTrackbar("Sat Max", "Trackbars", &smax, 255);
    cv::createTrackbar("Val Min", "Trackbars", &vmin, 255);
    cv::createTrackbar("Val Max", "Trackbars", &vmax, 255);

    if (path.empty())
    {
        cv::VideoCapture cap(0);
        while (true)
        {
            cap.read(bgr);
            std::vector<int> ranges = {hmin, smin, vmin, hmax, smax, vmax};
            mask = getMask(bgr, ranges);

            imshow("Image", bgr);
            imshow("Mask", mask);

            int ch = cv::waitKey(1);
            if (ch == 27)
                break;
        }
    }
    else
    {
        bgr = cv::imread(path);
        while (true)
        {
            std::vector<int> ranges = {hmin, smin, vmin, hmax, smax, vmax};
            mask = getMask(bgr, ranges);

            cv::imshow("Image", bgr);
            cv::imshow("Mask", mask);

            int ch = cv::waitKey(1);
            if (ch == 27)
                break;
        }
    }

    std::vector<int> ranges = {hmin, smin, vmin, hmax, smax, vmax};
    std::cout << "\nHSV min:\n"
              << ranges[0] << ", " << ranges[1] << ", " << ranges[2] << "\n";
    std::cout << "\nHSV max:\n"
              << ranges[3] << ", " << ranges[4] << ", " << ranges[5] << "\n\n";
    cv::destroyAllWindows();

    return ranges;
}

cv::Mat getMask(cv::Mat bgr, std::vector<int> hsvRanges, bool kernel)
{
    cv::Mat blurred, hsv, mask, morphMask;
    cv::GaussianBlur(bgr, blurred, cv::Size(5, 5), 0);
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

    int hmin = hsvRanges[0], hmax = hsvRanges[3];
    int smin = hsvRanges[1], smax = hsvRanges[4];
    int vmin = hsvRanges[2], vmax = hsvRanges[5];

    cv::Scalar lower(hmin, smin, vmin);
    cv::Scalar upper(hmax, smax, vmax);
    cv::inRange(hsv, lower, upper, mask);

    if (kernel)
    {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(6, 6));
        cv::morphologyEx(mask, morphMask, cv::MORPH_CLOSE, kernel);
        return morphMask;
    }

    return mask;
}

cv::Mat getEdges(cv::Mat img, bool isMask)
{
    cv::Mat gray, blur, edges, kernel, dilated;

    if (!isMask)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img;

    cv::GaussianBlur(gray, blur, cv::Size(7, 7), 5, 2);
    cv::Canny(blur, edges, 25, 100);
    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(edges, dilated, kernel);

    return dilated;
}

void getAllContours(cv::Mat mask, std::vector<std::vector<cv::Point>> *contours, std::vector<cv::Vec4i> *hierarchy)
{
    cv::findContours(mask, *contours, *hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
}

int getMaxAreaContourId(std::vector<std::vector<cv::Point>> contours)
{
    double maxArea = 0;
    int maxAreaContourId = 0;
    for (long unsigned int j = 0; j < contours.size(); j++)
    {
        double newArea = cv::contourArea(contours.at(j));
        if (newArea > maxArea)
        {
            maxArea = newArea;
            maxAreaContourId = j;
        }
    }

    return maxAreaContourId;
}

std::vector<cv::Point> getMaxAreaContour(cv::Mat mask)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<cv::Point> contour;
    getAllContours(mask, &contours, &hierarchy);

    int id = getMaxAreaContourId(contours);
    if (id != -1)
        contour = contours[id];

    return contour;
}

cv::Point getContourCenter(std::vector<cv::Point> contour)
{
    if (!contour.empty())
    {
        cv::Moments mu = cv::moments(contour);
        return cv::Point((int)mu.m10 / mu.m00, (int)mu.m01 / mu.m00);
    }
    return cv::Point(0, 0);
}

void crossHair(cv::Mat img, cv::Point point, int size, cv::Scalar color)
{
    line(img, cv::Point(point.x - size, point.y), cv::Point(point.x + size, point.y), color, 2);
    line(img, cv::Point(point.x, point.y - size), cv::Point(point.x, point.y + size), color, 2);
}

std::vector<cv::Vec3f> findCircles(cv::Mat img, bool isMask)
{
    cv::Mat edges;
    std::vector<cv::Vec3f> circles;
    edges = getEdges(img, isMask);
    cv::HoughCircles(edges, circles, cv::HOUGH_GRADIENT, 2.5, 40, 50, 100, 5, 150);

    return circles;
}

std::vector<cv::Vec4i> findLines(cv::Mat img, bool isMask)
{
    cv::Mat edges;
    std::vector<cv::Vec4i> linesP;
    edges = getEdges(img, isMask);
    cv::HoughLinesP(edges, linesP, 1, CV_PI / 180, 50, 50, 10);

    return linesP;
}

cv::Point getVanishingPoint(cv::Mat img, std::vector<cv::Vec4i> lines)
{
    std::vector<std::vector<cv::Point>> coordenates;

    bool isLeft = false;
    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::Vec4i l = lines[i];
        cv::Point p1(l[0], l[1]), p2(l[2], l[3]);
        double grad = (p2.y - p1.y) / ((double)(p2.x - p1.x));
        if (grad < 0 && !isLeft)
        {
            std::vector<cv::Point> p;
            p.push_back(p1);
            p.push_back(p2);
            coordenates.push_back(p);
            isLeft = true;
            cv::line(img, p1, p2, cv::Scalar(155, 100, 0), 3);
        }
        else if (grad >= 0)
        {
            std::vector<cv::Point> p;
            p.push_back(p1);
            p.push_back(p2);
            coordenates.push_back(p);
            cv::line(img, p1, p2, cv::Scalar(155, 100, 0), 3);
            break;
        }
    }

    std::vector<std::vector<double>> params;
    for (long unsigned int i = 0; i < coordenates.size(); i++)
    {
        cv::Point p1(coordenates[i][0]), p2(coordenates[i][1]);
        double m = (p2.y - p1.y) / ((double)(p2.x - p1.x));
        double h = p2.y - m * p2.x;
        std::vector<double> line = {m, h};
        params.push_back(line);
    }

    double m1 = params[0][0];
    double h1 = params[0][1];
    double m2 = params[1][0];
    double h2 = params[1][1];
    int px = (h2 - h1) / (m1 - m2);
    int py = m1 * px + h1;
    cv::Point p(px, py);
    cv::circle(img, p, 8, cv::Scalar(255, 0, 0), -1);

    return p;
}

double getAngleWithVertical(double m)
{
    double rads = CV_PI / 2 + atan(m);
    return rads /**180/CV_PI*/;
}
