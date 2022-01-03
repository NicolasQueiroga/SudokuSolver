#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#include "aux.hpp"

#include <iostream>
#include <sstream>

#define MIN_CONTOUR_AREA 100
#define RESIZED_IMAGE_WIDTH 20
#define RESIZED_IMAGE_HEIGHT 30

class ContourWithData
{
public:
    std::vector<cv::Point> ptContour;
    cv::Rect boundingRect;
    float fltArea;

    bool checkIfContourIsValid()
    {
        if (fltArea < MIN_CONTOUR_AREA)
            return false;
        return true;
    }

    static bool sortByBoundingRectXPosition(const ContourWithData &cwdLeft, const ContourWithData &cwdRight)
    {
        return (cwdLeft.boundingRect.x < cwdRight.boundingRect.x);
    }
};

int callme()
{
    std::vector<ContourWithData> allContoursWithData;
    std::vector<ContourWithData> validContoursWithData;
    cv::Mat matClassificationInts;
    cv::FileStorage fsClassifications("../classifications.xml", cv::FileStorage::READ);

    if (fsClassifications.isOpened())
    {
        std::cout << "error, unable to open training classifications file, exiting program\n\n";
        return 0;
    }

    fsClassifications["classifications"] >> matClassificationInts;
    fsClassifications.release();
    cv::Mat matTrainingImagesAsFlattenedFloats;
    cv::FileStorage fsTrainingImages("/home/nicolas/opencv_ws/SudokuSolver/model/images.xml", cv::FileStorage::READ);

    if (fsTrainingImages.isOpened() == false)
    {
        std::cout << "error, unable to open training images file, exiting program\n\n";
        return 0;
    }

    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;
    fsTrainingImages.release();

    cv::Ptr<cv::ml::KNearest> kNearest(cv::ml::KNearest::create());
    kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

    cv::Mat matTestingNumbers = cv::imread("../test/sudoku.png");
    cv::imshow("bgr", matTestingNumbers);
    if (matTestingNumbers.empty() == false)
    {
        std::cout << "error: image not read from file\n\n";
        return 0;
    }

    cv::Mat matGrayscale;
    cv::Mat matBlurred;
    cv::Mat matThresh;
    cv::Mat matThreshCopy;

    cv::cvtColor(matTestingNumbers, matGrayscale, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(matGrayscale, matBlurred, cv::Size(5, 5), 0);
    cv::adaptiveThreshold(matBlurred, matThresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

    matThreshCopy = matThresh.clone();

    std::vector<std::vector<cv::Point>> ptContours;
    std::vector<cv::Vec4i> v4iHierarchy;

    cv::findContours(matThreshCopy, ptContours, v4iHierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (long unsigned int i = 0; i < ptContours.size(); i++)
    {
        ContourWithData contourWithData;
        contourWithData.ptContour = ptContours[i];
        contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);
        contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);
        allContoursWithData.push_back(contourWithData);
    }

    for (long unsigned int i = 0; i < allContoursWithData.size(); i++)
        if (allContoursWithData[i].checkIfContourIsValid())
            validContoursWithData.push_back(allContoursWithData[i]);

    std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);
    std::string strFinalString;

    for (long unsigned int i = 0; i < validContoursWithData.size(); i++)
    {
        cv::rectangle(matTestingNumbers, validContoursWithData[i].boundingRect, cv::Scalar(0, 255, 0), 2);
        cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect);

        cv::Mat matROIResized;
        cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));

        cv::Mat matROIFloat;
        matROIResized.convertTo(matROIFloat, CV_32FC1);

        cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);
        cv::Mat matCurrentChar(0, 0, CV_32F);
        kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);

        float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);
        strFinalString = strFinalString + char(int(fltCurrentChar));
    }

    std::cout << "\n\n"
              << "numbers read = " << strFinalString << "\n\n";

    cv::imshow("matTestingNumbers", matTestingNumbers);
    cv::waitKey(0);
}


void processFrame(cv::Mat *input, cv::Mat *output)
{
    cv::Mat gray, thresh1;

    cv::cvtColor(*input, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, thresh1, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);

}

int main()
{
    std::string path = "/home/nicolas/opencv_ws/SudokuSolver/model/test/sudoku.png";
    cv::Mat bgr, gray, blurred, thresh, mask, frame;
    // cv::VideoCapture cap(0);
    // while (true)
    // {
    //     cap.read(frame);
    //     bgr = frame.clone();
    //     cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    //     cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    //     cv::adaptiveThreshold(blurred, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 7, 2.2);
    //     cv::imshow("Video", thresh);
    //     int ch = cv::waitKey(1);
    //     if (ch == 27)
    //     {
    //         break;
    //     }
    // }
    // cv::destroyAllWindows();

    colorPicker(path);

    // std::vector<int> hsv = colorPicker(path);
    // std::vector<std::vector<cv::Point>> contours = getAllContours(thresh);
    // cv::drawContours(bgr, contours, -1, cv::Scalar(255, 0, 0), 2);

    // cv::imshow("bgr", thresh);
    // cv::imshow("contours", bgr);
    // cv::waitKey(0);
    return 0;
}
