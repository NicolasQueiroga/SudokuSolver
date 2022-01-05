/*=============================================================================
#  Author:           Nicolas Queiroga - https://github.com/NicolasQueiroga/
#  Email:            n.macielqueiroga@gmail.com
#  FileName:         main.cpp
#  Description:      Main file
#  Version:          0.0.1
=============================================================================*/

#include <iostream>
#include "aux.hpp"
#include <opencv2/ml.hpp>

class SudokuProc
{
protected:
    cv::Ptr<cv::ml::KNearest> kNearest;
    cv::Mat matClassificationInts;
    cv::Mat matTrainingImagesAsFlattenedFloats;
    cv::Mat bgr;
    cv::Mat dilated;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    std::vector<std::vector<cv::Point>> numbers;

    std::vector<cv::Point> maxAreaContour;
    double boxArea;
    int board[9][9];

public:
    /**
     * @brief Construct a new Sudoku Proc object
     * 
     * @param path 
     */
    SudokuProc(std::string path)
    {
        this->bgr = cv::imread(path);
        this->kNearest = cv::ml::KNearest::create();
    }
    ~SudokuProc() {}

    void loadModel()
    {
        cv::FileStorage fsClassifications("../model/classifications.xml", cv::FileStorage::READ);
        fsClassifications["classifications"] >> this->matClassificationInts;
        fsClassifications.release();

        cv::FileStorage fsTrainingImages("../model/images.xml", cv::FileStorage::READ);
        fsTrainingImages["images"] >> this->matTrainingImagesAsFlattenedFloats;
        fsTrainingImages.release();

        this->kNearest->train(this->matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);
    }

    static bool sortByBoundingRectXPosition(const std::vector<cv::Point> &cwdLeft, const std::vector<cv::Point> &cwdRight)
    {
        cv::Rect a, b;
        a = cv::boundingRect(cwdLeft);
        b = cv::boundingRect(cwdRight);
        return /*(a.x < b.x) && */ (a.y < b.y);
    }

    void preProcessFrame()
    {
        cv::Mat gray, blur, thresh, kernel;
        cv::resize(this->bgr, this->bgr, cv::Size(), 0.5, 0.5);
        cv::cvtColor(this->bgr, gray, cv::COLOR_BGR2GRAY);
        // cv::GaussianBlur(gray, blur, cv::Size(1, 1), 3, 2);
        cv::threshold(gray, thresh, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);
        kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
        cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel);
        cv::dilate(thresh, this->dilated, kernel);
        getAllContours(this->dilated, &this->contours, &this->hierarchy);
        this->maxAreaContour = getMaxAreaContour(this->dilated);
    }

    void processFrame()
    {
        int xmin = 1000, ymin = 1000, xmax = 0, ymax = 0;
        for (cv::Point p : maxAreaContour)
        {
            if (p.x < xmin)
                xmin = p.x;
            else if (p.x > xmax)
                xmax = p.x;
            if (p.y > ymax)
                ymax = p.y;
            else if (p.y < ymin)
                ymin = p.y;
        }

        double d = (xmax - xmin) / 9;
        double center = d / 2;

        this->boxArea = d * d;

        // Mark all numbers
        std::vector<cv::Point> numberCenters;
        for (long unsigned int i = 0; i < this->contours.size(); i++)
        {
            double contourArea = cv::contourArea(contours[i]);
            cv::Point center = getContourCenter(contours[i]);
            if ((contourArea < (2 * this->boxArea / 3)) && (this->hierarchy[i][3] == -1) && ((center.x >= d/2) && (center.y >= d/2)))
            {
                this->numbers.push_back(contours[i]);
                numberCenters.push_back(center);
            }
        }
        cv::drawContours(this->bgr, numbers, -1, cv::Scalar(255, 0, 255), 2); // number contours
        // std::sort(numbers.begin(), numbers.end(), this->sortByBoundingRectXPosition);
        for (cv::Point p : numberCenters)
            cv::drawMarker(this->bgr, p, cv::Scalar(193, 0, 255));
        // ---

        // Mark all free spaces
        unsigned int count = numbers.size() - 1;
        for (int i = 0; i < 9; i++)
            for (int j = 0; j < 9; j++)
            {
                cv::Mat copy = this->bgr.clone();
                cv::Rect box(j * d + xmin, i * d + ymin, d, d);
                cv::Mat cropped = copy(box);
                cv::rectangle(copy, box, cv::Scalar(0, 255, 255), 2);

                bool contains = 1;
                for (cv::Point p : numberCenters)
                    contains = contains && !p.inside(box);
                if (contains)
                {
                    cv::drawMarker(this->bgr, cv::Point(xmin + center + j * d, ymin + center + i * d), cv::Scalar(255, 255, 0));
                    this->board[j][i] = 0;
                }
                else
                {
                    int detected = this->getNumbers(numbers[count]);
                    this->board[j][i] = detected;
                    count--;
                }
                // cv::imshow("sqrs", copy);
                // cv::waitKey(0);
            }
        // ---

        std::cout << "\n\n";
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
                std::cout << " " << this->board[j][i];
            std::cout << "\n";
        }

        cv::imshow("dilated", this->dilated);
        cv::imshow("BGR", this->bgr);
        cv::waitKey(0);
    }

    int getNumbers(std::vector<cv::Point> number)
    {
        std::string strFinalString;

        cv::Rect numberBox = cv::boundingRect(number);
        cv::rectangle(this->bgr, numberBox, cv::Scalar(255, 0, 123));
        cv::Mat ROI = this->dilated(numberBox);

        cv::Mat ROIResized;
        cv::resize(ROI, ROIResized, cv::Size(20, 30));

        cv::Mat ROIFloat;
        ROIResized.convertTo(ROIFloat, CV_32FC1);

        cv::Mat ROIFlattenedFloat = ROIFloat.reshape(1, 1);
        cv::Mat CurrentChar(0, 0, CV_32F);
        this->kNearest->findNearest(ROIFlattenedFloat, 1, CurrentChar);

        int fltCurrentChar = (int)CurrentChar.at<float>(0, 0);
        strFinalString = strFinalString + char(int(fltCurrentChar));
        std::cout << "info read = " << strFinalString << "\n";
        return (fltCurrentChar - '0');
    }
};

std::string path = "../img/sudoku.png";
int main(int, char **)
{
    SudokuProc sp = SudokuProc(path);
    sp.loadModel();
    sp.preProcessFrame();
    sp.processFrame();
    return 0;
}
