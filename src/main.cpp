/*=============================================================================
#  Author:           Nicolas Queiroga - https://github.com/NicolasQueiroga/
#  Email:            n.macielqueiroga@gmail.com
#  FileName:         main.cpp
#  Description:      Main file
#  Version:          0.0.1
=============================================================================*/

#include <iostream>
#include "aux.hpp"


class SudokuProc
{
protected:
    std::string path;
    cv::Mat bgr, mask, contour;
    cv::Mat *contour_ptr = &contour;

public:
    /**
     * @brief Construct a new Sudoku Proc object
     * 
     * @param path 
     */
    SudokuProc(std::string path)
    {
        this->path = path;
    }
    ~SudokuProc() {}

    /**
     * @brief 
     * 
     */
    void proc()
    {
        bgr = cv::imread(this->path);

        contour = bgr.clone();
        mask = getEdges(bgr, false);
        std::vector<cv::Point> maxAreaContour = getMaxAreaContour(mask, contour_ptr);

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

        int d = (xmax - xmin) / 9;
        // int c = (xmax - xmin)/18;
        for (int i = 0; i <= 9; i++)
            for (int j = 0; j <= 9; j++)
            {
                cv::drawMarker(bgr, cv::Point(xmin + i * d, ymin + j * d), cv::Scalar(0, 0, 255));
                // cv::drawMarker(bgr, cv::Point(xmin + c + i*d, ymin + c + j*d), cv::Scalar(255, 0, 0));
            }

        cv::imshow("BGR", bgr);
        cv::waitKey(0);
    }
};


std::string path = "../img/sudoku.png";
int main(int, char **)
{
    colorPicker();
    SudokuProc sp = SudokuProc(path);
    sp.proc();
    return 0;
}
