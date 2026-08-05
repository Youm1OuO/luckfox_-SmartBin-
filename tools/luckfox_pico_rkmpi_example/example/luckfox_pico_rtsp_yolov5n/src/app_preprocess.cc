// ============================================================================
//  app_preprocess.cc
//  Main-loop image preprocessing and coordinate conversion
// ============================================================================
#include "app_preprocess.h"

#include "opencv2/imgproc/imgproc.hpp"

cv::Mat letterbox(cv::Mat input)
{
	float scaleX = (float)model_width  / (float)width;
	float scaleY = (float)model_height / (float)height;
	scale = scaleX < scaleY ? scaleX : scaleY;

	int inputWidth   = (int)((float)width * scale);
	int inputHeight  = (int)((float)height * scale);

	leftPadding = (model_width  - inputWidth) / 2;
	topPadding  = (model_height - inputHeight) / 2;

	cv::Mat inputScale;
    cv::resize(input, inputScale, cv::Size(inputWidth,inputHeight), 0, 0, cv::INTER_LINEAR);
	cv::Mat letterboxImage(model_height, model_width, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Rect roi(leftPadding, topPadding, inputWidth, inputHeight);
    inputScale.copyTo(letterboxImage(roi));

	return letterboxImage;
}

void mapCoordinates(int *x, int *y) {
	int mx = *x - leftPadding;
	int my = *y - topPadding;

	int rx = (int)((float)mx / scale);
	int ry = (int)((float)my / scale);

	if (rx < 0)      rx = 0;
	if (ry < 0)      ry = 0;
	if (rx > width)  rx = width;
	if (ry > height) ry = height;

	*x = rx;
	*y = ry;
}

