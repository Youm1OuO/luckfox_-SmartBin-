// ============================================================================
//  app_preprocess.h
//  Main-loop image preprocessing and model-to-frame coordinate helpers
// ============================================================================
#ifndef __FRIDGE_APP_PREPROCESS_H
#define __FRIDGE_APP_PREPROCESS_H

#include "opencv2/core/core.hpp"

// These dimensions and the last letterbox transform remain owned by main.cc.
extern int width;
extern int height;
extern int model_width;
extern int model_height;
extern float scale;
extern int leftPadding;
extern int topPadding;

cv::Mat letterbox(cv::Mat input);
void mapCoordinates(int* x, int* y);

#endif  // __FRIDGE_APP_PREPROCESS_H
