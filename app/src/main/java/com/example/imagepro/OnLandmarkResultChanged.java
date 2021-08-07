package com.example.imagepro;

import org.opencv.core.Mat;

public interface OnLandmarkResultChanged
{
    void onFaceDrawn(Mat mat);
    void onCoordinatesChanged(int x,int y);
}
