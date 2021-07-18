package com.example.imagepro;

import org.opencv.core.Mat;

public interface success
{
    void onEye(Mat mat);
    void onCoChanged(int x,int y);
}
