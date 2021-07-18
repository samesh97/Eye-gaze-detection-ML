package com.example.imagepro;


import org.opencv.core.Rect;

public class Model
{
    private Rect face_roi = null;
    private Rect eye_roi = null;

    public Rect getFace_roi() {
        return face_roi;
    }

    public void setFace_roi(Rect face_roi) {
        this.face_roi = face_roi;
    }

    public Rect getEye_roi() {
        return eye_roi;
    }

    public void setEye_roi(Rect eye_roi) {
        this.eye_roi = eye_roi;
    }
}
