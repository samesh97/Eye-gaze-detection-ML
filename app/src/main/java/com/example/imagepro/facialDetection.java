package com.example.imagepro;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class facialDetection {

    // define Interpreter
    private final Interpreter interpreter;
    // now define input size and pixel size
    private final int INPUT_SIZE;
    private GpuDelegate gpuDelegate;

    private CascadeClassifier cascadeClassifier;

    private final success s;

    private int dHeight,dWidth = 0;

    // on start
    facialDetection(AssetManager assetManager, Context context, String modelPath, int inputSize,int height,int width,success s) throws IOException{
        INPUT_SIZE=inputSize;

        this.dHeight = height;
        this.dWidth = width;
        this.s = s;

        // define GPU and number of thread to Interpreter
        Interpreter.Options options=new Interpreter.Options();
        gpuDelegate=new GpuDelegate();
        options.addDelegate(gpuDelegate);
        options.setNumThreads(4); // change number of thread according to your phone

        // load CNN model
        interpreter=new Interpreter(loadModelFile(assetManager,modelPath),options);
        Log.d("FacialDetector","CNN model is loaded");
        // Now load haar cascade classifier
        try{
            // define input stream
            InputStream is=context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            // define folder path
            File cascadeDir=context.getDir("cascade",Context.MODE_PRIVATE);
            File mCascadeFile=new File(cascadeDir,"haarcascade_frontalface_alt.xml");
            // define output stream
            FileOutputStream os=new FileOutputStream(mCascadeFile);

            // copy classifier to that folder
            byte[] buffer =new byte[4096];
            int byteRead;
            while ((byteRead=is.read(buffer)) !=-1){
                os.write(buffer,0,byteRead);
            }
            // close input and output stream
            is.close();
            os.close();
            // define cascade classifier
            cascadeClassifier=new CascadeClassifier(mCascadeFile.getAbsolutePath());

            Log.d("FacialDetector","Classifier is loaded");


            // Before watching this video please watch my previous video :
            //Facial Landmark Detection Android App Using TFLite(GPU) and OpenCV: Load CNN Model Part 2
            // You will end up with this code

            // In this video, we will do two things:
            // 1. Detect face on frame
            // 2. Pass cropped face to Interpreter which will give x, y co-ordinate of 15 keypoints on face
            // Let's start

        }
        catch (IOException e){
            e.printStackTrace();
        }

    }

    // Creata a new function input as Mat and output is also Mat format
    public Mat recognizeImage(Mat mat_image)
    {

        // mat_image is not properly align it is 90 degree off
        // rotate mat_image by 90 degree
        Mat a = mat_image.t();
        Core.flip(a,mat_image,1);
        a.release();

        // do all process here
        // face detection
        // Convert mat_image to grayscale image
        Mat grayscaleImage=new Mat();
        Imgproc.cvtColor(mat_image,grayscaleImage,Imgproc.COLOR_RGBA2GRAY);
        // define height, width of grayscaleImage
        int height =grayscaleImage.height();
        int width=grayscaleImage.width();

        // define minimum height of face in original frame below this height no face will detected
        int absoluteFaceSize=(int) (height*0.1); // you can change this number to get better result

        // check if cascadeClassifier is loaded or not
        // define MatOfRect of faces
        MatOfRect faces=new MatOfRect();

        if(cascadeClassifier !=null){
            // detect face                        input       output
            cascadeClassifier.detectMultiScale(grayscaleImage,faces,1.1,2,2,
                    new Size(absoluteFaceSize,absoluteFaceSize),new Size());
                //      minimum size
        }

        // create faceArray
        Rect[] faceArray=faces.toArray();
        // loop through each face in faceArray

        for(int i=0;i<faceArray.length;i++)
        {
           int x1 = (int) faceArray[i].tl().x;
           int y1 = (int) faceArray[i].tl().y;

           int x2 = (int) faceArray[i].br().x;
           int y2 = (int) faceArray[i].br().y;

           int pp = 60;

           if(x1 - pp >= 0)
           {
               x1 -= pp;
           }
            if(y1 - pp >= 0)
            {
                y1 -= pp;
            }

            if(x2 + pp <= width)
            {
                x2 += pp;
            }
            if(y2 + pp <= height)
            {
                y2 += pp;
            }

            int w1 = x2 - x1;
            int h1 = y2 - y1;

            Rect face_roi = new Rect(x1,y1,w1,h1);

            Mat cropped_rgba = new Mat(mat_image,face_roi);



            Size sz = new Size(INPUT_SIZE,INPUT_SIZE);
            Mat resizeImage = new Mat();
            Imgproc.resize(cropped_rgba,resizeImage,sz,0,0,Imgproc.INTER_CUBIC);

            Bitmap bitmap = null;
            bitmap = Bitmap.createBitmap(cropped_rgba.cols(),cropped_rgba.rows(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(cropped_rgba,bitmap);

            int c_height = bitmap.getHeight();
            int c_width = bitmap.getWidth();

            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);


            float[][] result = new float[1][136];
            interpreter.run(byteBuffer,result);

            int eyeStartingX = 0,eyeEndingX = 0;
            int eyeStartingY = 0,eyeEndingY = 0;

            for(int j = 0; j < 136; j = j + 2)
            {
                if(j == 72)
                {
                    float x_val = (float) Array.get(Array.get(result,0),j);
                    float y_val = (float) Array.get(Array.get(result,0),j + 1);

                    eyeStartingX = (int) x_val;

                    Imgproc.line(resizeImage,new Point(x_val - 2,y_val - 10),new Point(x_val - 2,y_val + 10),new Scalar(255,0,0,255),1);

                   // Imgproc.circle(resizeImage,new Point(x_val,y_val),1, new Scalar(0,255,0,255), -1);
                }
                if(j == 78)
                {
                    float x_val = (float) Array.get(Array.get(result,0),j);
                    float y_val = (float) Array.get(Array.get(result,0),j + 1);

                    eyeEndingX = (int) x_val;

                    Imgproc.line(resizeImage,new Point(x_val,y_val - 10),new Point(x_val,y_val + 10),new Scalar(255,0,0,255),1);

                   // Imgproc.circle(resizeImage,new Point(x_val,y_val),1, new Scalar(0,255,0,255), -1);
                }
                if(j == 76)
                {
                    float x_val = (float) Array.get(Array.get(result,0),j);
                    float y_val = (float) Array.get(Array.get(result,0),j + 1);

                    eyeStartingY = (int) y_val;

                    //Imgproc.line(resizeImage,new Point(x_val - 10,y_val),new Point(x_val + 10,y_val),new Scalar(255,0,0,255),1);

                    //Imgproc.circle(resizeImage,new Point(x_val,y_val),1, new Scalar(0,255,0,255), -1);
                }
                if(j == 82)
                {
                    float x_val = (float) Array.get(Array.get(result,0),j);
                    float y_val = (float) Array.get(Array.get(result,0),j + 1);

                    eyeEndingY = (int) y_val;

                    //Imgproc.line(resizeImage,new Point(x_val - 10,y_val),new Point(x_val + 10,y_val),new Scalar(255,0,0,255),1);

                   // Imgproc.line(resizeImage,new Point(),new Point(),new Scalar(255,0,0,255),5);

                    //Imgproc.circle(resizeImage,new Point(x_val,y_val),1, new Scalar(0,255,0,255), -1);
                }

            }


            try
            {
                eyeStartingX -= 4;
                eyeEndingX += 4;
                eyeStartingY -= 4;
                eyeEndingY += 8;
                Rect eye_roi = new Rect(eyeStartingX,eyeStartingY,eyeEndingX - eyeStartingX,eyeEndingY - eyeStartingY);

                Mat eye = new Mat(resizeImage,eye_roi);

                Mat gray = new Mat();
                Imgproc.cvtColor(eye,gray,Imgproc.COLOR_BGR2GRAY);

                Imgproc.equalizeHist(gray,gray);
                Imgproc.blur(gray,gray,new Size(2,2));


                Mat binary = new Mat();
                Imgproc.threshold(gray, binary,45, 255, Imgproc.THRESH_BINARY);

                List<MatOfPoint> contours = new ArrayList<>();
                Mat hierarchy = new Mat();
                Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);



                if(contours.size() >= 1)
                {
                    contours.remove(0);
                }
                // now iterate over all top level contours

                double largestContuorArea = 0;
                int position = 0;

                for ( int contourIdx = 0; contourIdx < contours.size(); contourIdx++ )
                {

                    double contourArea = Imgproc.contourArea(contours.get(contourIdx));
                    if (largestContuorArea < contourArea)
                    {
                        largestContuorArea = contourArea;
                        position = contourIdx;
                    }
                }


                if(contours.size() > position)
                {
                    //draw contour
                    //Imgproc.drawContours ( eye, contours, position, new Scalar(0, 255, 0), 1);

                    //draw circle on middle point
                    Moments p = Imgproc.moments(contours.get(position));

                    int x = (int) (p.get_m10() / p.get_m00());
                    int y = (int) (p.get_m01() / p.get_m00());
                    //Imgproc.circle(eye, new Point(x, y), 4, new Scalar(255,49,0,255));


                    findViewPoint(x,y,eyeStartingX,eyeEndingX,eyeStartingY,eyeEndingY,binary);

                    //draw horizontal and vertical lines to create a cross
                    Imgproc.line(eye,new Point(x - 5,y),new Point(x + 5,y),new Scalar(0, 255, 0));
                    Imgproc.line(eye,new Point(x,y - 5),new Point(x,y + 5),new Scalar(0, 255, 0));

                    //draw circle in the gray frame
                    //Imgproc.circle(binary, new Point(x, y), 3, new Scalar(0,0,0,255),3);
                }



                Imgproc.cvtColor(eye,eye,Imgproc.THRESH_BINARY);
                s.onEye(binary);
            }
            catch (Exception e){}



            sz = new Size(c_width,c_height);
            Imgproc.resize(resizeImage,cropped_rgba,sz,0,0,Imgproc.INTER_CUBIC);
            cropped_rgba.copyTo(new Mat(mat_image,face_roi));


        }


        // but returned mat_image should be same as passing mat
        // rotate back it -90 degree
        Mat b = mat_image.t();
        Core.flip(b,mat_image,0);
        b.release();
        return mat_image;
    }

    private void findViewPoint(int x, int y,int sX,int eX,int sY,int eY,Mat mat)
    {


        int range = eX - sX;
        int equalXPixels = dWidth / range;

        if(x <= 9)
        {
            x = (range / 4) * 4;
        }
        else if(x == 10)
        {
            x = (range / 4) * 3;
        }
        else if(x == 11)
        {
            x = (range / 4) * 2;
        }
        else if(x == 12)
        {
            x = (range / 4) * 1;
        }
        else
        {
            x = (range / 4) * 0;
        }


        int calcX = x * equalXPixels;


        int rangeHeight = eY - sY;
        int equalYPixels = dHeight / rangeHeight;

        if(y <= 5)
        {
            y = (rangeHeight / 4) * 0;
        }
        else if(y == 6)
        {
            y = (rangeHeight / 4) * 1;
        }
        else if(y == 7)
        {
            y = (rangeHeight / 4) * 2;
        }
        else if(y == 8)
        {
            y = (rangeHeight / 4) * 3;
        }
        else if(y >= 8)
        {
            y = (rangeHeight / 4) * 4;
        }

        int calcY = y * equalYPixels;






        s.onCoChanged(calcX,calcY);

        Log.d("fsffsefsess","Y - " + y + " Range - " + (eY - sY));

    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap scaledBitmap) {
        ByteBuffer byteBuffer;
        int inputSize=INPUT_SIZE;// 96

        int quant = 1;
        if(quant == 0)
        {
            byteBuffer = ByteBuffer.allocateDirect(3 * 1 * inputSize * inputSize);
        }
        else
        {
            byteBuffer = ByteBuffer.allocateDirect(4 * 1 * inputSize * inputSize * 3);
        }

        byteBuffer.order(ByteOrder.nativeOrder());
        int pixel=0;
        int [] intValues=new int [inputSize*inputSize];
        scaledBitmap.getPixels(intValues,0,scaledBitmap.getWidth(),0,0,scaledBitmap.getWidth(),scaledBitmap.getHeight());

        for (int i=0;i<inputSize;++i){
            for(int j=0;j<inputSize;++j){
                final int val= intValues[pixel++];

                byteBuffer.putFloat((((val >> 16) & 0xFF))/255.0f);
                byteBuffer.putFloat((((val >> 8) & 0xFF))/255.0f);
                byteBuffer.putFloat(((val & 0xFF))/255.0f);

            }
        }
        return  byteBuffer;

    }
    // now call this function in CameraActivity

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException{
        // description of file
        AssetFileDescriptor assetFileDescriptor=assetManager.openFd(modelPath);
        FileInputStream inputStream=new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=assetFileDescriptor.getStartOffset();
        long declaredLength=assetFileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }

}
