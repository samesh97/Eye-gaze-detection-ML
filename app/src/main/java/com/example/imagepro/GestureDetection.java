package com.example.imagepro;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class GestureDetection
{
    private Interpreter interpreter;
    private GpuDelegate gpuDelegate;
    private final int INPUT_SIZE;

    public GestureDetection(AssetManager assetManager, Context context, String modelPath, int inputSize) throws IOException
    {
        INPUT_SIZE = inputSize;

//        Interpreter.Options options = new Interpreter.Options();
//        gpuDelegate = new GpuDelegate();
//        options.addDelegate(gpuDelegate);
//        options.setNumThreads(4);


        interpreter = new Interpreter(loadModelFile(assetManager,modelPath));
        Log.d("sssssssssssssss","Model Loaded");

    }
    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException
    {
        // description of file
        AssetFileDescriptor assetFileDescriptor=assetManager.openFd(modelPath);
        FileInputStream inputStream=new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=assetFileDescriptor.getStartOffset();
        long declaredLength=assetFileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
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
    public String detectGestures(Bitmap bitmap)
    {
        String output = "Idle";

        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);


        float[][] result = new float[1][3];
        interpreter.run(byteBuffer,result);

        for(int i = 0; i < result.length; i++)
        {
            float idle = (float) Array.get(Array.get(result,0),i);
            float left = (float) Array.get(Array.get(result,0),i+1);
            float right = (float) Array.get(Array.get(result,0),i+2);

//            Log.d("TestResults","Idle" + idle);
//            Log.d("TestResults","Left" + left);
//            Log.d("TestResults","Right" + right);


            if(idle > left)
            {
                if(idle > right)
                {
                    output = "Idle";
                    //idle
                }
                else
                {
                    output = "Right";
                    //right
                }
            }
            else if(right > left)
            {
                output = "Right";
                //right
            }
            else
            {
                output = "Left";
                //left
            }

        }

        return output;

    }
}
