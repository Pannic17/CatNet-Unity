using System;
using System.IO;
using UnityEngine;
using Unity.Barracuda;


using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.ObjdetectModule;
using OpenCVForUnity.Img_hashModule;

using CVRect = OpenCVForUnity.CoreModule.Rect;


/// <summary>
/// Static class script for cv processing
/// </summary>
public static class CVProcessor
{
    private const int SIZE = 224;
    public enum Mode
    {
        HSV,
        RBG,
        BGR,
    }


    // string[] labels = { "Bicolor", "Calico", "Colorpoint", "Mix", "Orange", "Solid", "Tabby" };

    /// <summary> 
    /// Detect the cat face using opencv haarcascade 
    /// </summary>
    /// <param name="input"> the orginal mat of input webcam or file </param>
    /// <param name="cascade"> the used cascade file </param>
    /// <returns> the detected boxes of cat face </returns>
    public static CVRect[] DetectCatFace(Mat input, CascadeClassifier cascade)
    {
        MatOfRect detected = new MatOfRect();
        Mat gray = new Mat(input.size(), CvType.CV_8UC1);
        Imgproc.cvtColor(input, gray, Imgproc.COLOR_RGBA2GRAY);
        cascade.detectMultiScale(gray, detected, 1.02, 5);


        CVRect[] cat_box = detected.toArray();
        return cat_box;
    }

    /// <summary>
    /// crop the selected area of index from original image and enlarge it by 120%
    /// </summary>
    /// <param name="area"> the detected boxes of cat face </param>
    /// <param name="original"> the origin image </param>
    /// <param name="index"> the rank index of detected cat face </param>
    /// <returns> the mat of cropped cat face </returns>
    public static Mat CropCatFace(CVRect[] area, Mat original, int index)
    {
        double delta = Math.Ceiling((double)area[index].width * 0.1);
        CVRect enlarged = new CVRect((int)(area[index].x - delta),
                                          (int)(area[index].y - delta * 2),
                                          (int)(area[index].width * 1.2),
                                          (int)(area[index].height * 1.2));
        return new Mat(original, enlarged);
    }

    /// <summary>
    /// resize the originally captured cat face to normalized 224*224 for neuro network
    /// </summary>
    /// <param name="original"> the orignal captured cat face texture </param>
    /// <returns> the normalized cat face in size of 224*224 </returns>
    public static Texture2D Normalize2Tensor(Texture2D original)
    {
        Texture2D normalized = new Texture2D(SIZE, SIZE);
        original.filterMode = FilterMode.Point;
        RenderTexture rt = RenderTexture.GetTemporary(SIZE, SIZE);
        rt.filterMode = FilterMode.Point;
        RenderTexture.active = rt;
        Graphics.Blit(original, rt);
        normalized.ReadPixels(new UnityEngine.Rect(0, 0, SIZE, SIZE), 0, 0);
        normalized.Apply();
        RenderTexture.active = null;
        return normalized;
    }

    /// <summary>
    /// used for debug, log all the classification result
    /// </summary>
    /// <param name="labels"></param>
    /// <param name="result"></param>
    private static void DebugResult(string[] labels, float[] result)
    {
        for (int i = 0; i < result.Length; i++)
        {
            Debug.Log(labels[i] + ":" + result[i]);
        }
    }

    /// <summary>
    /// use k-means algorithm to calculate the major color of the image
    /// </summary>
    /// <param name="input"> the mat of image </param>
    /// <param name="cluster"> the number of major color to get </param>
    /// <param name="mode"> the mode of color pattern </param>
    /// <returns> return list of major colors </returns>
    public static Color[] ColorKMEANS(Mat input, int cluster, Mode mode)
    {
        Mat pixels = Mat2Vector(input, 3, mode);
        Mat labels = new Mat();
        Mat center = new Mat(cluster, 1, pixels.type());
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, 10, 0.1);
        Core.kmeans(pixels, cluster, labels, criteria, 10, Core.KMEANS_PP_CENTERS, center);
        Mat result = GetCenterResult(center, 3, mode);
        Debug.Log(result.dump());
        Color[] colors = Mat2Color(result, cluster);
        return colors;
    }

    /// <summary>
    /// convert the mat to color
    /// </summary>
    /// <param name="result"> result of kmeans in mat form</param>
    /// <param name="cluster"> the number of major color to get </param>
    /// <returns> the center color </returns>
    private static Color[] Mat2Color(Mat result, int cluster)
    {
        Color[] colors = new Color[cluster];
        for (int i = 0; i < cluster; i++)
        {
            byte r = (byte)result.get(i, 0)[0];
            byte g = (byte)result.get(i, 0)[1];
            byte b = (byte)result.get(i, 0)[2];
            Color color = new Color32(r, g, b, 255);
            colors[i] = color;
            Debug.Log(color.ToString());
        }
        return colors;
    }

    /// <summary>
    /// get the centers value of the result in RGB by the input color mode
    /// </summary>
    /// <param name="input"> the mat of image </param>
    /// <param name="cluster"> the number of major color to get </param>
    /// <param name="mode"> the mode of color pattern </param>
    /// <returns> the mat of result </returns>
    private static Mat GetCenterResult(Mat center, int cluster, Mode mode)
    {
        Mat colors = new Mat(cluster, 1, CvType.CV_8UC3);
        Mat result = new Mat();
        for (int i = 0; i < cluster; i++)
        {
            double[] color = { center.get(i, 0)[0], center.get(i, 1)[0], center.get(i, 2)[0] };
            colors.put(i, 0, color);
        }

        if (mode == Mode.RBG)
        {
            result = colors.clone();
        }
        if (mode == Mode.HSV)
        {
            Imgproc.cvtColor(colors, result, Imgproc.COLOR_HSV2RGB);
        }
        if (mode == Mode.BGR)
        {
            Imgproc.cvtColor(colors, result, Imgproc.COLOR_BGR2RGB);
        }

        return result;
    }

    /// <summary>
    /// Convert the input image to vector mat for kmeans
    /// </summary>
    /// <param name="input"> the mat of image </param>
    /// <param name="cluster"> the number of major color to get </param>
    /// <param name="mode"> the mode of color pattern </param>
    /// <returns> the vector mat for kmeans </returns>
    private static Mat Mat2Vector(Mat input, int cluster, Mode mode)
    {

        int width = input.cols();
        int height = input.rows();
        int count = width * height;

        Mat img = new Mat(height, width, CvType.CV_8UC3);
        if (mode == Mode.RBG)
        {
            img = input.clone();
        }
        if (mode == Mode.HSV)
        {
            Imgproc.cvtColor(input, img, Imgproc.COLOR_RGB2HSV);
        }
        if (mode == Mode.BGR)
        {
            Imgproc.cvtColor(input, img, Imgproc.COLOR_RGB2BGR);
        }

        Mat pixels = new Mat(count, cluster, CvType.CV_32F, new Scalar(10));


        int index = 0;
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                index = row * width + col;
                double[] pixel = img.get(row, col);
                pixels.put(index, 0, pixel[0]);
                pixels.put(index, 1, pixel[1]);
                pixels.put(index, 2, pixel[2]);
            }
        }

        return pixels;
    }

    /// <summary>
    /// use barracuda tensor to predict current image
    /// </summary>
    /// <param name="input"> the input cropped cat face </param>
    /// <param name="worker"> the barracuda worker </param>
    /// <param name="labels"> labels of predict result </param>
    /// <returns></returns>
    public static string PredictNeuroNetwork(Texture2D input, IWorker worker, string[] labels)
    {
        // initial and classify
        Tensor inputTensor = CVProcessor.NormalizePixels(input);
        worker.Execute(inputTensor);
        Tensor outputTensor = worker.PeekOutput("softmax");

        // get model results
        var result = outputTensor.ToReadOnlyArray();
        var max = Mathf.Max(result);
        var index = Array.IndexOf(result, max);

        // dispose tensor
        inputTensor.Dispose();
        outputTensor.Dispose();
        Debug.Log("Result: " + labels[index] + "\nProb: " + max);

        return labels[index];
    }

    /// <summary>
    /// Save the input Texture2D to jpg
    /// </summary>
    /// <param name="input"> the input Texture2D file </param>
    public static void Save2JPG(Texture2D input)
    {
        byte[] pngData = input.EncodeToJPG();
        File.WriteAllBytes(Application.dataPath + "/Cat.jpg", pngData);
    }

    /// <summary>
    /// Normalize the pixel value to -1~1 and initialze the tensor for prediction
    /// </summary>
    /// <param name="input"> the image for prediction </param>
    /// <returns> return the input tensor </returns>
    private static Tensor NormalizePixels(Texture2D input)
    {
        float[] channels = new float[SIZE * SIZE * 3];
        int i = 0;
        for (int y = SIZE - 1; y >= 0; y--)
        {
            for (int x = 0; x < SIZE; x++)
            {
                Color pixel = input.GetPixel(x, y);
                channels[i * 3 + 0] = (pixel.r - 0.5f) * 2;
                channels[i * 3 + 1] = (pixel.g - 0.5f) * 2;
                channels[i * 3 + 2] = (pixel.b - 0.5f) * 2;
                i++;
            }
        }
        return new Tensor(1, SIZE, SIZE, 3, channels);
    }

    /// <summary>
    /// Calculate the image hash used as genotype
    /// </summary>
    /// <param name="input"></param>
    public static void CalculateImageHash(Mat input)
    {
        MarrHildrethHash function = MarrHildrethHash.create();
        Mat imageHash = new Mat();
        function.compute(input, imageHash);
    }
}
