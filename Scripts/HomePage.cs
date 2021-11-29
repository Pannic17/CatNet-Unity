#if !(PLATFORM_LUMIN && !UNITY_EDITOR)

using System;
using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;
using System.Diagnostics;


using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.ObjdetectModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.UnityUtils.Helper;

using CVRect = OpenCVForUnity.CoreModule.Rect;
using Debug = UnityEngine.Debug;

[RequireComponent(typeof(WebCamTextureToMatHelper))]
public class HomePage : MonoBehaviour
{

    public ResolutionPreset requestedResolution = ResolutionPreset._640x480;
    public FPSPreset requestedFPS = FPSPreset._30;
    public Toggle rotate90DegreeToggle;
    public Toggle flipVerticalToggle;
    public Toggle flipHorizontalToggle;

    public RawImage inputImage;
    public RawImage captureImage;

    //public Image color_0;
    //public Image color_1;
    //public Image color_2;
    public Text pattern;
    public Button classify;
    public Button generate;
    public Image board;

    public GameObject cat;
    public Texture BicolorTex;
    public Texture CalicoTex;
    public Texture ColorpointTex;
    public Texture MixTex;
    public Texture OrangeTex;
    public Texture SolidTex;
    public Texture TabbyTex;

    //result
    public string result;
    public Color[] color;

    // WebCam
    Texture2D webcam; // the texture for webcam display
    WebCamTextureToMatHelper webCamTextureToMatHelper;

    // OpenCV
    private const string catCascadePath = "haarcascade_frontalcatface_extended.xml";
    Mat cropped;
    Texture2D catFace; // the texuture of captured, cropped, normalized cat face
    CVRect[] catFaceBox; // the outline of captured cat face
    CascadeClassifier catCascade;

    // Barracuda
    public NNModel catModel;
    Model model;
    IWorker worker;
    string[] labels = { "Bicolor", "Calico", "Colorpoint", "Mix", "Orange", "Solid", "Tabby" };


    // Use this for initialization
    void Start()
    {
        webCamTextureToMatHelper = gameObject.GetComponent<WebCamTextureToMatHelper>();
        int width, height;
        Dimensions(requestedResolution, out width, out height);
        webCamTextureToMatHelper.requestedWidth = width;
        webCamTextureToMatHelper.requestedHeight = height;
        webCamTextureToMatHelper.requestedFPS = (int)requestedFPS;
        webCamTextureToMatHelper.Initialize();


        if (Application.platform == RuntimePlatform.Android || Application.platform == RuntimePlatform.IPhonePlayer)
        {
            RectTransform rt = inputImage.GetComponent<RectTransform>();
            //rt.sizeDelta = new Vector2(720, 1280);
            //rt.localScale = new Vector3(1.6f, 1.6f, 1f);
            rt.sizeDelta = new Vector2(480, 640);
            rt.localScale = new Vector3(2.75f, 2.75f, 1f);
        }

        //////////////////////////////////////////
        // Cascade initiation
        //////////////////////////////////////////
        catCascade = new CascadeClassifier();
        catCascade.load(Utils.getFilePath(catCascadePath));
        if (catCascade.empty())
        {
            Debug.LogError("Cannot load cascade");
        }
        else
        {
            Debug.Log("Successfuly loaded cascade");
        }


        //////////////////////////////////////////
        // Barracuda initiation
        //////////////////////////////////////////
        model = ModelLoader.Load(catModel);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
        if (worker == null)
        {
            Debug.LogError("Cannot load model");
        }
        else
        {
            Debug.Log("Successfully loaded model");
        }


        board.GetComponent<Image>().enabled = false;
        captureImage.GetComponent<RawImage>().enabled = false;
        cat.SetActive(false);
        classify.GetComponent<Button>().enabled = false;
        generate.GetComponent<Button>().enabled = false;
    }



    public void OnWebCamTextureToMatHelperInitialized()
    {
        Debug.Log("OnWebCamTextureToMatHelperInitialized");

        Mat webCamTextureMat = webCamTextureToMatHelper.GetMat();

        webcam = new Texture2D(webCamTextureMat.cols(), webCamTextureMat.rows(), TextureFormat.RGBA32, false);

        Utils.fastMatToTexture2D(webCamTextureMat, webcam);

        gameObject.GetComponent<Renderer>().material.mainTexture = webcam;

        gameObject.transform.localScale = new Vector3(webCamTextureMat.cols(), webCamTextureMat.rows(), 1);
        Debug.Log("Screen.width " + Screen.width + " Screen.height " + Screen.height + " Screen.orientation " + Screen.orientation);

        float width = webCamTextureMat.width();
        float height = webCamTextureMat.height();

        float widthScale = (float)Screen.width / width;
        float heightScale = (float)Screen.height / height;
        if (widthScale < heightScale)
        {
            Camera.main.orthographicSize = (width * (float)Screen.height / (float)Screen.width) / 2;
        }
        else
        {
            Camera.main.orthographicSize = height / 2;
        }
    }
    public void OnWebCamTextureToMatHelperDisposed()
    {
        Debug.Log("OnWebCamTextureToMatHelperDisposed");

        if (webcam != null)
        {
            Texture2D.Destroy(webcam);
            webcam = null;
        }

        worker.Dispose();
    }
    public void OnWebCamTextureToMatHelperErrorOccurred(WebCamTextureToMatHelper.ErrorCode errorCode)
    {
        Debug.Log("OnWebCamTextureToMatHelperErrorOccurred " + errorCode);

    }
    void OnDestroy()
    {
        webCamTextureToMatHelper.Dispose();
    }
    public void OnPlayButtonClick()
    {
        webCamTextureToMatHelper.Play();
    }
    public void OnPauseButtonClick()
    {
        webCamTextureToMatHelper.Pause();
    }
    public void OnStopButtonClick()
    {
        webCamTextureToMatHelper.Stop();
    }
    public void OnChangeCameraButtonClick()
    {
        webCamTextureToMatHelper.requestedIsFrontFacing = !webCamTextureToMatHelper.IsFrontFacing();
    }
    public void OnRequestedResolutionDropdownValueChanged(int result)
    {
        if ((int)requestedResolution != result)
        {
            requestedResolution = (ResolutionPreset)result;

            int width, height;
            Dimensions(requestedResolution, out width, out height);

            webCamTextureToMatHelper.Initialize(width, height);
        }
    }
    public void OnRequestedFPSDropdownValueChanged(int result)
    {
        string[] enumNames = Enum.GetNames(typeof(FPSPreset));
        int value = (int)System.Enum.Parse(typeof(FPSPreset), enumNames[result], true);

        if ((int)requestedFPS != value)
        {
            requestedFPS = (FPSPreset)value;

            webCamTextureToMatHelper.requestedFPS = (int)requestedFPS;
        }
    }
    public void OnRotate90DegreeToggleValueChanged()
    {
        if (rotate90DegreeToggle.isOn != webCamTextureToMatHelper.rotate90Degree)
        {
            webCamTextureToMatHelper.rotate90Degree = rotate90DegreeToggle.isOn;
        }
    }
    public void OnFlipVerticalToggleValueChanged()
    {
        if (flipVerticalToggle.isOn != webCamTextureToMatHelper.flipVertical)
        {
            webCamTextureToMatHelper.flipVertical = flipVerticalToggle.isOn;
        }


    }
    public void OnFlipHorizontalToggleValueChanged()
    {
        if (flipHorizontalToggle.isOn != webCamTextureToMatHelper.flipHorizontal)
        {
            webCamTextureToMatHelper.flipHorizontal = flipHorizontalToggle.isOn;
        }
    }
    public enum FPSPreset : int
    {
        _0 = 0,
        _1 = 1,
        _5 = 5,
        _10 = 10,
        _15 = 15,
        _30 = 30,
        _60 = 60,
    }
    public enum ResolutionPreset : byte
    {
        _50x50 = 0,
        _640x480,
        _1280x720,
        _1920x1080,
        _9999x9999,
    }
    private void Dimensions(ResolutionPreset preset, out int width, out int height)
    {
        switch (preset)
        {
            case ResolutionPreset._50x50:
                width = 50;
                height = 50;
                break;
            case ResolutionPreset._640x480:
                width = 640;
                height = 480;
                break;
            case ResolutionPreset._1280x720:
                width = 1280;
                height = 720;
                break;
            case ResolutionPreset._1920x1080:
                width = 1920;
                height = 1080;
                break;
            case ResolutionPreset._9999x9999:
                width = 9999;
                height = 9999;
                break;
            default:
                width = height = 0;
                break;
        }
    }


    void Update()
    {
        if (webCamTextureToMatHelper.IsPlaying() && webCamTextureToMatHelper.DidUpdateThisFrame())
        {
            Mat webcam = webCamTextureToMatHelper.GetMat();
            if (Time.frameCount % 5 == 0)
            {
                catFaceBox = CVProcessor.DetectCatFace(webcam, catCascade);
            }
            if (catFaceBox != null && catFaceBox.Length > 0)
            {
                Debug.Log("Detected...");
                Imgproc.rectangle(webcam, catFaceBox[0].tl(), catFaceBox[0].br(), new Scalar(255, 0, 0, 255), 2);
            }
            Utils.fastMatToTexture2D(webcam, this.webcam);
            inputImage.texture = this.webcam;
        }
    }


    public void Detect()
    {
        // capture and crop the cat face
        Mat img = webCamTextureToMatHelper.GetMat();
        Mat original = img.clone();
        Debug.Log("Capturing...");

        if (catFaceBox.Length < 1)
        {
            Debug.LogWarning("Failed to capture");
        }
        else
        {
            Debug.Log("Captured Cat!");
            Debug.Log("x:" + catFaceBox[0].x + ", y:" + catFaceBox[0].y + ", s:" + catFaceBox[0].width);
            captureImage.GetComponent<RawImage>().enabled = true;
            classify.GetComponent<Button>().enabled = true;
            

            // crop the face out by the index 0 detected cat face and normalize it
            cropped = CVProcessor.CropCatFace(catFaceBox, original, 0);

            // transform it to Texture2D
            Texture2D captured = new Texture2D(cropped.cols(), cropped.rows(), TextureFormat.RGBA32, false);
            Utils.matToTexture2D(cropped, captured);

            catFace = CVProcessor.Normalize2Tensor(captured);

            captureImage.texture = catFace;
        }

    }

    public void Classify()
    {

        Stopwatch timerCV = new Stopwatch();
        Stopwatch timerNN = new Stopwatch();

        // calculate main color via kmeans
        timerCV.Start();
        color = CVProcessor.ColorKMEANS(cropped, 3, CVProcessor.Mode.HSV);
        timerCV.Stop();
        // classify the pattern of the cat
        timerNN.Start();
        result = CVProcessor.PredictNeuroNetwork(catFace, worker, labels);
        timerNN.Stop();

        generate.GetComponent<Button>().enabled = true;
        // display result and color on to canvas
        //color_0.GetComponent<Image>().color = colors[0];
        //color_1.GetComponent<Image>().color = colors[1];
        //color_2.GetComponent<Image>().color = colors[2];
        pattern.GetComponent<Text>().text = "Your cat is " + result;

        Debug.Log("Neural Network Runtime: " + timerNN.ElapsedMilliseconds + "ms\n"
            + "K-Means Algorithm Runtime: " + timerCV.ElapsedMilliseconds + "ms\n"
            + "Result is: " + result);
    }

    public void Generate()
    {
        cat.SetActive(true);
        Material catMaterial = cat.GetComponent<MeshRenderer>().material;
        switch (result)
        {
            case "Bicolor":
                catMaterial.SetTexture("Mask", BicolorTex);
                catMaterial.SetColor("MajorColor", color[1]);
                catMaterial.SetColor("PatternColor", new Color((float)0.9, (float)0.9, (float)0.9, 1));
                break;
            case "Calico":
                catMaterial.SetTexture("Mask", CalicoTex);
                catMaterial.SetColor("MajorColor", color[0]);
                catMaterial.SetColor("PatternColor", color[1]);
                catMaterial.SetColor("SubColor", color[2]);
                break;
            case "Colorpoint":
                catMaterial.SetTexture("Mask", ColorpointTex);
                catMaterial.SetColor("MajorColor", new Color((float)0.9, (float)0.9, (float)0.9, 1));
                catMaterial.SetColor("PatternColor", color[1]);
                break;
            case "Mix":
                catMaterial.SetTexture("Mask", MixTex);
                catMaterial.SetColor("MajorColor", color[0]);
                catMaterial.SetColor("PatternColor", color[1]);
                catMaterial.SetColor("SubColor", new Color((float)0.9, (float)0.9, (float)0.9, 1));
                break;
            case "Orange":
                catMaterial.SetTexture("Mask", OrangeTex);
                catMaterial.SetColor("MajorColor", new Color((float)0.95, (float)0.6, (float)0.1, 1));
                break;
            case "Solid":
                catMaterial.SetTexture("Mask", SolidTex);
                catMaterial.SetColor("MajorColor", color[0]);
                break;
            case "Tabby":
                catMaterial.SetTexture("Mask", TabbyTex);
                catMaterial.SetColor("MajorColor", color[0]);
                catMaterial.SetColor("PatternColor", color[1]);
                break;
        }
    }

    public void Guide()
    {
        board.GetComponent<Image>().enabled = true;
    }

    public void Close()
    {
        if (board.GetComponent<Image>().isActiveAndEnabled == true)
        {
            board.GetComponent<Image>().enabled = false;
        }
    }
}


#endif