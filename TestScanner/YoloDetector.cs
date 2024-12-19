using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System.Collections.Generic;
using System.Linq;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Runtime.InteropServices;

public class YoloDetector
{
    private readonly InferenceSession _session;

    public YoloDetector(string modelPath)
    {
        // Load the YOLO model
        try
        {
            Console.WriteLine("Initializing ONNX Runtime session...");
            _session = new InferenceSession(modelPath);
            Console.WriteLine("ONNX Runtime session initialized.");

            // Check input metadata
            Console.WriteLine("Checking model input metadata...");
            foreach (var input in _session.InputMetadata)
            {
                Console.WriteLine($"Input Name: {input.Key}");
                Console.WriteLine($"Element Type: {input.Value.ElementType}");
                Console.WriteLine($"Shape: {string.Join(", ", input.Value.Dimensions)}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error initializing ONNX Runtime session: {ex.Message}");
            throw;
        }

    }




    public List<YoloPrediction> Detect(byte[] float16Input)
    {
        string inputName = "images";
        var inputShape = new int[] { 1, 3, 640, 640 };

        try
        {
            // Validate input size
            if (float16Input.Length != inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3] * 2)
            {
                Console.WriteLine($"Invalid input size: {float16Input.Length}. Expected size: {inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3] * 2}.");
                return new List<YoloPrediction>();
            }

            // Populate input tensor
            var inputTensor = new DenseTensor<Half>(inputShape);
            for (int i = 0; i < float16Input.Length / 2; i++)
            {
                inputTensor.Buffer.Span[i] = BitConverter.ToHalf(float16Input, i * 2);
            }

            Console.WriteLine("Input tensor initialized successfully.");

            // Wrap input tensor into NamedOnnxValue
            var inputs = new List<NamedOnnxValue>
            {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            int retries = 3;
            while (retries > 0)
            {
                try
                {
                    // Run inference
                    Console.WriteLine("Running inference...");
                    var results = _session.Run(inputs);

                    try
                    {
                        // Extract and process output tensor
                        var outputTensor = results.FirstOrDefault()?.AsTensor<Half>();
                        if (outputTensor == null)
                        {
                            Console.WriteLine("Output tensor is null.");
                            throw new Exception("Output tensor retrieval failed.");
                        }

                        Console.WriteLine("Inference successful. Processing output...");
                        return ParseOutput(outputTensor.ToArray(), new int[] { 1, 25200, 85 });
                    }
                    finally
                    {
                        foreach (var result in results)
                        {
                            result.Dispose();
                        }
                    }
                }
                catch (Exception ex)
                {
                    retries--;
                    Console.WriteLine($"Inference attempt failed: {ex.Message}");
                    if (retries == 0)
                    {
                        throw new Exception($"Inference failed after multiple attempts: {ex.Message}");
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Critical error in Detect method: {ex.Message}");
        }

        // Default return for safety
        return new List<YoloPrediction>();
    }






    private List<YoloPrediction> ParseOutput(Half[] outputArray, int[] outputShape)
    {
        var predictions = new List<YoloPrediction>();
        int numClasses = outputShape[2] - 5; // 85 - 5 = 80 classes

        for (int i = 0; i < outputShape[1]; i++) // Loop through 25200 grid cells
        {
            int offset = i * outputShape[2]; // Offset for each grid cell
            var confidence = (float)outputArray[offset + 4];

            if (confidence > 0.5) // Confidence threshold
            {
                // Get bounding box coordinates
                var x = (float)outputArray[offset];
                var y = (float)outputArray[offset + 1];
                var width = (float)outputArray[offset + 2];
                var height = (float)outputArray[offset + 3];

                // Get class scores
                var classScores = new float[numClasses];
                for (int j = 0; j < numClasses; j++)
                {
                    classScores[j] = (float)outputArray[offset + 5 + j];
                }

                // Find the best class
                var maxScore = classScores.Max();
                var classIndex = Array.IndexOf(classScores, maxScore);

                if (maxScore > 0.5) // Class probability threshold
                {
                    predictions.Add(new YoloPrediction
                    {
                        Label = GetClassName(classIndex), // Get the class name
                        Confidence = confidence * maxScore,
                        X = x,
                        Y = y,
                        Width = width,
                        Height = height
                    });
                }
            }
        }

        return predictions;
    }


    private string GetClassName(int classIndex)
    {
        var classNames = new Dictionary<int, string>
{
    { 0, "person" }, { 1, "bicycle" }, { 2, "car" }, { 3, "motorcycle" },
    { 4, "airplane" }, { 5, "bus" }, { 6, "train" }, { 7, "truck" },
    { 8, "boat" }, { 9, "traffic light" }, { 10, "fire hydrant" },
    { 11, "stop sign" }, { 12, "parking meter" }, { 13, "bench" },
    { 14, "bird" }, { 15, "cat" }, { 16, "dog" }, { 17, "horse" },
    { 18, "sheep" }, { 19, "cow" }, { 20, "elephant" }, { 21, "bear" },
    { 22, "zebra" }, { 23, "giraffe" }, { 24, "backpack" }, { 25, "umbrella" },
    { 26, "handbag" }, { 27, "tie" }, { 28, "suitcase" }, { 29, "frisbee" },
    { 30, "skis" }, { 31, "snowboard" }, { 32, "sports ball" }, { 33, "kite" },
    { 34, "baseball bat" }, { 35, "baseball glove" }, { 36, "skateboard" },
    { 37, "surfboard" }, { 38, "tennis racket" }, { 39, "bottle" },
    { 40, "wine glass" }, { 41, "cup" }, { 42, "fork" }, { 43, "knife" },
    { 44, "spoon" }, { 45, "bowl" }, { 46, "banana" }, { 47, "apple" },
    { 48, "sandwich" }, { 49, "orange" }, { 50, "broccoli" }, { 51, "carrot" },
    { 52, "hot dog" }, { 53, "pizza" }, { 54, "donut" }, { 55, "cake" },
    { 56, "chair" }, { 57, "couch" }, { 58, "potted plant" }, { 59, "bed" },
    { 60, "dining table" }, { 61, "toilet" }, { 62, "tv" }, { 63, "laptop" },
    { 64, "mouse" }, { 65, "remote" }, { 66, "keyboard" }, { 67, "cell phone" },
    { 68, "microwave" }, { 69, "oven" }, { 70, "toaster" }, { 71, "sink" },
    { 72, "refrigerator" }, { 73, "book" }, { 74, "clock" }, { 75, "vase" },
    { 76, "scissors" }, { 77, "teddy bear" }, { 78, "hair drier" }, { 79, "toothbrush" }
};

        return classNames.ContainsKey(classIndex) ? classNames[classIndex] : "unknown";
    }






    private DenseTensor<float> GetTensorFromImage(SKBitmap bitmap)
    {
        // Resize the image to 640x640
        using var resizedBitmap = bitmap.Resize(new SKImageInfo(640, 640), SKFilterQuality.High);

        // Create a tensor with Float32 precision
        var tensor = new DenseTensor<float>(new[] { 1, 3, 640, 640 });

        for (int y = 0; y < resizedBitmap.Height; y++)
        {
            for (int x = 0; x < resizedBitmap.Width; x++)
            {
                var pixel = resizedBitmap.GetPixel(x, y);
                tensor[0, 0, y, x] = pixel.Red / 255f;   // Red channel
                tensor[0, 1, y, x] = pixel.Green / 255f; // Green channel
                tensor[0, 2, y, x] = pixel.Blue / 255f;  // Blue channel
            }
        }

        // Log for debugging
        //Console.WriteLine($"Tensor Shape: {string.Join(", ", tensor.Dimensions)}");
        Console.WriteLine($"First Pixel Value: {tensor[0, 0, 0, 0]}");

        return tensor;
    }

    private List<YoloPrediction> ParseOutput(float[] output)
    {
        var predictions = new List<YoloPrediction>();

        // Example logic: parse output for confidence and bounding boxes
        for (int i = 0; i < output.Length; i += 6)
        {
            predictions.Add(new YoloPrediction
            {
                X = output[i],
                Y = output[i + 1],
                Width = output[i + 2],
                Height = output[i + 3],
                Confidence = output[i + 4],
                Label = "License Plate" // Replace with actual label mapping logic
            });
        }

        return predictions;
    }
}

public class YoloPrediction
{
    public string Label { get; set; }
    public float Confidence { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
    public float Width { get; set; }
    public float Height { get; set; }
}
