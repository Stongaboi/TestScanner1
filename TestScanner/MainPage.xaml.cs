using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Media;

namespace TestScanner
{
    public partial class MainPage : ContentPage
    {
        private YoloDetector _yoloDetector;

        public MainPage()
        {
            InitializeComponent();
            BarcodeScanner.Mobile.Methods.AskForRequiredPermission();

            InitializeYoloDetector();

        }

        private void InitializeYoloDetector()
        {
            try
            {
                string dcimDirectory = Android.App.Application.Context.GetExternalFilesDir(Android.OS.Environment.DirectoryDcim).AbsolutePath;
                string modelPath = Path.Combine(dcimDirectory, "yolov5s.onnx");

                if (!File.Exists(modelPath))
                {
                    throw new FileNotFoundException($"Model file not found at: {modelPath}");
                }

                _yoloDetector = new YoloDetector(modelPath);
                Console.WriteLine("YOLO Detector initialized successfully.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error initializing YOLO detector: {ex.Message}");
            }
        }



        //private async void OnTakePhotoClicked(object sender, EventArgs e)
        //{
        //    try
        //    {
        //        if (_yoloDetector == null)
        //        {
        //            await DisplayAlert("Error", "YOLO detector not initialized yet. Please wait.", "OK");
        //            return;
        //        }

        //        // Capture photo
        //        var photo = await MediaPicker.CapturePhotoAsync();
        //        if (photo == null)
        //        {
        //            ResultLabel.Text = "Photo capture canceled.";
        //            return;
        //        }

        //        // Save the photo locally
        //        var filePath = Path.Combine(FileSystem.CacheDirectory, photo.FileName);
        //        using (var stream = await photo.OpenReadAsync())
        //        using (var fileStream = File.Create(filePath))
        //        {
        //            await stream.CopyToAsync(fileStream);
        //        }

        //        // Load and preprocess image
        //        using var bitmap = SKBitmap.Decode(filePath);
        //        var tensor = GetTensorFromImage(bitmap);

        //        // Run YOLO detection
        //        var predictions = _yoloDetector.Detect(tensor);

        //        // Display results
        //        if (predictions.Any())
        //        {
        //            var resultText = string.Join("\n", predictions.Select(p =>
        //                $"{p.Label}: {p.Confidence:P} at ({p.X}, {p.Y}) [{p.Width}x{p.Height}]"));
        //            ResultLabel.Text = resultText;
        //        }
        //        else
        //        {
        //            ResultLabel.Text = "No license plates detected.";
        //        }
        //    }
        //    catch (Exception ex)
        //    {
        //        await DisplayAlert("Error", ex.Message, "OK");
        //    }
        //}

        private async void OnTakePhotoClicked(object sender, EventArgs e)
        {
            try
            {
                if (_yoloDetector == null)
                {
                    await DisplayAlert("Error", "YOLO detector not initialized. Please wait.", "OK");
                    return;
                }

                var photo = await MediaPicker.CapturePhotoAsync();
                if (photo == null)
                {
                    ResultLabel.Text = "Photo capture canceled.";
                    return;
                }

                var filePath = Path.Combine(FileSystem.CacheDirectory, photo.FileName);
                using (var stream = await photo.OpenReadAsync())
                using (var fileStream = File.Create(filePath))
                {
                    await stream.CopyToAsync(fileStream);
                }

                using var bitmap = SKBitmap.Decode(filePath);
                var float32Tensor = GetTensorFromImage(bitmap); // float32 array
                var float16Input = ConvertFloat32ToFloat16(float32Tensor); // float16 as byte[]

                var predictions = _yoloDetector.Detect(float16Input);

                if (predictions.Any())
                {
                    var resultText = string.Join("\n", predictions.Select(p =>
                        $"{p.Label}: {p.Confidence:P} at ({p.X}, {p.Y}) [{p.Width}x{p.Height}]"));
                    ResultLabel.Text = resultText;
                }
                else
                {
                    ResultLabel.Text = "No license plates detected.";
                }
            }
            catch (Exception ex)
            {
                await DisplayAlert("Error", ex.Message, "OK");
            }
        }



        private float[] GetTensorFromImage(SKBitmap bitmap)
        {
            // Resize the image to 640x640
            using var resizedBitmap = bitmap.Resize(new SKImageInfo(640, 640), SKFilterQuality.High);
            var float32Buffer = new float[1 * 3 * 640 * 640];
            int index = 0;

            for (int y = 0; y < resizedBitmap.Height; y++)
            {
                for (int x = 0; x < resizedBitmap.Width; x++)
                {
                    var pixel = resizedBitmap.GetPixel(x, y);
                    float32Buffer[index++] = pixel.Red / 255f;   // Red channel normalized
                    float32Buffer[index++] = pixel.Green / 255f; // Green channel normalized
                    float32Buffer[index++] = pixel.Blue / 255f;  // Blue channel normalized
                }
            }

            return float32Buffer;
        }


        private byte[] ConvertFloat32ToFloat16(float[] float32Array)
        {
            // Ensure that the destination array is twice the size since Half is 2 bytes
            var byteArray = new byte[float32Array.Length * 2];
            for (int i = 0; i < float32Array.Length; i++)
            {
                // Convert each float32 value to float16
                Half halfValue = (Half)float32Array[i];
                byte[] halfBytes = BitConverter.GetBytes(halfValue);

                // Copy the 2 bytes of the float16 to the byte array
                byteArray[i * 2] = halfBytes[0];
                byteArray[i * 2 + 1] = halfBytes[1];
            }

            return byteArray;
        }



        private Half[] ToFloat16Array(float[] float32Array)
        {
            var float16Array = new Half[float32Array.Length];
            for (int i = 0; i < float32Array.Length; i++)
            {
                float16Array[i] = (Half)float32Array[i];
            }
            return float16Array;
        }


    }
}
