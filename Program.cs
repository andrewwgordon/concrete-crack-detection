// ML training workflow for concrete image crack detection
//
// Microsoft library namespace imports
//
// The following packages are required for this project 
// to build
//
// Microsoft.ML Version=1.7.1
// Microsoft.ML.TensorFlow Version=1.7.1
// Microsoft.ML.Vision Version=1.7.1
// Microsoft.ML.ImageAnalytics Version=1.7.1
// SciSharp.TensorFlow.Redist Version=1.14.0
//
// See concrete_cracks.csproj for details
//
using System.Runtime.InteropServices;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Vision;

// Namespace
namespace concrete.images
{
    // Helper function to detect Operating System to support correct folder paths
    public static class myOperatingSystem
    {
        public static bool isWindows() =>RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        public static bool isMacOS() =>RuntimeInformation.IsOSPlatform(OSPlatform.OSX);
        public static bool isLinux() =>RuntimeInformation.IsOSPlatform(OSPlatform.Linux);
    }

    // Main program   
    public class Program
    {
        // Function to load and return a list of images contained
        // as an ImageData class
        private static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            // Get the list of files to process
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);
            // For each file in the list of files
            Console.WriteLine("Looking for images in {0}",folder);
            foreach (var file in files)
            {       
                // If the file extension is not JPG or PNG ignore the file
                // with the continue command
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;
                // Set label to the filename
                var label = Path.GetFileName(file);

                // If useFolderNameAsLabel is true then use the parent
                // folder name as the label for the image
                if (useFolderNameAsLabel)
                    label = Directory.GetParent(file).Name;
                // Else look for some not character seperator to look
                // for the label as a suffix
                // For example 00121021_D.JPG or 00121021-D.JPG
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }
                // Create a new instance of the ImageData class
                // and setting the ImagePath to the filename
                // and label
                // Note the yield command returns the next value
                // in the Iterator
                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };
            }        
        }

        // Function to display image and prediction to the console
        private static void OutputPrediction(ModelOutput prediction)
        {
            // Get the image filename
            string imageName = Path.GetFileName(prediction.ImagePath);
            // Display the image and prediction from the model to the console
            Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
        }

        // Function to classify a single image
        private static void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {       
            // Create a prediction engine from the current model held in the mlContext
            PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
            // Set the model import as the first image in the list
            ModelInput image = mlContext.Data.CreateEnumerable<ModelInput>(data,reuseRowObject:true).First();
            // Run the prediction and store the output in prediction
            ModelOutput prediction = predictionEngine.Predict(image);
            // Display the output to the console
            Console.WriteLine("Classifying single image");
            OutputPrediction(prediction);
        }

        // Main program entry point
        public static void Main(string[] args)
        {
            Console.WriteLine("Starting...");
            // Set up to scan folders for images
            // Set the project folder
            string basePath;
            string assetPath;
            if (myOperatingSystem.isLinux() || myOperatingSystem.isMacOS())
            {
                basePath = "../../../";
                assetPath = "assets/D";
            }
            else
            {
                basePath = "..\\..\\..\\";
                assetPath = "assets\\D";
            }
            var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, basePath));
            var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
            // Look for the images in the assets/D folder relative to the project
            var assetsRelativePath = Path.Combine(projectDirectory, assetPath);

            // Create a new instance of the ML.NET Engine context
            Console.WriteLine("Creating new ML.NET context...");
            MLContext mlContext = new MLContext();

            // Set images as a list of references to images on the disk
            // searching for files in the folder assets/D
            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);

            // Load the images from disk to memory (this uses streaming to limit memory use)
            IDataView imageData = mlContext.Data.LoadFromEnumerable(images);
            // Shuffle the data from disk to memory as the model training process requires
            Console.WriteLine("Setting up images stream...");
            IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);

            // Create the machine learning training pipeline
            // Set the input column as Label
            // MapValueToKey automatically handles categoricals to numbers
            Console.WriteLine("Creating preprocessing pipeline...");
            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "Label",
                    outputColumnName: "LabelAsKey")
                .Append(mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "Image",
                    imageFolder: assetsRelativePath,
                    inputColumnName: "ImagePath"));
            
            // Process the pipeline as defined above and store result
            // in the preProcessData variables
            Console.WriteLine("Executing the preprocessing pipeline...");
            IDataView preProcessedData = preprocessingPipeline
                    .Fit(shuffledData)
                    .Transform(shuffledData);

            // Train / Test split 70/30
            Console.WriteLine("Spliting the training and test set 70/30...");
            TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.3);
            // Validation is a split of the test set
            TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);
            
            // Set the variables for training, validation and testing
            IDataView trainSet = trainSplit.TrainSet;
            IDataView validationSet = validationTestSplit.TrainSet;
            IDataView testSet = validationTestSplit.TestSet;

            // Select the machine learning algorithm
            // The ImageClassificationTrainer searches through CNN Deep Learning models
            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                // Define the feature, in this case the image
                FeatureColumnName = "Image",
                // Define the target label, i.e. crack / no crack
                LabelColumnName = "LabelAsKey",
                // Set the validation set
                ValidationSet = validationSet,
                // Define the architecture of the Deep Learning Network
                // In this case ResnetV2101
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                // Setup a callback to display the metrics for each model training to the console
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                // Do not test the model on the training dataset
                TestOnTrainSet = false,
                // Use cached data if possible
                ReuseTrainSetBottleneckCachedValues = true,
                // Reuse validation data if possible
                ReuseValidationSetBottleneckCachedValues = true
            };

            // Configure the training pipeline as a ImageClassification
            // with the options
            Console.WriteLine("Creating the training pipeline...");
            var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));   

            // Start the training process
            // Wait for a long time here unless you have a GPU cluster
            Console.WriteLine("Training the model (go get a coffee!)...");
            ITransformer trainedModel = trainingPipeline.Fit(trainSet);
            
            // Save the model to disk
            string modelPath;
            if (myOperatingSystem.isLinux() || myOperatingSystem.isMacOS())
            {
                modelPath = "./model/model.zip";
            }
            else
            {
                modelPath = ".\\model\\model.zip";
            }
            Console.WriteLine("Saving the model to {0}",modelPath);
            mlContext.Model.Save(trainedModel,trainSet.Schema,modelPath);

            // Test the model out                        
            ClassifySingleImage(mlContext, testSet, trainedModel);
            Console.WriteLine("Finished...");
        }
    }
}