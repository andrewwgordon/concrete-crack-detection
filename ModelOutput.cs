// Class to for output from model
class ModelOutput
{
    // Path to image on disk
    public string ImagePath { get; set; }
    // Label for classification
    public string Label { get; set; }
    // Predicted classification (by model)
    public string PredictedLabel { get; set; }
}