// Class to hold input to the model
class ModelInput
{
    // Image in a byte array
    public byte[] Image { get; set; }
    // Target label as a key
    public UInt32 LabelAsKey { get; set; }
    // Path to image on disk
    public string ImagePath { get; set; }
    // Label as string
    public string Label { get; set; }
}