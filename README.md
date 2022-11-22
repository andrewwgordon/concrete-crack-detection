# Microsoft ML.NET Image Classification
## Overview
This is an implementation of the [Microsoft ML.NET Image Classification tutorial for detecting cracks in concrete images](https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification-api-transfer-learning).
## Requirements
.NET Version=6.0.110 Linux x64 (tested on Ubuntu 22.04-x64)

Microsoft.ML Version=1.7.1

Microsoft.ML.TensorFlow Version=1.7.1

Microsoft.ML.Vision Version=1.7.1

Microsoft.ML.ImageAnalytics Version=1.7.1

SciSharp.TensorFlow.Redist Version=1.14.0

## Build
From command line (Linux)
```bash
$ dotnet restore
$ dotnet build
```
(Windows)
```cmd
C:\My Project>dotnet restore
C:\My Project>dotnet build
```
Check the packages.
```bash
$ dotnet list package
Project concrete_cracks has the following package references
   [net6.0]: 
   Top-level Package                  Requested   Resolved
   > Microsoft.ML                     1.7.1       1.7.1   
   > Microsoft.ML.ImageAnalytics      1.7.1       1.7.1   
   > Microsoft.ML.TensorFlow          1.7.1       1.7.1   
   > Microsoft.ML.Vision              1.7.1       1.7.1   
   > SciSharp.TensorFlow.Redist       1.14.0      1.14.0
$
```
## Deploying Images
[Download](https://digitalcommons.usu.edu/all_datasets/48) the concrete images, unzip the archive and deploy to the assets directory in your project folder. You should have a directory structure that looks something like:
```bash
assets
 D
  CD
  UD
 P
  CP
  UP
 W
  CW
  UW
```
For this exercise only the images in the D subdirectory containing cracked (CD) or uncracked (UD) images of concrete.
## Training the Model
At the command prompt 
```bash
$ dotnet run
```
Once complete the model is saved in the model directory.

Note: this model is configured to run on CPU and could take a couple of hours on a typical i5 4 Core CPU.