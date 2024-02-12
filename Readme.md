# Running the code:

This repository contains a minimum code to reproduce coreML issue on Mac M1. Below are the instructions to run it:

## Cloning the repo

1. Clone this repository to your local machine.
2. Navigate to the directory where you cloned the repository.


## Prerequisites

Install the requirements using the following command, (if needed, create a new conda environment and activate it first):

```
conda create -n test_coreML python=3.11
conda activate test_coreML
pip install -r requirements.txt
```


## Usage

1. Open a terminal or command prompt.
2. Navigate to the directory where the Python file is located.
3. Run the Python script using the following command:
4. ```
   python coreML_grid_sample.py
   ```

## Expected Output

The code will convert two grid_sample layer to coreML models and save them as `GridSampleModel.mlpackage` and `CustomGridSampleModel.mlpackage` respectively in the same directory.


The GridSampleModel.mlpackage is created using the `torch.nn.functional.grid_sample` function and the CustomGridSampleModel.mlpackage is created using the `custom grid_sample` function.

Performance of `GridSampleModel.mlpackage` when measured on M1 it could be noticed that it does not run on Neural Engine. However, it runs on Neural Engine on M2.

## Requirements

We created our own grid_sample function and converted it to coreML model: `CustomGridSampleModel.mlpackage` it runs on GPU. 

We would like to know if there is a way to convert `torch.nn.functional.grid_sample` to coreML model that runs on Neural Engine on M1. Or, if there is a way to convert `CustomGridSampleModel.mlpackage` to run on Neural Engine .




