# ShapeVVE: Variable Evaluator for Multivariate Time Series Shapelets Extraction

This repository contains the official implementation of the paper "ShapeVVE: Variable Evaluator for multivariate time series shapelets extraction".

## Description
This repository provides code for evaluating the utility of shapelets in time series classification tasks. 

🔍 **Enhancing Interpretability in Time Series Classification via High-Value Shapelets**  

This repository explores the value of **shapelets** to improve the interpretability of time series classification while maintaining competitive accuracy. By identifying and utilizing **high-value shapelets**, we aim to enhance model efficiency and provide meaningful insights into decision-making processes.  

✨ **Key Features:**  
- **Shapelet Value Assessment**: Evaluates shapelets based on their discriminative power and interpretability.  
- **Efficiency Optimization**: Uses high-value shapelets for classification without sacrificing accuracy.  
- **Interpretability Enhancement**: Investigates how **dimensionality selection** of shapelets contributes to clearer model explanations.  

📊 **Applications:** Suitable for time series analysis tasks where both **performance** and **human-understandable explanations** are crucial.  

Contributions and feedback are welcome! 🚀

## Configuration
**Note**: This repository contains large data files. To manage repository size effectively, we have included only the BasicMotions dataset directly. The main dataset `Multivariate2018_arff.zip` located in the `data_files/` folder is managed using [Git LFS](https://git-lfs.com) (Large File Storage).

### For the Multivariate2018_arff.zip Dataset

#### Option 1: Manual Download (Recommended for quick access)
You can download the complete dataset directly from the source:
- **Download URL**: http://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_arff.zip
- After downloading, place the file in the `data_files/` folder

#### Option 2: Using Git LFS (For complete repository cloning)
If you want to clone the entire repository with all files:

1. **Install Git LFS**:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install git-lfs
   
   # On macOS with Homebrew
   brew install git-lfs
   
   # On Windows (Git Bash)
   git lfs install
    ```
2. **Clone the repository with LFS**:

  ```bash
   git lfs clone https://github.com/your-username/your-repository.git
   ```

or if you've already cloned without LFS:

  ```bash
  git lfs pull
  ```


## Usage

- Two Jupyter notebook examples are provided in Test/Example/ to demonstrate how to run the shapelet learning algorithm:

    - Without dimension selection

    - With dimension selection

## Experiments

- The Test/Experiment/ directory contains:

    - Main results table

    - Friedman test results

    - Wilcoxon signed-rank test results