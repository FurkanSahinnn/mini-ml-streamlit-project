# Breast Cancer Classification Project
This project aims to evaluate the performance of classification algorithms on the Wisconsin Breast Cancer dataset. The dataset consists of features extracted from digital images of breast cancer cells.

## Algorithms Used:
The project includes K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Naive Bayes algorithms. The best parameters for each algorithm were found using the GridSearchCV method, and the models were trained accordingly.

## Findings:
* Best parameters for KNN: (number of samples, number of neighbors)
* Best parameters for SVM: (C value, kernel function)(The kernel parameter calculation took a long time, so it has not been included!)
* Best parameters for Naive Bayes: (parameters listed)
Performance metrics such as accuracy, precision, and recall were calculated for each model, and the results were presented comparatively.


## Installation/Requirements:
```console
pip install numpy
pip install pandas
pip install seaborn
pip install streamlit
pip install scikit-learn
pip install matplotlib
```

## Usage - Local:
```console
streamlit run main.py
```

## Usage - Local (Auto-Save Version):
```console
streamlit run main.py --server.runOnSave true
```

## If you need to check installed packages:
```console
pip list
```

