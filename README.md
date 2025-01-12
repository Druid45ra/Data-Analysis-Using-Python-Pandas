# Data-Analysis-Using-Python-Pandas
The code is a unit test for the preprocess data function from the dataset analyzer module. It uses the unittest framework to verify that the preprocessing function correctly handles missing values and outliers.

**Project: Data Analysis Using Python Pandas**
Overview
This project provides a comprehensive tool for analyzing datasets using Python and the Pandas library. It includes functionalities for data preprocessing, basic and advanced statistical analysis, and visualization. Additionally, it features a simple graphical user interface (GUI) for ease of use.

Features
Data Preprocessing:

Handles missing values by filling them with the mean of the respective columns.
Removes outliers that are more than 3 standard deviations from the mean.
Basic Analysis:

Provides an overview of the dataset, including the number of rows and columns, and the names of the columns.
Displays the first 10 rows of the dataset.
Generates a data quality report showing the count of missing values in each column.
Performs numerical and categorical column analysis, including summary statistics and value counts.
Advanced Analysis:

Performs linear regression analysis on specified columns.
Conducts Principal Component Analysis (PCA) for dimensionality reduction.
Visualization:

Generates histograms for numerical columns.
Creates bar charts for categorical columns.
Saves visualizations as PNG files.
Graphical User Interface (GUI):

Allows users to select a CSV file for analysis through a simple GUI built with tkinter.
Unit Testing:

Includes unit tests to verify the functionality of the data preprocessing steps using the unittest framework.
Installation
Install Python:

Ensure you have Python installed on your system. You can download it from python.org.
Install pip:

If pip is not installed, download get-pip.py from get-pip.py and run it using:
Install Required Libraries:

Install the required libraries using pip:
Usage
Run the Analysis Tool:

Execute the dataset_analyzer.py script to launch the GUI and select a CSV file for analysis:
View Analysis Report:

The analysis report will be displayed in the terminal and saved as a text file in the same directory.
Run Unit Tests:

Execute the test_dataset_analyzer.py script to run the unit tests:
Code Explanation
dataset_analyzer.py
Preprocess Data:

Advanced Analysis:

Analyze Dataset:

Select File and GUI:

test_dataset_analyzer.py
Unit Test for Preprocess Data:
Conclusion
This project provides a robust tool for dataset analysis with preprocessing, visualization, and advanced statistical analysis capabilities. The inclusion of a GUI makes it user-friendly, and the unit tests ensure the reliability of the preprocessing steps.
