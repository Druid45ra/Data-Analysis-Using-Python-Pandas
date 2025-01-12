import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Button, Label
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): The dataset to preprocess
    
    Returns:
        pd.DataFrame: The preprocessed dataset
    """
    # Fill missing values with the mean of the column for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Remove outliers (values that are more than 3 standard deviations from the mean)
    for col in numeric_columns:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] >= mean - 3 * std) & (df[col] <= mean + 3 * std)]
    
    return df

def advanced_analysis(df):
    """
    Perform advanced statistical analysis on the dataset.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
    """
    # Linear Regression
    if 'Income' in df.columns and 'Years_Experience' in df.columns:
        X = df[['Years_Experience']]
        y = df['Income']
        model = LinearRegression()
        model.fit(X, y)
        print("\n=== Linear Regression ===")
        print(f"Intercept: {model.intercept_}")
        print(f"Coefficient: {model.coef_[0]}")
    
    # PCA
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 1:
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(df[numeric_columns])
        print("\n=== Principal Component Analysis (PCA) ===")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

def analyze_dataset(file_path):
    """
    Analyze a dataset using pandas with comprehensive error handling and reporting.
    
    Args:
        file_path (str): Path to the dataset file
    """
    try:
        print("\nLoading dataset...")
        df = pd.read_csv(file_path)
        
        # Preprocess data
        df = preprocess_data(df)
        
        report = []
        
        report.append("\n=== Dataset Overview ===")
        report.append(f"Number of rows: {len(df)}")
        report.append(f"Number of columns: {len(df.columns)}")
        report.append("\nColumn names:")
        for col in df.columns:
            report.append(f"- {col}")
            
        report.append("\n=== First 10 rows of data ===")
        report.append(df.head(10).to_string())
        
        report.append("\n=== Data Quality Report ===")
        report.append("\nMissing values count:")
        report.append(df.isnull().sum().to_string())
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            report.append("\n=== Numerical Columns Analysis ===")
            report.append("\nSummary statistics:")
            report.append(df[numeric_columns].describe().round(2).to_string())
            
            report.append("\nCorrelation matrix:")
            correlation_matrix = df[numeric_columns].corr()
            report.append(correlation_matrix.round(2).to_string())
            
            # Plot histograms for numerical columns
            for col in numeric_columns:
                plt.figure()
                df[col].hist(bins=20)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.savefig(f'{col}_distribution.png')
                plt.close()
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            report.append("\n=== Categorical Columns Analysis ===")
            for col in categorical_columns:
                report.append(f"\nValue counts in {col}:")
                report.append(df[col].value_counts().to_string())
                
                # Plot bar charts for categorical columns
                plt.figure()
                df[col].value_counts().plot(kind='bar')
                plt.title(f'Value counts of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.savefig(f'{col}_value_counts.png')
                plt.close()
        
        report.append("\n=== Basic Distribution Patterns ===")
        for col in numeric_columns:
            values = df[col].dropna()
            if len(values) > 0:
                report.append(f"\nDistribution of {col}:")
                hist = np.histogram(values, bins=10)
                max_count = max(hist[0])
                for count in hist[0]:
                    bar = '#' * int(50 * count / max_count)
                    report.append(bar)
        
        analysis_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        report.append(f"\nAnalysis completed at: {analysis_timestamp}")
        
        # Save report to file
        report_filename = f"dataset_analysis_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w') as f:
            f.write("\n".join(report))
        
        print(f"\nAnalysis report saved to: {report_filename}")
        
        # Perform advanced analysis
        advanced_analysis(df)
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def select_file():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename()
    if file_path:
        analyze_dataset(file_path)

def main():
    root = Tk()
    root.title("Dataset Analysis Tool")
    
    label = Label(root, text="Select a CSV file to analyze")
    label.pack(pady=10)
    
    button = Button(root, text="Select File", command=select_file)
    button.pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()
