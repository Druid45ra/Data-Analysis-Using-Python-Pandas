import unittest
import pandas as pd
import numpy as np
from dataset_analyzer import preprocess_data

class TestDatasetAnalyzer(unittest.TestCase):
    
    def test_preprocess_data(self):
        data = {
            'Age': [24, 30, np.nan, 35, 28],
            'Income': [50000, 60000, 70000, 80000, 90000],
            'Years_Experience': [1, 2, 3, 4, 5]
        }
        df = pd.DataFrame(data)
        df_processed = preprocess_data(df)
        self.assertFalse(df_processed.isnull().values.any(), "There should be no missing values")
        self.assertTrue((df_processed['Age'] >= df_processed['Age'].mean() - 3 * df_processed['Age'].std()).all(), "There should be no outliers in Age")

if __name__ == '__main__':
    unittest.main()
