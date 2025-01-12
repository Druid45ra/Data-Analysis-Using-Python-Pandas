import pandas as pd

def create_sample_dataset():
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
        'Age': [24, 30, 22, 35, 28],
        'Score': [85, 90, 78, 88, 92]
    }
    df = pd.DataFrame(data)
    file_path = 'sample_dataset.csv'
    df.to_csv(file_path, index=False)
    print(f"Sample dataset created at: {file_path}")
    return file_path

if __name__ == "__main__":
    create_sample_dataset()