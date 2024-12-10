import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataPreprocessor:
    def __init__(self, scaling_method='standard', test_size=0.2, random_state=42):
        """
        Initialize the data preprocessor
        
        Args:
            scaling_method: 'standard' or 'minmax'
            test_size: proportion of data to use for testing
            random_state: random seed for reproducibility
        """
        self.scaling_method = scaling_method
        self.test_size = test_size
        self.random_state = random_state
        self.scalers = {}
        self.label_encoders = {}
        
    def load_dataset(self, dataset_name):
        """Load dataset using seaborn's data loading utility"""
        try:
            data = sns.load_dataset(dataset_name)
            print(f"Successfully loaded {dataset_name} dataset")
            return data
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values in the dataset
        
        Args:
            df: pandas DataFrame
            strategy: 'mean', 'median', 'mode', or 'drop'
        """
        if strategy == 'drop':
            return df.dropna()
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                if strategy == 'mean':
                    df[column].fillna(df[column].mean(), inplace=True)
                elif strategy == 'median':
                    df[column].fillna(df[column].median(), inplace=True)
            else:
                df[column].fillna(df[column].mode()[0], inplace=True)
        return df

    def encode_categorical_variables(self, df, columns=None):
        """
        Encode categorical variables using Label Encoding
        
        Args:
            df: pandas DataFrame
            columns: list of categorical columns to encode
        """
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns
            
        df_encoded = df.copy()
        for column in columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df_encoded[column] = self.label_encoders[column].fit_transform(df[column].astype(str))
            else:
                df_encoded[column] = self.label_encoders[column].transform(df[column].astype(str))
        return df_encoded

    def scale_features(self, df, columns=None):
        """
        Scale numerical features
        
        Args:
            df: pandas DataFrame
            columns: list of numerical columns to scale
        """
        if columns is None:
            columns = df.select_dtypes(include=['int64', 'float64']).columns
            
        df_scaled = df.copy()
        for column in columns:
            if column not in self.scalers:
                if self.scaling_method == 'standard':
                    self.scalers[column] = StandardScaler()
                else:
                    self.scalers[column] = MinMaxScaler()
                df_scaled[column] = self.scalers[column].fit_transform(df[[column]])
            else:
                df_scaled[column] = self.scalers[column].transform(df[[column]])
        return df_scaled

    def create_time_features(self, df, date_column):
        """
        Create time-based features from date column
        
        Args:
            df: pandas DataFrame
            date_column: name of the date column
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        df[f'{date_column}_year'] = df[date_column].dt.year
        df[f'{date_column}_month'] = df[date_column].dt.month
        df[f'{date_column}_day'] = df[date_column].dt.day
        df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
        df[f'{date_column}_quarter'] = df[date_column].dt.quarter
        
        return df

    def prepare_sequence_data(self, df, sequence_length, target_column):
        """
        Prepare sequential data for time series analysis
        
        Args:
            df: pandas DataFrame
            sequence_length: length of input sequences
            target_column: name of the target column
        """
        sequences = []
        targets = []
        
        for i in range(len(df) - sequence_length):
            sequence = df.iloc[i:(i + sequence_length)]
            target = df.iloc[i + sequence_length][target_column]
            sequences.append(sequence.values)
            targets.append(target)
            
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        return sequences, targets

    def split_data(self, X, y):
        """Split data into training and testing sets"""
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

    def prepare_dataset(self, data, target_column, sequence=False, sequence_length=10):
        """
        Main method to prepare dataset for training
        
        Args:
            data: pandas DataFrame
            target_column: name of the target column
            sequence: whether to prepare sequential data
            sequence_length: length of sequences if sequence=True
        """
        # Handle missing values
        data = self.handle_missing_values(data)
        
        # Create time features if datetime columns exist
        date_columns = data.select_dtypes(include=['datetime64']).columns
        for date_column in date_columns:
            data = self.create_time_features(data, date_column)
            data.drop(columns=[date_column], inplace=True)
        
        # Encode categorical variables
        data = self.encode_categorical_variables(data)
        
        # Scale features
        data = self.scale_features(data)
        
        if sequence:
            X, y = self.prepare_sequence_data(data, sequence_length, target_column)
        else:
            X = data.drop(columns=[target_column])
            y = data[target_column]
        
        return self.split_data(X, y) 