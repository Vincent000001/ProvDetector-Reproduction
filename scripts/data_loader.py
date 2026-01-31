"""
Data loader for provenance graph CSV files
"""
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import glob


class ProvenanceDataLoader:
    """Load and preprocess provenance graph data from CSV files"""
    
    def __init__(self, benign_dir: str = "benign", malicious_dir: str = "malicious"):
        self.benign_dir = benign_dir
        self.malicious_dir = malicious_dir
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load a single CSV file"""
        try:
            df = pd.read_csv(filepath)
            # Add file identifier
            df['file_id'] = os.path.basename(filepath)
            
            # Handle Linux format (may not have pid1 column)
            if 'pid1' not in df.columns:
                df['pid1'] = ''
            
            # Ensure all required columns exist
            required_columns = ['sourceId', 'sourceType', 'destinationId', 
                              'destinationType', 'action', 'processName', 'timestamp']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ''
            
            # Fill missing pid0 if needed
            if 'pid0' not in df.columns:
                df['pid0'] = ''
            
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
    
    def load_all_files(self) -> Tuple[List[pd.DataFrame], List[int]]:
        """
        Load all CSV files and return dataframes with labels
        Returns: (dataframes, labels) where 0=benign, 1=malicious
        """
        dataframes = []
        labels = []
        
        # Load benign files
        benign_files = glob.glob(os.path.join(self.benign_dir, "*.csv"))
        print(f"Found {len(benign_files)} benign files")
        for filepath in benign_files:
            df = self.load_csv(filepath)
            if not df.empty:
                dataframes.append(df)
                labels.append(0)  # Benign
        
        # Load malicious files
        malicious_files = glob.glob(os.path.join(self.malicious_dir, "*.csv"))
        print(f"Found {len(malicious_files)} malicious files")
        for filepath in malicious_files:
            df = self.load_csv(filepath)
            if not df.empty:
                dataframes.append(df)
                labels.append(1)  # Malicious
        
        print(f"Total loaded: {len(dataframes)} files ({sum(1 for l in labels if l == 0)} benign, {sum(1 for l in labels if l == 1)} malicious)")
        return dataframes, labels
    
    def get_file_statistics(self, df: pd.DataFrame) -> Dict:
        """Get basic statistics about a provenance graph"""
        stats = {}
        
        if df.empty:
            return stats
        
        # Basic counts
        stats['total_events'] = len(df)
        stats['unique_processes'] = df['processName'].nunique()
        stats['unique_actions'] = df['action'].nunique()
        stats['unique_sources'] = df['sourceId'].nunique()
        stats['unique_destinations'] = df['destinationId'].nunique()
        
        # Action type counts
        action_counts = df['action'].value_counts().to_dict()
        for action in ['start', 'read', 'write', 'connect']:
            stats[f'action_{action}_count'] = action_counts.get(action, 0)
        
        # Process type distribution
        stats['process_to_process'] = len(df[(df['sourceType'] == 'process') & (df['destinationType'] == 'process')])
        stats['file_to_process'] = len(df[(df['sourceType'] == 'file') & (df['destinationType'] == 'process')])
        stats['process_to_file'] = len(df[(df['sourceType'] == 'process') & (df['destinationType'] == 'file')])
        stats['process_to_socket'] = len(df[(df['sourceType'] == 'process') & (df['destinationType'] == 'socket')])
        
        # Temporal features
        if 'timestamp' in df.columns:
            timestamps = df[df['timestamp'] > 0]['timestamp']
            if len(timestamps) > 0:
                stats['duration'] = (timestamps.max() - timestamps.min()) / 1e9  # Convert to seconds
                stats['avg_time_between_events'] = stats['duration'] / max(len(timestamps) - 1, 1)
            else:
                stats['duration'] = 0
                stats['avg_time_between_events'] = 0
        
        return stats

