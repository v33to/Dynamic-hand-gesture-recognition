import h5py
import numpy as np
from collections import defaultdict


class DatasetManager:

    def __init__(self, sequence_length=10):
        self.data = []  # List of sequences (each sequence is a list of landmark arrays)
        self.labels = []  # List of gesture IDs
        self.sequence_length = sequence_length
        self.metadata = {
            'num_landmarks': 21,
            'landmark_dims': 3,  # x, y, z
            'feature_size': 63,  # 21 landmarks * 3 coordinates
            'sequence_length': sequence_length
        }
    
    def add_sequence(self, sequence, gesture_id):
        """Add a new sequence to the dataset"""
        if len(sequence) != self.sequence_length:
            raise ValueError(f"Expected sequence of length {self.sequence_length}, "
                           f"got {len(sequence)}")
        
        processed_sequence = []
        for frame_landmarks in sequence:
            if frame_landmarks.shape[0] != self.metadata['feature_size']:
                raise ValueError(f"Expected {self.metadata['feature_size']} features per frame, "
                               f"got {frame_landmarks.shape[0]}")
            processed_sequence.append(frame_landmarks.astype(np.float32))
        
        self.data.append(processed_sequence)
        self.labels.append(int(gesture_id))
    
    def delete_sequence(self, sequence_idx):
        """Delete a sequence from the dataset"""
        if 0 <= sequence_idx < len(self.data):
            del self.data[sequence_idx]
            del self.labels[sequence_idx]
            return True
        return False
    
    def get_statistics(self):
        """Get statistics about the collected dataset"""
        stats = {
            'total_sequences': len(self.data),
            'num_classes': len(set(self.labels)) if self.labels else 0,
            'sequences_per_class': defaultdict(int),
            'sequence_length': self.sequence_length
        }
        
        for label in self.labels:
            stats['sequences_per_class'][label] += 1
        
        return stats
    
    def get_class_sequences(self, class_id):
        """Get all sequence indices for a specific class"""
        return [idx for idx, label in enumerate(self.labels) if label == class_id]
    
    def save_to_h5(self, filename):
        """Save the dataset to an HDF5 file"""
        if not self.data:
            raise ValueError("No data to save")
        
        data_array = np.array(self.data, dtype=np.float32)
        labels_array = np.array(self.labels, dtype=np.int32)
        
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('sequences', data=data_array, compression='gzip')
            hf.create_dataset('labels', data=labels_array, compression='gzip')
            
            # Add metadata
            hf.attrs['total_sequences'] = len(self.data)
            hf.attrs['num_classes'] = len(set(self.labels))
            hf.attrs['num_landmarks'] = self.metadata['num_landmarks']
            hf.attrs['landmark_dims'] = self.metadata['landmark_dims']
            hf.attrs['feature_size'] = self.metadata['feature_size']
            hf.attrs['sequence_length'] = self.metadata['sequence_length']
            
            # Add class distribution
            stats = self.get_statistics()
            for class_id, count in stats['sequences_per_class'].items():
                hf.attrs[f'class_{class_id}_count'] = count
        
        print(f"Dataset saved to {filename}")
        print(f"Total sequences: {len(self.data)}")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Number of classes: {len(set(self.labels))}")
    
    def load_from_h5(self, filename):
        """Load dataset from an HDF5 file"""
        with h5py.File(filename, 'r') as hf:
            sequences = hf['sequences'][:]
            self.labels = list(hf['labels'][:])
            self.data = [list(seq) for seq in sequences]
            
            # Load metadata
            self.metadata['num_landmarks'] = hf.attrs['num_landmarks']
            self.metadata['landmark_dims'] = hf.attrs['landmark_dims']
            self.metadata['feature_size'] = hf.attrs['feature_size']
            self.metadata['sequence_length'] = hf.attrs['sequence_length']
            self.sequence_length = hf.attrs['sequence_length']
        
        print(f"Dataset loaded from {filename}")
        print(f"Total sequences: {len(self.data)}")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Number of classes: {len(set(self.labels))}")
    
    def clear(self):
        """Clear all collected data"""
        self.data = []
        self.labels = []
    
    def get_data_for_training(self):
        """Get data in format ready for training"""
        X = np.array(self.data, dtype=np.float32)
        y = np.array(self.labels, dtype=np.int32)
        return X, y