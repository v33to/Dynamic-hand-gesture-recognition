"""
gesture_classifier_seq.py

This module provides the Classifier class for sequential gesture recognition,
which uses sequences of 21 3D landmarks to predict dynamic hand gestures with 
unknown detection. It also includes the SequenceBuffer class for collecting
and managing landmark sequences with sliding window support for continuous recognition.

Usage:
    import gesture_classifier_seq

    classifier = gesture_classifier_seq.Classifier()
    sequence_buffer = gesture_classifier_seq.SequenceBuffer(
        min_frames=10, max_frames=20, sequence_length=10, feature_size=63
    )
    windows = sequence_buffer.get_sliding_windows(max_windows=7)
    result = classifier.classify_with_majority_voting(windows)

Classes:
    SequenceBuffer: Manages collection and normalization of landmark sequences with sliding windows.
    Classifier: Predict sequential hand gestures with confidence and uncertainty estimation.
"""
import time
import numpy as np
from collections import Counter
import tflite_runtime.interpreter as tflite

# Hardcoded constants
PROCESSING_DISPLAY_DURATION = 4.0

# Unknown detection thresholds
UNKNOWN_CONFIDENCE_THRESHOLD = 0.9
UNKNOWN_ENTROPY_THRESHOLD = 0.5


class SequenceBuffer:
    """Enhanced sequential buffer with sliding window support for continuous recognition"""
    
    def __init__(self, min_frames=10, max_frames=20, sequence_length=10, feature_size=63):
        """
        Initialize buffer for continuous gesture recognition
        
        Args:
            min_frames: Minimum frames required before processing
            max_frames: Maximum frames to capture before forcing processing
            sequence_length: Length of each sliding window
            feature_size: Size of flattened landmark features (21 landmarks * 3 = 63)
        """
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.buffer = []
        self.frame_count = 0
        
        self.sequence_reference_wrist = None
        self.sequence_reference_scale = None
        
        self.consecutive_nulls = 0
        self.max_consecutive_nulls = 2
        self.last_valid_landmarks = None
        self.pending_interpolations = []
        
        # Track which frames are null/interpolated
        self.frame_types = []  # 'valid', 'null', 'interpolated'
        
    def reset(self, reason="Unknown"):
        """Reset buffer to initial state"""
        self.buffer = []
        self.frame_types = []
        self.frame_count = 0
        self.sequence_reference_wrist = None
        self.sequence_reference_scale = None
        self.consecutive_nulls = 0
        self.last_valid_landmarks = None
        self.pending_interpolations = []
    
    def add_frame(self, landmarks):
        """Add a frame with landmarks (21, 3)"""
        # Perform pending interpolations
        if self.pending_interpolations and self.last_valid_landmarks is not None:
            current_normalized = self._normalize_landmarks(landmarks)
            num_missing = len(self.pending_interpolations)
            
            interpolated_frames = self._perform_linear_interpolation(
                self.last_valid_landmarks,
                current_normalized,
                num_missing
            )
            
            for idx, interp_frame in zip(self.pending_interpolations, interpolated_frames):
                self.buffer[idx] = interp_frame
                self.frame_types[idx] = 'interpolated'
            
            self.pending_interpolations = []
        
        self.consecutive_nulls = 0
        normalized = self._normalize_landmarks(landmarks)
        
        # If at max capacity, remove oldest frame
        if len(self.buffer) >= self.max_frames:
            self.buffer.pop(0)
            self.frame_types.pop(0)
        
        self.buffer.append(normalized)
        self.frame_types.append('valid')
        self.last_valid_landmarks = normalized
        
        self.frame_count = len(self.buffer)
        
        return True
    
    def add_null_frame(self):
        """
        Handle missing detection
        
        Returns:
            True: Continue normal operation
            False: Reset due to too many nulls and insufficient frames
            'process': Trigger processing (hand lost with sufficient frames)
        """
        self.consecutive_nulls += 1
        
        # If we have enough frames and hand is lost, trigger processing
        if self.consecutive_nulls > self.max_consecutive_nulls:
            if self.frame_count >= self.min_frames:
                # Signal that we should process (hand lost after sufficient frames)
                return 'process'
            else:
                # Not enough frames, just reset
                self.reset("Hand lost")
                return False
        
        if self.last_valid_landmarks is not None and len(self.buffer) > 0:
            placeholder = self.last_valid_landmarks.copy()
            
            # If at max capacity, remove oldest frame
            if len(self.buffer) >= self.max_frames:
                self.buffer.pop(0)
                self.frame_types.pop(0)
            
            self.buffer.append(placeholder)
            self.frame_types.append('null')
            
            # Mark this frame index for interpolation
            self.pending_interpolations.append(len(self.buffer) - 1)
            
            self.frame_count = len(self.buffer)
        
        return True
    
    def _normalize_landmarks(self, landmarks):
        """Normalize landmarks"""
        if len(self.buffer) == 0:
            self.sequence_reference_wrist = landmarks[0].copy()
            
            # Calculate scale based on max distance from wrist (scale-invariant)
            distances = np.linalg.norm(landmarks - landmarks[0], axis=1)
            hand_span = np.max(distances)
            
            if hand_span < 0.001:
                hand_span = 1.0
                
            self.sequence_reference_scale = hand_span
        
        normalized = landmarks - self.sequence_reference_wrist
        normalized = normalized / self.sequence_reference_scale
        
        return normalized.flatten().astype(np.float32)
    
    def _perform_linear_interpolation(self, start_landmarks, end_landmarks, num_steps):
        """Perform linear interpolation between two landmark sets"""
        interpolated = []
        for i in range(1, num_steps + 1):
            t = i / (num_steps + 1)
            interpolated_frame = start_landmarks + t * (end_landmarks - start_landmarks)
            interpolated.append(interpolated_frame)
        
        return interpolated
    
    def is_ready_for_processing(self):
        """Check if buffer has minimum frames for processing"""
        return self.frame_count >= self.min_frames
    
    def get_sliding_windows(self, max_windows=7):
        """
        Get up to max_windows sliding window sequences, prioritizing recent frames
        
        Args:
            max_windows: Maximum number of windows to return
            
        Returns:
            List of sliding window sequences (numpy arrays)
        """
        if not self.is_ready_for_processing():
            return []
        
        # Clean buffer: if over min_frames, remove null sequences
        cleaned_buffer = self._get_cleaned_buffer()
        
        if len(cleaned_buffer) < self.sequence_length:
            return []
        
        windows = []
        # Calculate how many windows are actually possible
        num_possible_windows = len(cleaned_buffer) - self.sequence_length + 1
        # Take minimum of what's possible and what's requested
        num_windows = min(num_possible_windows, max_windows)
        
        # Start from the most recent position and slide backwards
        for i in range(num_windows):
            start_idx = len(cleaned_buffer) - self.sequence_length - i
            if start_idx < 0:
                break
            
            window = cleaned_buffer[start_idx:start_idx + self.sequence_length]
            windows.append(np.array(window, dtype=np.float32))
        
        return windows
    
    def _get_cleaned_buffer(self):
        """Get buffer with null sequences removed if over minimum"""
        if self.frame_count <= self.min_frames:
            # Keep everything if at or below minimum
            return self.buffer
        
        # Remove null frames if we're over minimum
        cleaned = []
        for i, (frame, frame_type) in enumerate(zip(self.buffer, self.frame_types)):
            if frame_type != 'null':
                cleaned.append(frame)
        
        # Ensure we have at least min_frames
        if len(cleaned) < self.min_frames:
            return self.buffer
        
        return cleaned
    
    def get_status(self):
        """Get current buffer status"""
        null_count = sum(1 for ft in self.frame_types if ft == 'null')
        interpolated_count = sum(1 for ft in self.frame_types if ft == 'interpolated')
        
        return {
            'frame_count': self.frame_count,
            'min_frames': self.min_frames,
            'max_frames': self.max_frames,
            'is_ready': self.is_ready_for_processing(),
            'consecutive_nulls': self.consecutive_nulls,
            'null_count': null_count,
            'interpolated_count': interpolated_count,
            'has_pending_interp': len(self.pending_interpolations) > 0
        }


class Classifier:
    """Sequential gesture classifier with unknown detection and majority voting"""
    
    def __init__(self, model, external_delegate=None):
        if external_delegate:
            external_delegate = [tflite.load_delegate(external_delegate)]
            
        self._interpreter = tflite.Interpreter(
            model, experimental_delegates=external_delegate
        )
        self._interpreter.allocate_tensors()

        _out_details = self._interpreter.get_output_details()
        _in_details = self._interpreter.get_input_details()

        self._in_idx = _in_details[0]["index"]
        self._out_idx = _out_details[0]["index"]
        
        # Get sequence length from model input shape
        self.sequence_length = _in_details[0]['shape'][1]
        self.feature_size = _in_details[0]['shape'][2]

        # Unknown detection thresholds
        self.UNKNOWN_CONFIDENCE_THRESHOLD = UNKNOWN_CONFIDENCE_THRESHOLD
        self.UNKNOWN_ENTROPY_THRESHOLD = UNKNOWN_ENTROPY_THRESHOLD

        # Ignore the first invoke (Warm-up time)
        batch, seq_len, features = tuple(_in_details[0]["shape"].tolist())
        self._interpreter.set_tensor(
            self._in_idx, np.random.rand(batch, seq_len, features).astype("float32")
        )
        self._interpreter.invoke()

    def __call__(self, sequence):
        """
        Classify gesture from sequence of landmarks with unknown detection
        
        Args:
            sequence: (sequence_length, feature_size) array of normalized landmarks
            
        Returns:
            dict with:
                - predictions: probability array
                - max_confidence: highest probability
                - entropy: uncertainty measure
                - is_unknown: boolean flag
                - predicted_class: index of predicted class
        """
        # Prepare input data
        if len(sequence.shape) == 2:
            input_data = np.expand_dims(sequence, axis=0).astype("float32")
        else:
            input_data = sequence.astype("float32")

        # Run inference
        self._interpreter.set_tensor(self._in_idx, input_data)
        self._interpreter.invoke()
        output_data = self._interpreter.get_tensor(self._out_idx)
        
        predictions = output_data.squeeze()

        # Calculate confidence metrics
        max_confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)
        
        # Calculate entropy for uncertainty
        entropy = -np.sum(predictions * np.log(predictions + 1e-10))
        normalized_entropy = entropy / np.log(len(predictions))
        
        # Determine if unknown based on thresholds
        is_unknown = (max_confidence < self.UNKNOWN_CONFIDENCE_THRESHOLD) or \
                     (normalized_entropy > self.UNKNOWN_ENTROPY_THRESHOLD)
        
        return {
            'predictions': predictions,
            'max_confidence': float(max_confidence),
            'entropy': float(normalized_entropy),
            'is_unknown': bool(is_unknown),
            'predicted_class': int(predicted_class)
        }
    
    def classify_with_majority_voting(self, windows):
        """
        Classify multiple sliding windows and use majority voting
        
        Args:
            windows: List of window sequences to classify
            
        Returns:
            dict with:
                - predicted_class: final predicted class
                - max_confidence: confidence of the prediction
                - entropy: entropy of the prediction
                - is_unknown: whether prediction is unknown
                - num_windows: total number of windows analyzed
                - num_valid: number of valid (non-unknown) predictions
                - voting_results: vote counts per class (None if all unknown)
        """
        if not windows:
            return None
        
        valid_predictions = []
        all_results = []
        
        for window in windows:
            # Get prediction
            result = self(window)
            all_results.append(result)
            
            # Only consider predictions that pass thresholds
            if not result['is_unknown']:
                valid_predictions.append({
                    'class': result['predicted_class'],
                    'confidence': result['max_confidence'],
                    'entropy': result['entropy']
                })
        
        # If no valid predictions, return unknown with raw class
        if not valid_predictions:
            if all_results:
                # We had predictions but all were unknown or low confidence
                best_result = max(all_results, key=lambda x: x['max_confidence'])
                return {
                    'predicted_class': best_result['predicted_class'],
                    'max_confidence': best_result['max_confidence'],
                    'entropy': best_result['entropy'],
                    'is_unknown': True,
                    'num_windows': len(windows),
                    'num_valid': 0,
                    'voting_results': None
                }
            else:
                # No predictions at all (shouldn't happen)
                return {
                    'predicted_class': -1,
                    'max_confidence': 0.0,
                    'entropy': 0.0,
                    'is_unknown': True,
                    'num_windows': len(windows),
                    'num_valid': 0,
                    'voting_results': None
                }
        
        # Perform majority voting
        class_votes = [p['class'] for p in valid_predictions]
        vote_counts = Counter(class_votes)
        
        # Get the most common class(es)
        max_votes = max(vote_counts.values())
        top_classes = [cls for cls, count in vote_counts.items() if count == max_votes]
        
        # If tie, use highest confidence among tied classes
        if len(top_classes) > 1:
            tied_predictions = [p for p in valid_predictions if p['class'] in top_classes]
            final_prediction = max(tied_predictions, key=lambda x: x['confidence'])
        else:
            # Get best prediction for the winning class
            class_predictions = [p for p in valid_predictions if p['class'] == top_classes[0]]
            final_prediction = max(class_predictions, key=lambda x: x['confidence'])
        
        return {
            'predicted_class': final_prediction['class'],
            'max_confidence': final_prediction['confidence'],
            'entropy': final_prediction['entropy'],
            'is_unknown': False,
            'num_windows': len(windows),
            'num_valid': len(valid_predictions),
            'voting_results': dict(vote_counts)
        }