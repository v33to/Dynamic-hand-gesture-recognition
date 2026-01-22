import os
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from data_collector.hand_tracker import HandTracker
import time
from collections import Counter

path = os.getcwd()

PALM_MODEL = path + "/models/palm_detection_full_quant.tflite"
LANDMARK_MODEL = path + "/models/hand_landmark_full_quant.tflite"
ANCHORS = path + "/models/anchors.csv"

GESTURE_MODEL = path + "/trained_models/gesture_conv1d_quant.tflite"
LABELS_FILE = path + "/trained_models/gesture_labels.txt"

SEQUENCE_LENGTH = 10  # Size of each sliding window
MIN_FRAMES = 10  # Minimum frames before processing
MAX_FRAMES = 20  # Maximum frames to capture
MAX_SLIDING_WINDOWS = 7  # Maximum number of sliding windows to analyze

UNKNOWN_CONFIDENCE_THRESHOLD = 0.9
UNKNOWN_ENTROPY_THRESHOLD = 0.5

PROCESSING_DISPLAY_DURATION = 4.0  # Duration to display results after processing

GESTURE_NAMES = {
    0: "Swipe Up",
    1: "Swipe Down",
    2: "Swipe Left",
    3: "Swipe Right",
    4: "Zoom In",
    5: "Zoom Out",
    6: "Rotate Clockwise",
    7: "Rotate Counter Clockwise",
    8: "Unknown"
}


class SequenceBuffer:
    """Enhanced sequential buffer with sliding window support"""
    
    def __init__(self, min_frames=MIN_FRAMES, max_frames=MAX_FRAMES, 
                 sequence_length=SEQUENCE_LENGTH, feature_size=63):
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
        """Handle missing detection"""
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
        """Normalize landmarks (same as data collection)"""
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
    
    def get_sliding_windows(self, max_windows=MAX_SLIDING_WINDOWS):
        """
        Get up to max_windows sliding window sequences, prioritizing recent frames
        Returns all available windows up to max_windows. If we have fewer frames, we return fewer windows.
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


class Gesture1D_ConvClassifier:
    
    def __init__(self, model_path, labels_path=None, 
                 unknown_confidence_threshold=UNKNOWN_CONFIDENCE_THRESHOLD,
                 unknown_entropy_threshold=UNKNOWN_ENTROPY_THRESHOLD):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.unknown_confidence_threshold = unknown_confidence_threshold
        self.unknown_entropy_threshold = unknown_entropy_threshold
        
        # Get sequence length from model input shape
        self.sequence_length = self.input_details[0]['shape'][1]
        self.feature_size = self.input_details[0]['shape'][2]
        
        self.labels = []
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        
        print(f"Loaded 1D_Conv gesture classifier:")
        print(f"  Input shape: {self.input_details[0]['shape']}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Feature size: {self.feature_size}")
        print(f"  Classes: {len(self.labels)}")
        print(f"  Unknown confidence threshold: {unknown_confidence_threshold:.2f}")
        print(f"  Unknown entropy threshold: {unknown_entropy_threshold:.2f}")
        print(f"  Min frames: {MIN_FRAMES}, Max frames: {MAX_FRAMES}")
        print(f"  Max sliding windows: {MAX_SLIDING_WINDOWS}")
        
    def __call__(self, sequence):
        """Classify gesture from sequence of landmarks"""
        if len(sequence.shape) == 2:
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
        else:
            input_data = sequence.astype(np.float32)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        predictions = output_data[0]
        
        # Calculate confidence metrics
        max_confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)
        
        # Calculate entropy for uncertainty
        entropy = -np.sum(predictions * np.log(predictions + 1e-10))
        normalized_entropy = entropy / np.log(len(predictions))
        
        is_unknown = (max_confidence < self.unknown_confidence_threshold) or \
                     (normalized_entropy > self.unknown_entropy_threshold)
        
        return {
            'predictions': predictions,
            'max_confidence': max_confidence,
            'entropy': normalized_entropy,
            'is_unknown': is_unknown,
            'predicted_class': predicted_class
        }
    
    def classify_with_majority_voting(self, windows):
        """Classify multiple sliding windows and use majority voting"""
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


class GestureRecognitionApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("1D_Conv Gesture Recognition - Sliding Window Mode")
        self.root.geometry("800x700")
        
        self.cap = None
        self.tracker = None
        self.classifier = None
        self.sequence_buffer = None
        self.current_frame = None
        self.is_running = False
        
        # Processing state
        self.is_processing = False
        self.processing_start_time = None
        self.last_prediction_result = None
        
        self.camera_index = tk.IntVar(value=0)
        self.available_cameras = self._detect_cameras()
        
        self._setup_ui()
        self._initialize_models()
        
    def _detect_cameras(self):
        """Detect available cameras"""
        available = []
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available if available else [0]
    
    def _setup_ui(self):
        """Setup the user interface"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=0)
        
        # Video frame
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        center_frame = ttk.Frame(video_frame)
        center_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        center_frame.columnconfigure(0, weight=1)
        center_frame.rowconfigure(0, weight=1)
        
        self.video_label = ttk.Label(center_frame, background='black')
        self.video_label.place(relx=0.5, rely=0.5, anchor='center')
        
        # Control panel
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.grid(row=1, column=0, pady=5)
        
        # Camera selection
        camera_frame = ttk.LabelFrame(control_frame, text="Camera Selection", padding="10")
        camera_frame.pack()
        
        ttk.Label(camera_frame, text="Select Camera:").grid(row=0, column=0, sticky=tk.W, pady=2, padx=5)
        camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_index, 
                                     values=self.available_cameras, state="readonly", width=10)
        camera_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
        camera_combo.bind('<<ComboboxSelected>>', self._change_camera)
        
    def _initialize_models(self):
        """Initialize hand tracker and gesture classifier"""
        try:
            self.tracker = HandTracker(
                palm_detection_model=PALM_MODEL,
                hand_landmark_model=LANDMARK_MODEL,
                anchors=ANCHORS,
                num_hands=1
            )
            
            self.classifier = Gesture1D_ConvClassifier(
                GESTURE_MODEL,
                LABELS_FILE,
                unknown_confidence_threshold=UNKNOWN_CONFIDENCE_THRESHOLD,
                unknown_entropy_threshold=UNKNOWN_ENTROPY_THRESHOLD
            )
            
            self.sequence_buffer = SequenceBuffer(
                min_frames=MIN_FRAMES,
                max_frames=MAX_FRAMES,
                sequence_length=self.classifier.sequence_length,
                feature_size=self.classifier.feature_size
            )
            
            self._open_camera()
            self.is_running = True
            self._update_frame()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize models:\n{str(e)}\n\nCheck file paths.")
            self.root.quit()
    
    def _open_camera(self):
        """Open the camera"""
        if self.cap is not None:
            self.cap.release()
        
        camera_idx = self.camera_index.get()
        self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_idx}")
        
        # Set resolution (try high res first)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Read actual resolution achieved
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # If Full HD not supported, try HD
        if actual_width < 1920:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    def _change_camera(self, event=None):
        """Change camera source"""
        self._open_camera()
        self.sequence_buffer.reset("Camera changed")
        self.is_processing = False
        self.processing_start_time = None
    
    def _draw_landmarks(self, landmarks, frame):
        """Draw hand landmarks on frame"""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        # Draw connections
        for connection in connections:
            x0, y0 = landmarks[connection[0]][:2]
            x1, y1 = landmarks[connection[1]][:2]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
        
        # Draw points
        for i, point in enumerate(landmarks):
            x, y = point[:2]
            color = (0, 255, 0) if i in [0, 1, 5, 9, 13, 17] else (0, 200, 255)
            cv2.circle(frame, (int(x), int(y)), 6, color, -1)
    
    def _draw_overlay(self, frame, status, prediction_result=None):
        """Draw overlay information at bottom of frame"""
        h, w = frame.shape[:2]
        overlay_height = 80
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - overlay_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Progress bar
        bar_width = 250
        bar_height = 20
        bar_x = 20
        bar_y = h - overlay_height + 15
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        if self.is_processing:
            # Show processing countdown
            elapsed = time.time() - self.processing_start_time
            progress = min(1.0, elapsed / PROCESSING_DISPLAY_DURATION)
            fill_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (255, 165, 0), -1)
        else:
            # Show buffer fill progress
            progress = min(1.0, status['frame_count'] / status['max_frames'])
            fill_width = int(bar_width * progress)
            
            if status['is_ready']:
                color = (0, 255, 0)  # Green when ready
            else:
                color = (100, 100, 100)  # Gray when collecting
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
        
        # Buffer status text
        buffer_text_y = bar_y + bar_height + 20
        buffer_text = f"Buffer: {status['frame_count']}/{status['max_frames']}"
        if status['null_count'] > 0:
            buffer_text += f" (Nulls: {status['null_count']})"
        cv2.putText(frame, buffer_text, (bar_x, buffer_text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Prediction/status text
        text_x = bar_x + bar_width + 30
        text_y_line1 = h - overlay_height + 25
        text_y_line2 = h - overlay_height + 50
        text_y_line3 = h - overlay_height + 70
        
        if self.is_processing and prediction_result:
            elapsed = time.time() - self.processing_start_time
            remaining = max(0, PROCESSING_DISPLAY_DURATION - elapsed)
            
            if prediction_result['is_unknown']:
                # Display raw class for unknown predictions
                raw_label = GESTURE_NAMES.get(prediction_result['predicted_class'], 
                                              f"Class {prediction_result['predicted_class']}")
                result_text = f"UNKNOWN ({raw_label})"
                cv2.putText(frame, result_text, (text_x, text_y_line1), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                conf_text = f"C:{prediction_result['max_confidence']:.0%} | E:{prediction_result['entropy']:.2f}"
                cv2.putText(frame, conf_text, (text_x, text_y_line2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            else:
                label = GESTURE_NAMES.get(prediction_result['predicted_class'], 
                                         f"Class {prediction_result['predicted_class']}")
                result_text = f"{label}"
                cv2.putText(frame, result_text, (text_x, text_y_line1), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                conf_text = f"Confidence: {prediction_result['max_confidence']:.0%} | Entropy: {prediction_result['entropy']:.2f}"
                cv2.putText(frame, conf_text, (text_x, text_y_line2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            if prediction_result.get('voting_results'):
                voting_text = f"Windows: {prediction_result['num_valid']}/{prediction_result['num_windows']} valid"
                cv2.putText(frame, voting_text, (text_x, text_y_line3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                
        elif status['consecutive_nulls'] > 0:
            warning_text = f"Missing detection ({status['consecutive_nulls']}/{self.sequence_buffer.max_consecutive_nulls})"
            cv2.putText(frame, warning_text, (text_x, text_y_line1), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        elif status['is_ready']:
            ready_text = "READY - Collecting data..."
            cv2.putText(frame, ready_text, (text_x, text_y_line1), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            collecting_text = f"Collecting... ({status['frame_count']}/{status['min_frames']} minimum)"
            cv2.putText(frame, collecting_text, (text_x, text_y_line1), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    
    def _update_frame(self):
        """Update video frame with recognition"""
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            # Check if we're in processing/display mode
            if self.is_processing:
                elapsed = time.time() - self.processing_start_time
                if elapsed >= PROCESSING_DISPLAY_DURATION:
                    # Processing display period is over, reset buffer and resume
                    self.sequence_buffer.reset("Processing complete")
                    self.is_processing = False
                    self.processing_start_time = None
                    self.last_prediction_result = None
            
            # Continue capturing frames if not processing
            if not self.is_processing:
                detections = self.tracker(frame)
                
                if detections:
                    for landmarks, hand_bbox, handedness in detections:
                        self._draw_landmarks(landmarks, frame)
                        self.sequence_buffer.add_frame(landmarks)
                else:
                    result = self.sequence_buffer.add_null_frame()
                    
                    # Check if hand was lost and we should process
                    if result == 'process':
                        # Hand lost after capturing sufficient frames
                        # Get sliding windows (this will automatically clean null frames)
                        windows = self.sequence_buffer.get_sliding_windows(MAX_SLIDING_WINDOWS)
                        
                        if windows:
                            # Perform classification with majority voting
                            prediction_result = self.classifier.classify_with_majority_voting(windows)
                            
                            if prediction_result:
                                self.last_prediction_result = prediction_result
                                self.is_processing = True
                                self.processing_start_time = time.time()
                        else:
                            # No valid windows after cleaning, just reset
                            self.sequence_buffer.reset("Insufficient valid frames")
                
                status = self.sequence_buffer.get_status()
                
                # Also process if we reach MAX_FRAMES (to prevent indefinite capture)
                if status['frame_count'] >= MAX_FRAMES and not self.is_processing:
                    windows = self.sequence_buffer.get_sliding_windows(MAX_SLIDING_WINDOWS)
                    
                    if windows:
                        prediction_result = self.classifier.classify_with_majority_voting(windows)
                        
                        if prediction_result:
                            self.last_prediction_result = prediction_result
                            self.is_processing = True
                            self.processing_start_time = time.time()
            
            status = self.sequence_buffer.get_status()
            self._draw_overlay(frame, status, self.last_prediction_result)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            parent_width = self.video_label.master.winfo_width()
            parent_height = self.video_label.master.winfo_height()
            
            if parent_width > 1 and parent_height > 1:
                img_ratio = img.width / img.height
                parent_ratio = parent_width / parent_height
                
                if img_ratio > parent_ratio:
                    new_width = parent_width - 20
                    new_height = int(new_width / img_ratio)
                else:
                    new_height = parent_height - 20
                    new_width = int(new_height * img_ratio)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.root.after(10, self._update_frame)
    
    def _cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureRecognitionApp(root)
    
    def on_closing():
        app._cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()