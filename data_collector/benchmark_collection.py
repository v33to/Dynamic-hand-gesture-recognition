import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import h5py
from hand_tracker import HandTracker

path = os.getcwd()

PALM_MODEL = path + "/models/palm_detection_full_quant.tflite"
LANDMARK_MODEL = path + "/models/hand_landmark_full_quant.tflite"
ANCHORS = path + "/models/anchors.csv"

SEQUENCE_LENGTH = 10
MIN_FRAMES = 10
MAX_FRAMES = 20
MAX_SLIDING_WINDOWS = 7

GESTURE_NAMES = {
    0: "Swipe Up",
    1: "Swipe Down",
    2: "Swipe Left",
    3: "Swipe Right",
    4: "Zoom In",
    5: "Zoom Out",
    6: "Rotate Clockwise",
    7: "Rotate Counter Clockwise",
    8: "Unknown/Natural Movement"
}


def show_dataset_selector():
    """Show dialog to choose between creating new or loading existing dataset"""
    dialog = tk.Tk()
    dialog.title("Dataset Selection")
    dialog.geometry("500x250")
    dialog.resizable(False, False)
    
    result = {'mode': None, 'dataset_path': None}
    
    def center_window():
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_new():
        result['mode'] = 'new'
        result['dataset_path'] = 'datasets/continuous_gestures_labeled.h5'
        dialog.destroy()
    
    def load_existing():
        filepath = filedialog.askopenfilename(
            title="Select Dataset File",
            initialdir="datasets",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if filepath:
            result['mode'] = 'load'
            result['dataset_path'] = filepath
            dialog.destroy()
    
    def cancel():
        result['mode'] = None
        dialog.destroy()
    
    main_frame = ttk.Frame(dialog, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    title_label = ttk.Label(main_frame, 
                           text="Continuous Gesture Dataset Collection",
                           font=('Arial', 14, 'bold'))
    title_label.pack(pady=(0, 10))
    
    info_label = ttk.Label(main_frame,
                          text="Choose whether to create a new dataset or load an existing one:",
                          wraplength=450)
    info_label.pack(pady=(0, 20))
    
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=10)
    
    new_btn = ttk.Button(button_frame, text="Create New Dataset", 
                        command=create_new, width=20)
    new_btn.pack(side=tk.LEFT, padx=5)
    
    load_btn = ttk.Button(button_frame, text="Load Existing Dataset", 
                         command=load_existing, width=20)
    load_btn.pack(side=tk.LEFT, padx=5)
    
    cancel_btn = ttk.Button(main_frame, text="Cancel", 
                           command=cancel, width=15)
    cancel_btn.pack(pady=(20, 0))
    
    center_window()
    dialog.mainloop()
    
    return result


class SequenceBuffer:
    """Sequential buffer with sliding window support"""
    
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
        
        self.frame_types = []
        
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
        
        if self.consecutive_nulls > self.max_consecutive_nulls:
            if self.frame_count >= self.min_frames:
                return 'process'
            else:
                self.reset("Hand lost")
                return False
        
        if self.last_valid_landmarks is not None and len(self.buffer) > 0:
            placeholder = self.last_valid_landmarks.copy()
            
            if len(self.buffer) >= self.max_frames:
                self.buffer.pop(0)
                self.frame_types.pop(0)
            
            self.buffer.append(placeholder)
            self.frame_types.append('null')
            
            self.pending_interpolations.append(len(self.buffer) - 1)
            
            self.frame_count = len(self.buffer)
        
        return True
    
    def _normalize_landmarks(self, landmarks):
        """Normalize landmarks"""
        if len(self.buffer) == 0:
            self.sequence_reference_wrist = landmarks[0].copy()
            
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
        """Get up to max_windows sliding window sequences"""
        if not self.is_ready_for_processing():
            return []
        
        cleaned_buffer = self._get_cleaned_buffer()
        
        if len(cleaned_buffer) < self.sequence_length:
            return []
        
        windows = []
        num_possible_windows = len(cleaned_buffer) - self.sequence_length + 1
        num_windows = min(num_possible_windows, max_windows)
        
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
            return self.buffer
        
        cleaned = []
        for i, (frame, frame_type) in enumerate(zip(self.buffer, self.frame_types)):
            if frame_type != 'null':
                cleaned.append(frame)
        
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


class WindowReviewDialog:
    """Dialog to review and label continuous gesture windows"""
    
    def __init__(self, parent, windows):
        self.result = None
        self.selected_class = None
        self.current_frame = 0
        self.current_window_idx = 0
        self.is_playing = True
        self.play_speed = 100
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Label Continuous Gesture - {len(windows)} Windows")
        self.dialog.geometry("1000x800")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.windows = windows
        self.num_windows = len(windows)
        
        self.class_var = tk.StringVar(value="8")  # Default to Unknown
        
        self._setup_ui()
        self._center_window()
        self._start_playback()
        
    def _center_window(self):
        """Center dialog on screen"""
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
        
    def _setup_ui(self):
        """Setup review dialog UI"""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="Continuous Gesture Information", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        info_text = f"Total Sliding Windows: {self.num_windows}\n"
        info_text += f"This represents one continuous gesture captured from the camera.\n"
        info_text += f"All {self.num_windows} windows will be labeled with the same class."
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack()
        
        # Window selector
        window_frame = ttk.LabelFrame(main_frame, text="Window Viewer", padding="10")
        window_frame.pack(fill=tk.X, pady=(0, 10))
        
        window_row = ttk.Frame(window_frame)
        window_row.pack(fill=tk.X)
        
        ttk.Label(window_row, text="View Window:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.window_var = tk.IntVar(value=1)
        self.window_spinbox = ttk.Spinbox(window_row, from_=1, to=self.num_windows, 
                                          textvariable=self.window_var, width=10,
                                          command=self._change_window)
        self.window_spinbox.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(window_row, text=f"of {self.num_windows}").pack(side=tk.LEFT)
        
        # Canvas frame
        canvas_frame = ttk.LabelFrame(main_frame, text="Sequence Preview", padding="10")
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.canvas = tk.Canvas(canvas_frame, bg='black', height=350)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Playback controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack()
        
        ttk.Button(btn_frame, text="Prev", command=self._prev_frame).pack(side=tk.LEFT, padx=2)
        self.play_button = ttk.Button(btn_frame, text="Pause", command=self._toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Next", command=self._next_frame).pack(side=tk.LEFT, padx=2)
        
        self.frame_label = ttk.Label(control_frame, text=f"Frame: 1 / {SEQUENCE_LENGTH}")
        self.frame_label.pack(pady=5)
        
        # Class selection frame
        class_frame = ttk.LabelFrame(main_frame, text="Label This Gesture", padding="10")
        class_frame.pack(fill=tk.X, pady=(0, 10))
        
        class_row = ttk.Frame(class_frame)
        class_row.pack(fill=tk.X)
        
        ttk.Label(class_row, text="Class ID:").pack(side=tk.LEFT, padx=(0, 5))
        
        class_entry = ttk.Entry(class_row, textvariable=self.class_var, width=5)
        class_entry.pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(class_row, text="Quick Select:").pack(side=tk.LEFT, padx=(0, 5))
        
        quick_names = {
            0: "Swipe Up", 1: "Swipe Down", 2: "Swipe Left", 3: "Swipe Right",
            4: "Zoom In", 5: "Zoom Out", 6: "Rot CW", 7: "Rot CCW", 8: "Unknown"
        }
        
        for class_id in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            btn = ttk.Button(class_row, text=quick_names[class_id], 
                           command=lambda cid=class_id: self._quick_select_class(cid))
            btn.pack(side=tk.LEFT, padx=2)
        
        # Decision buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Save Gesture", 
                  command=self._save_gesture).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(button_frame, text="Discard", 
                  command=self._discard_gesture).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.dialog.bind('<Left>', lambda e: self._prev_frame())
        self.dialog.bind('<Right>', lambda e: self._next_frame())
        self.dialog.bind('<space>', lambda e: self._toggle_play())
        self.dialog.bind('<Return>', lambda e: self._save_gesture())
        self.dialog.bind('<Escape>', lambda e: self._discard_gesture())
        
    def _quick_select_class(self, class_id):
        """Quick select a class"""
        self.class_var.set(str(class_id))
        
    def _change_window(self):
        """Change which window is being viewed"""
        try:
            new_idx = int(self.window_var.get()) - 1
            if 0 <= new_idx < self.num_windows:
                self.current_window_idx = new_idx
                self.current_frame = 0
                self._visualize_frame()
        except ValueError:
            pass
        
    def _start_playback(self):
        """Start automatic playback"""
        self._visualize_frame()
        if self.is_playing:
            self.dialog.after(self.play_speed, self._play_loop)
    
    def _play_loop(self):
        """Playback loop"""
        if self.is_playing:
            self._next_frame()
            self.dialog.after(self.play_speed, self._play_loop)
    
    def _toggle_play(self):
        """Toggle playback"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.config(text="Pause")
            self._play_loop()
        else:
            self.play_button.config(text="Play")
    
    def _next_frame(self):
        """Go to next frame"""
        self.current_frame = (self.current_frame + 1) % SEQUENCE_LENGTH
        self._visualize_frame()
    
    def _prev_frame(self):
        """Go to previous frame"""
        self.current_frame = (self.current_frame - 1) % SEQUENCE_LENGTH
        self._visualize_frame()
    
    def _visualize_frame(self):
        """Visualize current frame"""
        self.canvas.delete("all")
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        current_window = self.windows[self.current_window_idx]
        frame_landmarks = current_window[self.current_frame].reshape(21, 3)
        landmarks_2d = frame_landmarks[:, :2]
        
        # Calculate global bounds for consistent scaling
        all_frames = current_window.reshape(-1, 21, 3)[:, :, :2]
        global_min_x, global_min_y = all_frames.reshape(-1, 2).min(axis=0)
        global_max_x, global_max_y = all_frames.reshape(-1, 2).max(axis=0)
        
        padding = 40
        range_x = global_max_x - global_min_x
        range_y = global_max_y - global_min_y
        
        if range_x == 0:
            range_x = 1
        if range_y == 0:
            range_y = 1
        
        max_range = max(range_x, range_y)
        scale = min(canvas_width - 2 * padding, canvas_height - 2 * padding) / max_range
        
        center_x = canvas_width / 2
        center_y = canvas_height / 2
        global_center_x = (global_min_x + global_max_x) / 2
        global_center_y = (global_min_y + global_max_y) / 2
        
        scaled_landmarks = []
        for x, y in landmarks_2d:
            canvas_x = center_x + (x - global_center_x) * scale
            canvas_y = center_y + (y - global_center_y) * scale
            scaled_landmarks.append((canvas_x, canvas_y))
        
        # Draw ghost frames
        if self.current_frame > 0:
            for frame_idx in range(max(0, self.current_frame - 3), self.current_frame):
                prev_frame = current_window[frame_idx].reshape(21, 3)[:, :2]
                prev_scaled = []
                for x, y in prev_frame:
                    canvas_x = center_x + (x - global_center_x) * scale
                    canvas_y = center_y + (y - global_center_y) * scale
                    prev_scaled.append((canvas_x, canvas_y))
                
                alpha = 50 + 50 * (frame_idx - max(0, self.current_frame - 3)) / max(1, self.current_frame - max(0, self.current_frame - 3))
                gray = int(alpha)
                color = f'#{gray:02x}{gray:02x}{gray:02x}'
                
                for x, y in prev_scaled:
                    self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline=color)
        
        # Hand skeleton connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        for idx1, idx2 in connections:
            x1, y1 = scaled_landmarks[idx1]
            x2, y2 = scaled_landmarks[idx2]
            self.canvas.create_line(x1, y1, x2, y2, fill='#1E90FF', width=3)
        
        # Draw landmarks
        palm_joints = [0, 1, 2, 5, 9, 13, 17]
        for idx, (x, y) in enumerate(scaled_landmarks):
            if idx == 0:
                color = '#FF0000'
                size = 10
            elif idx in palm_joints:
                color = '#FFA500'
                size = 8
            else:
                color = '#00FF00'
                size = 6
            
            self.canvas.create_oval(x - size, y - size, x + size, y + size,
                                   fill=color, outline='white', width=2)
        
        self.frame_label.config(text=f"Frame: {self.current_frame + 1} / {SEQUENCE_LENGTH} (Window {self.current_window_idx + 1}/{self.num_windows})")
    
    def _save_gesture(self):
        """User chose to save"""
        try:
            self.selected_class = int(self.class_var.get())
            if self.selected_class < 0:
                messagebox.showerror("Error", "Class ID must be non-negative")
                return
            
            self.result = True
            self.dialog.destroy()
        except ValueError:
            messagebox.showerror("Error", "Invalid class ID")
    
    def _discard_gesture(self):
        """User chose to discard"""
        self.result = False
        self.dialog.destroy()
    
    def show(self):
        """Show dialog and return result"""
        self.dialog.wait_window()
        return self.result, self.selected_class


class ContinuousGestureCollector:
    
    def __init__(self, root, config):
        self.root = root
        self.root.title("Continuous Gesture Sequence Collector")
        self.root.geometry("1000x800")
        
        self.config = config
        self.dataset_path = config.get('dataset_path', 'datasets/continuous_gestures_labeled.h5')
        
        self.cap = None
        self.tracker = None
        self.sequence_buffer = None
        self.current_frame = None
        self.is_running = False
        
        # Collection state
        self.is_reviewing = False
        self.pending_windows = []
        
        # Collected data
        self.collected_gestures = []  # List of dicts: {'windows': [...], 'label': int}
        
        # Load existing dataset if in load mode
        if config['mode'] == 'load' and config['dataset_path']:
            self._load_existing_dataset(config['dataset_path'])
        
        self.camera_index = tk.IntVar(value=0)
        self.available_cameras = self._detect_cameras()
        
        self._setup_ui()
        self._initialize_tracker()
    
    def _load_existing_dataset(self, filepath):
        """Load existing dataset from HDF5 file"""
        try:
            with h5py.File(filepath, 'r') as f:
                gesture_grp = f['gestures']
                
                for gesture_id in gesture_grp.keys():
                    g = gesture_grp[gesture_id]
                    windows = np.array(g['windows'])
                    label = g.attrs['label']
                    
                    # Convert windows to list for storage
                    windows_list = [windows[i] for i in range(len(windows))]
                    
                    self.collected_gestures.append({
                        'windows': windows_list,
                        'label': label
                    })
            
            messagebox.showinfo("Success", 
                               f"Loaded {len(self.collected_gestures)} gestures from existing dataset")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
        
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
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Camera and stats
        left_frame = ttk.LabelFrame(control_frame, text="Camera & Statistics", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        camera_row = ttk.Frame(left_frame)
        camera_row.pack(fill=tk.X)
        
        ttk.Label(camera_row, text="Camera:").pack(side=tk.LEFT, padx=(0, 5))
        camera_combo = ttk.Combobox(camera_row, textvariable=self.camera_index, 
                                     values=self.available_cameras, state="readonly", width=8)
        camera_combo.pack(side=tk.LEFT, padx=(0, 20))
        camera_combo.bind('<<ComboboxSelected>>', self._change_camera)
        
        self.stats_label = ttk.Label(camera_row, text=f"Collected: {len(self.collected_gestures)} gestures", 
                                     justify=tk.LEFT)
        self.stats_label.pack(side=tk.LEFT)
        
        # Actions
        right_frame = ttk.LabelFrame(control_frame, text="Actions", padding="10")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(5, 0))
        
        ttk.Button(right_frame, text="Save & Exit", 
                  command=self._save_and_exit).pack(fill=tk.X, pady=2)
        
        self.status_var = tk.StringVar(value="Initializing...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self._update_statistics()
        
    def _initialize_tracker(self):
        """Initialize hand tracker"""
        try:
            self.tracker = HandTracker(
                palm_detection_model=PALM_MODEL,
                hand_landmark_model=LANDMARK_MODEL,
                anchors=ANCHORS,
                num_hands=1
            )
            
            self.sequence_buffer = SequenceBuffer(
                min_frames=MIN_FRAMES,
                max_frames=MAX_FRAMES,
                sequence_length=SEQUENCE_LENGTH,
                feature_size=63
            )
            
            self._open_camera()
            self.is_running = True
            self._update_frame()
            self.status_var.set("Ready - Perform gestures naturally")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize:\n{str(e)}")
            self.root.quit()
    
    def _open_camera(self):
        """Open the camera"""
        if self.cap is not None:
            self.cap.release()
        
        camera_idx = self.camera_index.get()
        self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_idx}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        if actual_width < 1920:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    def _change_camera(self, event=None):
        """Change camera source"""
        self._open_camera()
        self.sequence_buffer.reset("Camera changed")
        self.status_var.set(f"Switched to camera {self.camera_index.get()}")
    
    def _draw_landmarks(self, landmarks, frame):
        """Draw hand landmarks on frame"""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        for connection in connections:
            x0, y0 = landmarks[connection[0]][:2]
            x1, y1 = landmarks[connection[1]][:2]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
        
        for i, point in enumerate(landmarks):
            x, y = point[:2]
            color = (0, 255, 0) if i in [0, 1, 5, 9, 13, 17] else (0, 200, 255)
            cv2.circle(frame, (int(x), int(y)), 6, color, -1)
    
    def _draw_overlay(self, frame, status):
        """Draw overlay information"""
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
        
        progress = min(1.0, status['frame_count'] / status['max_frames'])
        fill_width = int(bar_width * progress)
        
        if status['is_ready']:
            color = (0, 255, 0)
        else:
            color = (100, 100, 100)
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
        
        # Status text
        buffer_text_y = bar_y + bar_height + 20
        buffer_text = f"Buffer: {status['frame_count']}/{status['max_frames']}"
        if status['null_count'] > 0:
            buffer_text += f" (Nulls: {status['null_count']})"
        cv2.putText(frame, buffer_text, (bar_x, buffer_text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        text_x = bar_x + bar_width + 30
        text_y = h - overlay_height + 35
        
        if self.is_reviewing:
            review_text = "REVIEWING - Label gesture in dialog"
            cv2.putText(frame, review_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        elif status['consecutive_nulls'] > 0:
            warning_text = f"Missing detection ({status['consecutive_nulls']}/2)"
            cv2.putText(frame, warning_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        elif status['is_ready']:
            ready_text = "READY - Collecting..."
            cv2.putText(frame, ready_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            collecting_text = f"Collecting... ({status['frame_count']}/{status['min_frames']} minimum)"
            cv2.putText(frame, collecting_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    
    def _update_frame(self):
        """Update video frame"""
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            if not self.is_reviewing:
                detections = self.tracker(frame)
                
                if detections:
                    for landmarks, hand_bbox, handedness in detections:
                        self._draw_landmarks(landmarks, frame)
                        self.sequence_buffer.add_frame(landmarks)
                else:
                    result = self.sequence_buffer.add_null_frame()
                    
                    if result == 'process':
                        windows = self.sequence_buffer.get_sliding_windows(MAX_SLIDING_WINDOWS)
                        
                        if windows:
                            self._review_gesture(windows)
                        
                        self.sequence_buffer.reset("Processing triggered")
                
                status = self.sequence_buffer.get_status()
                
                if status['frame_count'] >= MAX_FRAMES and not self.is_reviewing:
                    windows = self.sequence_buffer.get_sliding_windows(MAX_SLIDING_WINDOWS)
                    
                    if windows:
                        self._review_gesture(windows)
                    
                    self.sequence_buffer.reset("Max frames reached")
            
            status = self.sequence_buffer.get_status()
            self._draw_overlay(frame, status)
            
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
    
    def _review_gesture(self, windows):
        """Review and label captured gesture"""
        self.is_reviewing = True
        self.status_var.set(f"Captured gesture with {len(windows)} windows - Label it")
        
        dialog = WindowReviewDialog(self.root, windows)
        should_save, selected_class = dialog.show()
        
        if should_save and selected_class is not None:
            self.collected_gestures.append({
                'windows': windows,
                'label': selected_class
            })
            
            self._update_statistics()
            self.status_var.set(
                f"Saved as class {selected_class} | Total: {len(self.collected_gestures)} gestures"
            )
        else:
            self.status_var.set("Gesture discarded")
        
        self.is_reviewing = False
    
    def _update_statistics(self):
        """Update statistics display"""
        total = len(self.collected_gestures)
        
        if total > 0:
            class_counts = {}
            for gesture in self.collected_gestures:
                label = gesture['label']
                class_counts[label] = class_counts.get(label, 0) + 1
            
            counts_str = ", ".join([f"C{k}:{v}" for k, v in sorted(class_counts.items())])
            stats_text = f"Collected: {total} gestures ({counts_str})"
        else:
            stats_text = "Collected: 0 gestures"
        
        self.stats_label.config(text=stats_text)
    
    def _save_dataset(self):
        """Save collected gestures to HDF5"""
        if len(self.collected_gestures) == 0:
            messagebox.showwarning("Warning", "No gestures collected yet")
            return False
        
        filename = self.dataset_path
        
        try:
            with h5py.File(filename, 'w') as f:
                # Each gesture has multiple windows
                gesture_grp = f.create_group('gestures')
                
                for idx, gesture in enumerate(self.collected_gestures):
                    gesture_id = f'gesture_{idx}'
                    g = gesture_grp.create_group(gesture_id)
                    
                    # Store all windows
                    windows_array = np.array(gesture['windows'], dtype=np.float32)
                    g.create_dataset('windows', data=windows_array)
                    
                    # Store label
                    g.attrs['label'] = gesture['label']
                    g.attrs['num_windows'] = len(gesture['windows'])
                
                # Global attributes
                f.attrs['num_gestures'] = len(self.collected_gestures)
                f.attrs['sequence_length'] = SEQUENCE_LENGTH
                f.attrs['feature_size'] = 63
                f.attrs['min_frames'] = MIN_FRAMES
                f.attrs['max_frames'] = MAX_FRAMES
                f.attrs['max_sliding_windows'] = MAX_SLIDING_WINDOWS
            
            messagebox.showinfo("Success", 
                               f"Dataset saved to {filename}\n"
                               f"Total gestures: {len(self.collected_gestures)}")
            self.status_var.set(f"Dataset saved to {filename}")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save dataset: {str(e)}")
            return False
    
    def _save_and_exit(self):
        """Save dataset and exit"""
        if len(self.collected_gestures) == 0:
            result = messagebox.askyesno("Exit", "No data saved. Exit anyway?")
            if result:
                self._cleanup()
                self.root.quit()
        else:
            if self._save_dataset():
                self._cleanup()
                self.root.quit()
    
    def _cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Show dataset selector dialog
    config = show_dataset_selector()
    
    if config['mode'] is None:
        exit()
    
    root = tk.Tk()
    app = ContinuousGestureCollector(root, config)
    
    def on_closing():
        if app.is_reviewing:
            result = messagebox.askyesno("Exit", 
                                        "Gesture review in progress. Exit anyway?")
            if not result:
                return
        
        app._cleanup()
        root.quit()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()