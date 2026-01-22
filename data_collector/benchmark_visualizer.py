import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import h5py
from collections import defaultdict


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


class BenchmarkDatasetVisualizer:

    def __init__(self, root):
        self.root = root
        self.root.title("Benchmark Gesture Dataset Visualizer")
        self.root.geometry("1400x900")
        
        self.dataset_path = None
        self.gestures_data = []  # List of {'windows': array, 'label': int, 'num_windows': int}
        self.metadata = {}
        self.class_gestures = defaultdict(list)  # {class_id: [gesture_indices]}
        self.available_classes = []
        
        self.current_class = tk.IntVar(value=0)
        self.current_gesture_idx = 0
        self.current_window_idx = 0
        self.current_frame_idx = 0
        self.is_playing = False
        self.play_speed = 100  # ms between frames
        
        self._setup_ui()
        self._bind_keys()
        
    def _setup_ui(self):
        """Setup the user interface"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=0, minsize=380)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10", width=380)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        control_frame.grid_propagate(False)
        
        ttk.Button(control_frame, text="Load Benchmark Dataset", 
                  command=self._load_dataset).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.file_label = ttk.Label(control_frame, text="No dataset loaded", wraplength=360)
        self.file_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 20))
        
        # Class selection
        ttk.Label(control_frame, text="Select Gesture Class:").grid(
            row=2, column=0, sticky=tk.W, pady=(0, 5))
        
        self.class_combo = ttk.Combobox(control_frame, textvariable=self.current_class,
                                        state="readonly", width=45)
        self.class_combo.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        self.class_combo.bind('<<ComboboxSelected>>', self._on_class_changed)
        
        # Gesture navigation
        gesture_nav_frame = ttk.LabelFrame(control_frame, text="Gesture Navigation", padding="10")
        gesture_nav_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(gesture_nav_frame, text="◄ Previous Gesture", 
                  command=self._previous_gesture).pack(fill=tk.X, pady=2)
        ttk.Button(gesture_nav_frame, text="Next Gesture ►", 
                  command=self._next_gesture).pack(fill=tk.X, pady=2)
        
        self.gesture_label = ttk.Label(gesture_nav_frame, text="Gesture: 0 / 0", 
                                       font=('Arial', 10, 'bold'))
        self.gesture_label.pack(pady=5)
        
        ttk.Button(gesture_nav_frame, text="🗑 Delete This Gesture", 
                  command=self._delete_gesture).pack(fill=tk.X, pady=(10, 2))
        
        # Window navigation
        window_nav_frame = ttk.LabelFrame(control_frame, text="Window Navigation", padding="10")
        window_nav_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        win_btn_frame = ttk.Frame(window_nav_frame)
        win_btn_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(win_btn_frame, text="◄ Prev", 
                  command=self._previous_window).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(win_btn_frame, text="Next ►", 
                  command=self._next_window).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        self.window_label = ttk.Label(window_nav_frame, text="Window: 0 / 0", 
                                      font=('Arial', 10, 'bold'))
        self.window_label.pack(pady=5)
        
        # Frame navigation
        frame_nav_frame = ttk.LabelFrame(control_frame, text="Frame Playback", padding="10")
        frame_nav_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        frame_btn_frame = ttk.Frame(frame_nav_frame)
        frame_btn_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(frame_btn_frame, text="◄", 
                  command=self._previous_frame).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.play_button = ttk.Button(frame_btn_frame, text="▶ Play", 
                  command=self._toggle_play)
        self.play_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(frame_btn_frame, text="►", 
                  command=self._next_frame).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        self.frame_label = ttk.Label(frame_nav_frame, text="Frame: 0 / 10", 
                                     font=('Arial', 10, 'bold'))
        self.frame_label.pack(pady=5)
        
        # Speed control
        speed_frame = ttk.Frame(frame_nav_frame)
        speed_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT, padx=(0, 5))
        self.speed_var = tk.IntVar(value=100)
        speed_scale = ttk.Scale(speed_frame, from_=50, to=500, variable=self.speed_var,
                               orient=tk.HORIZONTAL, command=self._on_speed_change)
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.speed_label = ttk.Label(speed_frame, text="100ms")
        self.speed_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Info display
        info_frame = ttk.LabelFrame(control_frame, text="Current Sample Info", padding="10")
        info_frame.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.info_label = ttk.Label(info_frame, text="", justify=tk.LEFT, wraplength=340)
        self.info_label.pack(fill=tk.BOTH, expand=True)
        
        # Statistics - Compact format
        stats_frame = ttk.LabelFrame(control_frame, text="Dataset Statistics", padding="10")
        stats_frame.grid(row=8, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        
        self.stats_label = ttk.Label(stats_frame, text="", justify=tk.LEFT, wraplength=340)
        self.stats_label.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for visualization
        canvas_frame = ttk.LabelFrame(main_frame, text="Landmark Visualization", padding="10")
        canvas_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(canvas_frame, bg='black')
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load a benchmark dataset to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def _bind_keys(self):
        """Bind keyboard shortcuts"""
        self.root.bind('<Left>', lambda e: self._previous_frame())
        self.root.bind('<Right>', lambda e: self._next_frame())
        self.root.bind('<Up>', lambda e: self._previous_window())
        self.root.bind('<Down>', lambda e: self._next_window())
        self.root.bind('<space>', lambda e: self._toggle_play())
        self.root.bind('<Prior>', lambda e: self._previous_gesture())  # Page Up
        self.root.bind('<Next>', lambda e: self._next_gesture())  # Page Down
        
    def _on_canvas_resize(self, event):
        """Handle canvas resize"""
        if self.gestures_data:
            self._visualize_current_frame()
        
    def _on_speed_change(self, value):
        """Handle speed slider change"""
        self.play_speed = int(float(value))
        self.speed_label.config(text=f"{self.play_speed}ms")
        
    def _load_dataset(self):
        """Load benchmark dataset from H5 file"""
        filename = filedialog.askopenfilename(
            title="Select Benchmark Dataset File",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            self.gestures_data = []
            
            with h5py.File(filename, 'r') as f:
                # Read metadata
                self.metadata = {
                    'num_gestures': f.attrs.get('num_gestures', 0),
                    'sequence_length': f.attrs.get('sequence_length', 10),
                    'feature_size': f.attrs.get('feature_size', 63),
                    'min_frames': f.attrs.get('min_frames', 10),
                    'max_frames': f.attrs.get('max_frames', 20),
                    'max_sliding_windows': f.attrs.get('max_sliding_windows', 7)
                }
                
                # Read gestures
                gestures_grp = f['gestures']
                for gesture_id in sorted(gestures_grp.keys(), key=lambda x: int(x.split('_')[1])):
                    g = gestures_grp[gesture_id]
                    windows = g['windows'][:]
                    label = g.attrs['label']
                    num_windows = g.attrs['num_windows']
                    
                    self.gestures_data.append({
                        'windows': windows,
                        'label': int(label),
                        'num_windows': int(num_windows)
                    })
            
            self.dataset_path = filename
            self._process_dataset()
            self._update_statistics()
            self._populate_class_selector()
            self._on_class_changed()
            
            self.file_label.config(text=f"Loaded: {filename.split('/')[-1]}")
            self.status_var.set(f"Dataset loaded - {self.metadata['num_gestures']} gestures, "
                              f"{sum(g['num_windows'] for g in self.gestures_data)} total windows")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")
    
    def _process_dataset(self):
        """Process dataset to organize gestures by class"""
        self.class_gestures.clear()
        
        for idx, gesture in enumerate(self.gestures_data):
            self.class_gestures[gesture['label']].append(idx)
        
        self.available_classes = sorted(self.class_gestures.keys())
    
    def _populate_class_selector(self):
        """Populate the class selection combobox"""
        if not self.available_classes:
            self.class_combo['values'] = []
            return
        
        class_labels = []
        for cls in self.available_classes:
            gesture_count = len(self.class_gestures[cls])
            window_count = sum(self.gestures_data[idx]['num_windows'] 
                             for idx in self.class_gestures[cls])
            class_name = GESTURE_NAMES.get(cls, f"Class {cls}")
            class_labels.append(f"{cls}: {class_name} ({gesture_count} gestures, {window_count} windows)")
        
        self.class_combo['values'] = class_labels
        
        if self.available_classes:
            self.class_combo.current(0)
            self.current_class.set(self.available_classes[0])
    
    def _update_statistics(self):
        """Update statistics display in compact format"""
        if not self.gestures_data:
            return
        
        total_windows = sum(g['num_windows'] for g in self.gestures_data)
        
        # Compact statistics format
        stats_text = f"Total Gestures: {self.metadata['num_gestures']}\n"
        stats_text += f"Total Windows: {total_windows}\n"
        stats_text += f"Sequence Length: {self.metadata['sequence_length']}\n"
        stats_text += f"Max Windows/Gesture: {self.metadata['max_sliding_windows']}\n\n"
        
        # Compact gestures per class - show counts in a row
        gesture_counts = []
        window_counts = []
        for cls in self.available_classes:
            gesture_counts.append(str(len(self.class_gestures[cls])))
            window_counts.append(str(sum(self.gestures_data[idx]['num_windows'] 
                                        for idx in self.class_gestures[cls])))
        
        stats_text += "Gestures per Class:\n"
        stats_text += ', '.join(gesture_counts)
        stats_text += "\n\nWindows per Class:\n"
        stats_text += ', '.join(window_counts)
        
        self.stats_label.config(text=stats_text)
    
    def _on_class_changed(self, event=None):
        """Handle class selection change"""
        if not self.available_classes:
            return
        
        combo_text = self.class_combo.get()
        if combo_text:
            try:
                class_id = int(combo_text.split(':')[0])
                self.current_class.set(class_id)
            except:
                pass
        
        self.current_gesture_idx = 0
        self.current_window_idx = 0
        self.current_frame_idx = 0
        self._visualize_current_frame()
    
    def _next_gesture(self):
        """Navigate to next gesture"""
        if not self.available_classes:
            return
        
        current_class = self.current_class.get()
        if current_class not in self.class_gestures:
            return
        
        num_gestures = len(self.class_gestures[current_class])
        self.current_gesture_idx = (self.current_gesture_idx + 1) % num_gestures
        self.current_window_idx = 0
        self.current_frame_idx = 0
        self._visualize_current_frame()
    
    def _previous_gesture(self):
        """Navigate to previous gesture"""
        if not self.available_classes:
            return
        
        current_class = self.current_class.get()
        if current_class not in self.class_gestures:
            return
        
        num_gestures = len(self.class_gestures[current_class])
        self.current_gesture_idx = (self.current_gesture_idx - 1) % num_gestures
        self.current_window_idx = 0
        self.current_frame_idx = 0
        self._visualize_current_frame()
    
    def _next_window(self):
        """Navigate to next window"""
        if not self.gestures_data:
            return
        
        current_class = self.current_class.get()
        gesture_idx = self.class_gestures[current_class][self.current_gesture_idx]
        num_windows = self.gestures_data[gesture_idx]['num_windows']
        
        self.current_window_idx = (self.current_window_idx + 1) % num_windows
        self.current_frame_idx = 0
        self._visualize_current_frame()
    
    def _previous_window(self):
        """Navigate to previous window"""
        if not self.gestures_data:
            return
        
        current_class = self.current_class.get()
        gesture_idx = self.class_gestures[current_class][self.current_gesture_idx]
        num_windows = self.gestures_data[gesture_idx]['num_windows']
        
        self.current_window_idx = (self.current_window_idx - 1) % num_windows
        self.current_frame_idx = 0
        self._visualize_current_frame()
    
    def _next_frame(self):
        """Navigate to next frame"""
        if not self.gestures_data:
            return
        
        self.current_frame_idx = (self.current_frame_idx + 1) % self.metadata['sequence_length']
        self._visualize_current_frame()
    
    def _previous_frame(self):
        """Navigate to previous frame"""
        if not self.gestures_data:
            return
        
        self.current_frame_idx = (self.current_frame_idx - 1) % self.metadata['sequence_length']
        self._visualize_current_frame()
    
    def _toggle_play(self):
        """Toggle playback of sequence"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.config(text="⏸ Pause")
            self._play_sequence()
        else:
            self.play_button.config(text="▶ Play")
    
    def _play_sequence(self):
        """Play sequence animation"""
        if not self.is_playing:
            return
        
        self._next_frame()
        self.root.after(self.play_speed, self._play_sequence)
    
    def _delete_gesture(self):
        """Delete the current gesture"""
        if not self.gestures_data:
            return
        
        current_class = self.current_class.get()
        if current_class not in self.class_gestures:
            return
        
        gesture_indices = self.class_gestures[current_class]
        if not gesture_indices:
            return
        
        dataset_idx = gesture_indices[self.current_gesture_idx]
        gesture = self.gestures_data[dataset_idx]
        
        result = messagebox.askyesno("Delete Gesture", 
                                     f"Delete gesture {self.current_gesture_idx + 1} "
                                     f"of {GESTURE_NAMES.get(current_class, f'Class {current_class}')}?\n\n"
                                     f"This gesture has {gesture['num_windows']} windows.\n"
                                     f"Dataset index: {dataset_idx}\n"
                                     f"This cannot be undone!")
        
        if not result:
            return
        
        try:
            # Remove from data
            del self.gestures_data[dataset_idx]
            
            # Save updated dataset
            with h5py.File(self.dataset_path, 'w') as f:
                gestures_grp = f.create_group('gestures')
                
                for idx, gesture in enumerate(self.gestures_data):
                    gesture_id = f'gesture_{idx}'
                    g = gestures_grp.create_group(gesture_id)
                    
                    g.create_dataset('windows', data=gesture['windows'])
                    g.attrs['label'] = gesture['label']
                    g.attrs['num_windows'] = gesture['num_windows']
                
                # Update metadata
                self.metadata['num_gestures'] = len(self.gestures_data)
                for key, value in self.metadata.items():
                    f.attrs[key] = value
            
            self._process_dataset()
            self._update_statistics()
            self._populate_class_selector()
            
            if self.current_gesture_idx >= len(self.class_gestures[current_class]):
                self.current_gesture_idx = max(0, len(self.class_gestures[current_class]) - 1)
            
            self._visualize_current_frame()
            
            self.status_var.set(f"Gesture deleted. Total: {len(self.gestures_data)}")
            messagebox.showinfo("Success", "Gesture deleted successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete gesture:\n{str(e)}")
    
    def _visualize_current_frame(self):
        """Visualize the current frame of current window"""
        if not self.gestures_data:
            return
        
        current_class = self.current_class.get()
        if current_class not in self.class_gestures:
            return
        
        gesture_indices = self.class_gestures[current_class]
        if not gesture_indices:
            return
        
        dataset_idx = gesture_indices[self.current_gesture_idx]
        gesture = self.gestures_data[dataset_idx]
        
        windows = gesture['windows']
        current_window = windows[self.current_window_idx]
        frame_landmarks = current_window[self.current_frame_idx]
        
        num_gestures = len(gesture_indices)
        num_windows = gesture['num_windows']
        seq_length = len(current_window)
        
        class_name = GESTURE_NAMES.get(current_class, f"Class {current_class}")
        
        info_text = f"Gesture Class: {current_class} - {class_name}\n"
        info_text += f"Gesture: {self.current_gesture_idx + 1} / {num_gestures}\n"
        info_text += f"Window: {self.current_window_idx + 1} / {num_windows}\n"
        info_text += f"Frame: {self.current_frame_idx + 1} / {seq_length}\n"
        info_text += f"Dataset Index: {dataset_idx}"
        self.info_label.config(text=info_text)
        
        self.gesture_label.config(text=f"Gesture: {self.current_gesture_idx + 1} / {num_gestures}")
        self.window_label.config(text=f"Window: {self.current_window_idx + 1} / {num_windows}")
        self.frame_label.config(text=f"Frame: {self.current_frame_idx + 1} / {seq_length}")
        
        self._draw_landmarks(frame_landmarks, current_window)
        
        self.status_var.set(f"Viewing {class_name} - "
                          f"Gesture {self.current_gesture_idx + 1}/{num_gestures}, "
                          f"Window {self.current_window_idx + 1}/{num_windows}, "
                          f"Frame {self.current_frame_idx + 1}/{seq_length}")
    
    def _draw_landmarks(self, landmarks, full_sequence):
        """Draw hand landmarks on the canvas"""
        self.canvas.delete("all")
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        landmarks = landmarks.reshape(21, 3)
        landmarks_2d = landmarks[:, :2]
        
        # Calculate global bounds for consistent scaling
        all_frames = full_sequence.reshape(-1, 21, 3)[:, :, :2]
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
        
        # Draw ghost frames (previous frames)
        if self.current_frame_idx > 0:
            for frame_idx in range(max(0, self.current_frame_idx - 3), self.current_frame_idx):
                prev_frame = full_sequence[frame_idx].reshape(21, 3)[:, :2]
                prev_scaled = []
                for x, y in prev_frame:
                    canvas_x = center_x + (x - global_center_x) * scale
                    canvas_y = center_y + (y - global_center_y) * scale
                    prev_scaled.append((canvas_x, canvas_y))
                
                # Calculate alpha based on frame distance
                num_ghost_frames = self.current_frame_idx - max(0, self.current_frame_idx - 3)
                if num_ghost_frames > 0:
                    alpha = 50 + 50 * (frame_idx - max(0, self.current_frame_idx - 3)) / num_ghost_frames
                else:
                    alpha = 50
                gray = int(alpha)
                color = f'#{gray:02x}{gray:02x}{gray:02x}'
                
                # Draw ghost landmarks
                for x, y in prev_scaled:
                    self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, 
                                          fill=color, outline=color)
        
        # Hand skeleton connections
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (5, 9), (9, 13), (13, 17)
        ]
        
        for idx1, idx2 in connections:
            x1, y1 = scaled_landmarks[idx1]
            x2, y2 = scaled_landmarks[idx2]
            self.canvas.create_line(x1, y1, x2, y2, fill='#1E90FF', width=3)
        
        # Draw landmarks (points)
        palm_joints = [0, 1, 2, 5, 9, 13, 17]
        for idx, (x, y) in enumerate(scaled_landmarks):
            if idx == 0:  # Wrist
                color = '#FF0000'
                size = 10
            elif idx in palm_joints:  # Palm joints
                color = '#FFA500'
                size = 8
            else:  # Finger joints
                color = '#00FF00'
                size = 6
            
            self.canvas.create_oval(
                x - size, y - size, x + size, y + size,
                fill=color, outline='white', width=2
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = BenchmarkDatasetVisualizer(root)
    
    def on_closing():
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()