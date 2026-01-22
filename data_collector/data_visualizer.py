import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import h5py
from collections import defaultdict
import numpy as np


class DatasetVisualizer:

    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Dataset Visualizer - Sequence Mode")
        self.root.geometry("1200x800")
        
        self.dataset_path = None
        self.sequences_data = None
        self.labels_data = None
        self.metadata = {}
        self.class_sequences = defaultdict(list)  # {class_id: [sequence_indices]}
        self.available_classes = []
        
        self.current_class = tk.IntVar(value=0)
        self.current_sequence_idx = 0
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
        main_frame.columnconfigure(0, weight=0, minsize=350)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10", width=350)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        control_frame.grid_propagate(False)
        
        ttk.Button(control_frame, text="Load Dataset", 
                  command=self._load_dataset).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.file_label = ttk.Label(control_frame, text="No dataset loaded", wraplength=330)
        self.file_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 20))
        
        ttk.Label(control_frame, text="Select Gesture Class:").grid(
            row=2, column=0, sticky=tk.W, pady=(0, 5))
        
        self.class_combo = ttk.Combobox(control_frame, textvariable=self.current_class,
                                        state="readonly", width=40)
        self.class_combo.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        self.class_combo.bind('<<ComboboxSelected>>', self._on_class_changed)
        
        seq_nav_frame = ttk.LabelFrame(control_frame, text="Sequence Navigation", padding="10")
        seq_nav_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(seq_nav_frame, text="Previous Sequence", 
                  command=self._previous_sequence).pack(fill=tk.X, pady=2)
        ttk.Button(seq_nav_frame, text="Next Sequence", 
                  command=self._next_sequence).pack(fill=tk.X, pady=2)
        
        self.sequence_label = ttk.Label(seq_nav_frame, text="Sequence: 0 / 0", 
                                       font=('Arial', 10, 'bold'))
        self.sequence_label.pack(pady=5)
        
        ttk.Button(seq_nav_frame, text="Delete This Sequence", 
                  command=self._delete_sequence).pack(fill=tk.X, pady=(10, 2))
        
        frame_nav_frame = ttk.LabelFrame(control_frame, text="Frame Navigation", padding="10")
        frame_nav_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        btn_frame = ttk.Frame(frame_nav_frame)
        btn_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_frame, text="Prev", 
                  command=self._previous_frame).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.play_button = ttk.Button(btn_frame, text="Play", 
                  command=self._toggle_play)
        self.play_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btn_frame, text="Next", 
                  command=self._next_frame).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        self.frame_label = ttk.Label(frame_nav_frame, text="Frame: 0 / 10", 
                                     font=('Arial', 10, 'bold'))
        self.frame_label.pack(pady=5)
        
        speed_frame = ttk.Frame(frame_nav_frame)
        speed_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT, padx=(0, 5))
        self.speed_var = tk.IntVar(value=100)
        speed_scale = ttk.Scale(speed_frame, from_=50, to=500, variable=self.speed_var,
                               orient=tk.HORIZONTAL, command=self._on_speed_change)
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.speed_label = ttk.Label(speed_frame, text="100ms")
        self.speed_label.pack(side=tk.LEFT, padx=(5, 0))
        
        info_frame = ttk.LabelFrame(control_frame, text="Sample Info", padding="10")
        info_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.info_label = ttk.Label(info_frame, text="", justify=tk.LEFT, wraplength=310)
        self.info_label.pack(fill=tk.BOTH, expand=True)
        
        stats_frame = ttk.LabelFrame(control_frame, text="Dataset Statistics", padding="10")
        stats_frame.grid(row=7, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        
        self.stats_label = ttk.Label(stats_frame, text="", justify=tk.LEFT, wraplength=310)
        self.stats_label.pack(fill=tk.BOTH, expand=True)
        
        canvas_frame = ttk.LabelFrame(main_frame, text="Landmark Visualization", padding="10")
        canvas_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(canvas_frame, bg='black')
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        self.status_var = tk.StringVar(value="Ready - Load a dataset to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def _bind_keys(self):
        """Bind keyboard shortcuts"""
        self.root.bind('<Left>', lambda e: self._previous_frame())
        self.root.bind('<Right>', lambda e: self._next_frame())
        self.root.bind('<Up>', lambda e: self._previous_sequence())
        self.root.bind('<Down>', lambda e: self._next_sequence())
        self.root.bind('<space>', lambda e: self._toggle_play())
        self.root.bind('a', lambda e: self._previous_frame())
        self.root.bind('A', lambda e: self._previous_frame())
        self.root.bind('d', lambda e: self._next_frame())
        self.root.bind('D', lambda e: self._next_frame())
        self.root.bind('w', lambda e: self._previous_sequence())
        self.root.bind('W', lambda e: self._previous_sequence())
        self.root.bind('s', lambda e: self._next_sequence())
        self.root.bind('S', lambda e: self._next_sequence())
    
    def _on_canvas_resize(self, event):
        """Handle canvas resize"""
        if self.sequences_data is not None:
            self._visualize_current_frame()
        
    def _on_speed_change(self, value):
        """Handle speed slider change"""
        self.play_speed = int(float(value))
        self.speed_label.config(text=f"{self.play_speed}ms")
        
    def _load_dataset(self):
        """Load dataset from H5 file"""
        filename = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            with h5py.File(filename, 'r') as hf:
                self.sequences_data = hf['sequences'][:]
                self.labels_data = hf['labels'][:]
                
                self.metadata = {
                    'total_sequences': hf.attrs.get('total_sequences', len(self.sequences_data)),
                    'num_classes': hf.attrs.get('num_classes', 0),
                    'num_landmarks': hf.attrs.get('num_landmarks', 21),
                    'landmark_dims': hf.attrs.get('landmark_dims', 3),
                    'feature_size': hf.attrs.get('feature_size', 63),
                    'sequence_length': hf.attrs.get('sequence_length', 10)
                }
            
            self.dataset_path = filename
            self._process_dataset()
            self._update_statistics()
            self._populate_class_selector()
            self._on_class_changed()
            
            self.file_label.config(text=f"Loaded: {filename.split('/')[-1]}")
            self.status_var.set(f"Dataset loaded - {self.metadata['total_sequences']} sequences")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")
    
    def _process_dataset(self):
        """Process dataset to organize sequences by class"""
        self.class_sequences.clear()
        
        for idx, label in enumerate(self.labels_data):
            self.class_sequences[int(label)].append(idx)
        
        self.available_classes = sorted(self.class_sequences.keys())
    
    def _populate_class_selector(self):
        """Populate the class selection combobox"""
        if not self.available_classes:
            self.class_combo['values'] = []
            return
        
        class_labels = [f"Gesture {cls} ({len(self.class_sequences[cls])} sequences)" 
                       for cls in self.available_classes]
        self.class_combo['values'] = class_labels
        
        if self.available_classes:
            self.class_combo.current(0)
            self.current_class.set(self.available_classes[0])
    
    def _update_statistics(self):
        """Update statistics display"""
        if self.sequences_data is None:
            return
        
        counts = [str(len(self.class_sequences[cls])) for cls in self.available_classes]
        counts_str = ', '.join(counts) if counts else '0'
        
        stats_text = f"Total Sequences: {self.metadata['total_sequences']}\n"
        stats_text += f"Number of Classes: {self.metadata['num_classes']}\n"
        stats_text += f"Landmarks: {self.metadata['num_landmarks']} ({self.metadata['landmark_dims']}D)\n\n"
        stats_text += f"Sequences per Class:\n{counts_str}"
        
        self.stats_label.config(text=stats_text)
    
    def _on_class_changed(self, event=None):
        """Handle class selection change"""
        if not self.available_classes:
            return
        
        combo_text = self.class_combo.get()
        if combo_text:
            try:
                class_id = int(combo_text.split()[1])
                self.current_class.set(class_id)
            except:
                pass
        
        self.current_sequence_idx = 0
        self.current_frame_idx = 0
        self._visualize_current_frame()
    
    def _next_sequence(self):
        """Navigate to next sequence"""
        if not self.available_classes:
            return
        
        current_class = self.current_class.get()
        if current_class not in self.class_sequences:
            return
        
        num_sequences = len(self.class_sequences[current_class])
        self.current_sequence_idx = (self.current_sequence_idx + 1) % num_sequences
        self.current_frame_idx = 0
        self._visualize_current_frame()
    
    def _previous_sequence(self):
        """Navigate to previous sequence"""
        if not self.available_classes:
            return
        
        current_class = self.current_class.get()
        if current_class not in self.class_sequences:
            return
        
        num_sequences = len(self.class_sequences[current_class])
        self.current_sequence_idx = (self.current_sequence_idx - 1) % num_sequences
        self.current_frame_idx = 0
        self._visualize_current_frame()
    
    def _next_frame(self):
        """Navigate to next frame"""
        if self.sequences_data is None:
            return
        
        self.current_frame_idx = (self.current_frame_idx + 1) % self.metadata['sequence_length']
        self._visualize_current_frame()
    
    def _previous_frame(self):
        """Navigate to previous frame"""
        if self.sequences_data is None:
            return
        
        self.current_frame_idx = (self.current_frame_idx - 1) % self.metadata['sequence_length']
        self._visualize_current_frame()
    
    def _toggle_play(self):
        """Toggle playback of sequence"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.config(text="Pause")
            self._play_sequence()
        else:
            self.play_button.config(text="Play")
    
    def _play_sequence(self):
        """Play sequence animation"""
        if not self.is_playing:
            return
        
        self._next_frame()
        self.root.after(self.play_speed, self._play_sequence)
    
    def _delete_sequence(self):
        """Delete the current sequence"""
        if self.sequences_data is None:
            return
        
        current_class = self.current_class.get()
        if current_class not in self.class_sequences:
            return
        
        sequence_indices = self.class_sequences[current_class]
        if not sequence_indices:
            return
        
        dataset_idx = sequence_indices[self.current_sequence_idx]
        
        result = messagebox.askyesno("Delete Sequence", 
                                     f"Delete sequence {self.current_sequence_idx + 1} "
                                     f"of Gesture {current_class}?\n\n"
                                     f"Dataset index: {dataset_idx}\n"
                                     f"This cannot be undone!")
        
        if not result:
            return
        
        try:
            self.sequences_data = np.delete(self.sequences_data, dataset_idx, axis=0)
            self.labels_data = np.delete(self.labels_data, dataset_idx, axis=0)
            
            with h5py.File(self.dataset_path, 'w') as hf:
                hf.create_dataset('sequences', data=self.sequences_data, compression='gzip')
                hf.create_dataset('labels', data=self.labels_data, compression='gzip')
                
                self.metadata['total_sequences'] = len(self.sequences_data)
                for key, value in self.metadata.items():
                    hf.attrs[key] = value
            
            self._process_dataset()
            self._update_statistics()
            self._populate_class_selector()
            
            if self.current_sequence_idx >= len(self.class_sequences[current_class]):
                self.current_sequence_idx = max(0, len(self.class_sequences[current_class]) - 1)
            
            self._visualize_current_frame()
            
            self.status_var.set(f"Sequence deleted. Total: {len(self.sequences_data)}")
            messagebox.showinfo("Success", "Sequence deleted successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete sequence:\n{str(e)}")
    
    def _visualize_current_frame(self):
        """Visualize the current frame of current sequence"""
        if self.sequences_data is None:
            return
        
        current_class = self.current_class.get()
        if current_class not in self.class_sequences:
            return
        
        sequence_indices = self.class_sequences[current_class]
        if not sequence_indices:
            return
        
        dataset_idx = sequence_indices[self.current_sequence_idx]
        sequence = self.sequences_data[dataset_idx]
        frame_landmarks = sequence[self.current_frame_idx]
        
        num_sequences = len(sequence_indices)
        seq_length = len(sequence)
        info_text = f"Gesture Class: {current_class}\n"
        info_text += f"Sequence: {self.current_sequence_idx + 1} / {num_sequences}\n"
        info_text += f"Frame: {self.current_frame_idx + 1} / {seq_length}\n"
        info_text += f"Dataset Index: {dataset_idx}"
        self.info_label.config(text=info_text)
        
        self.sequence_label.config(text=f"Sequence: {self.current_sequence_idx + 1} / {num_sequences}")
        self.frame_label.config(text=f"Frame: {self.current_frame_idx + 1} / {seq_length}")
        
        self._draw_landmarks(frame_landmarks, sequence)
        self.status_var.set(f"Viewing Gesture {current_class} - "
                          f"Seq {self.current_sequence_idx + 1}/{num_sequences}, "
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
        
        if self.current_frame_idx > 0:
            for frame_idx in range(max(0, self.current_frame_idx - 3), self.current_frame_idx):
                prev_frame = full_sequence[frame_idx].reshape(21, 3)[:, :2]
                prev_scaled = []
                for x, y in prev_frame:
                    canvas_x = center_x + (x - global_center_x) * scale
                    canvas_y = center_y + (y - global_center_y) * scale
                    prev_scaled.append((canvas_x, canvas_y))
                
                # Calculate alpha based on frame distance
                alpha = 50 + 50 * (frame_idx - max(0, self.current_frame_idx - 3)) / max(1, self.current_frame_idx - max(0, self.current_frame_idx - 3))
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
            if idx == 0:
                color = '#FF0000'
                size = 10
            elif idx in palm_joints:
                color = '#FFA500'
                size = 8
            else:
                color = '#00FF00'
                size = 6
            
            self.canvas.create_oval(
                x - size, y - size, x + size, y + size,
                fill=color, outline='white', width=2
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetVisualizer(root)
    
    def on_closing():
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()