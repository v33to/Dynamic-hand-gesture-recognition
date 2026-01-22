import os
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from hand_tracker import HandTracker
from dataset_manager import DatasetManager
from dataset_selector import show_dataset_selector
from utils import calculate_dataset_diversity, rotate_sequence

path = os.getcwd()


class AugmentationDialog:
    """Dialog for interactive augmentation with diversity scoring"""
    
    def __init__(self, parent, sequence, gesture_id, dataset_manager, pending_sequences):
        self.result_sequences = []  # List of tuples: (augmented_sequence, angle)
        self.current_frame = 0
        self.is_playing = True
        self.play_speed = 100
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Data Augmentation - Apply Rotations")
        self.dialog.geometry("1200x850")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.original_sequence = sequence
        self.gesture_id = gesture_id
        self.dataset_manager = dataset_manager
        self.pending_sequences = pending_sequences
        
        # Track which augmentations have been saved in this session
        self.saved_augmentations = {}  # {angle: augmented_sequence}
        
        # Current preview state
        self.current_angle = 0
        self.current_augmented = None
        self.current_diversity = 1.0
        self.current_similarities = []
        
        self._setup_ui()
        self._center_window()
        self._preview_rotation(0)
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
        """Setup augmentation dialog UI"""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top control frame - horizontal layout
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)
        
        # Rotation buttons frame (left side)
        rotation_frame = ttk.LabelFrame(top_frame, text="Apply Rotation", padding="10")
        rotation_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        info_label = ttk.Label(rotation_frame, 
                              text=f"Gesture ID: {self.gesture_id}\n"
                                   "Click a rotation angle to preview the augmented data.",
                              justify=tk.LEFT)
        info_label.pack(pady=(0, 5))
        
        angles = [-90, -60, -30, 30, 60, 90]
        
        btn_frame = ttk.Frame(rotation_frame)
        btn_frame.pack(pady=5)
        
        for angle in angles:
            btn = ttk.Button(btn_frame, text=f"{angle:+d}°", 
                           command=lambda a=angle: self._preview_rotation(a),
                           width=10)
            btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(rotation_frame, text="Preview Original (0°)", 
                  command=lambda: self._preview_rotation(0)).pack(pady=5)
        
        # Diversity score frame (right side)
        diversity_frame = ttk.LabelFrame(top_frame, text="Diversity Analysis", padding="10")
        diversity_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.diversity_label = ttk.Label(diversity_frame, text="", justify=tk.LEFT,
                                        font=('Arial', 10))
        self.diversity_label.pack()
        
        self.diversity_warning = tk.Label(diversity_frame, text="", 
                                         font=('Arial', 9, 'bold'), pady=5)
        self.diversity_warning.pack()
        
        # Visualization frame with side-by-side canvases
        canvas_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="10")
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.columnconfigure(1, weight=1)
        canvas_frame.rowconfigure(1, weight=1)
        
        # Original canvas
        original_label_frame = ttk.Frame(canvas_frame)
        original_label_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Label(original_label_frame, text="Original", 
                 font=('Arial', 12, 'bold')).pack()
        
        self.canvas_original = tk.Canvas(canvas_frame, bg='black', height=300)
        self.canvas_original.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Augmented canvas
        augmented_label_frame = ttk.Frame(canvas_frame)
        augmented_label_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.augmented_title = ttk.Label(augmented_label_frame, text="Augmented (0°)", 
                                        font=('Arial', 12, 'bold'))
        self.augmented_title.pack()
        
        self.canvas_augmented = tk.Canvas(canvas_frame, bg='black', height=300)
        self.canvas_augmented.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Playback controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack()
        
        ttk.Button(btn_frame, text="Prev", command=self._prev_frame).pack(side=tk.LEFT, padx=2)
        self.play_button = ttk.Button(btn_frame, text="Pause", command=self._toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Next", command=self._next_frame).pack(side=tk.LEFT, padx=2)
        
        self.frame_label = ttk.Label(control_frame, text=f"Frame: 1 / {len(self.original_sequence)}")
        self.frame_label.pack(pady=5)
        
        # Saved augmentations display
        saved_frame = ttk.LabelFrame(main_frame, text="Saved Augmentations", padding="10")
        saved_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.saved_label = ttk.Label(saved_frame, text="None saved yet", 
                                    justify=tk.CENTER, font=('Arial', 9))
        self.saved_label.pack()
        
        # Combined Action/Decision buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.save_aug_button = ttk.Button(button_frame, text="Save Current Augmentation", 
                  command=self._save_current_augmentation)
        self.save_aug_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Done button
        ttk.Button(button_frame, text="Done - Save All", 
                  command=self._finish_augmentation).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Cancel button
        ttk.Button(button_frame, text="Cancel", 
                  command=self._cancel_augmentation).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.dialog.bind('<Left>', lambda e: self._prev_frame())
        self.dialog.bind('<Right>', lambda e: self._next_frame())
        self.dialog.bind('<space>', lambda e: self._toggle_play())
        
    def _preview_rotation(self, angle):
        """Preview a rotation angle"""
        self.current_angle = angle
        
        if angle == 0:
            self.current_augmented = self.original_sequence.copy()
        else:
            self.current_augmented = rotate_sequence(self.original_sequence, angle)
        
        self.augmented_title.config(text=f"Augmented ({angle:+d}°)")
        
        # Calculate diversity considering all saved augmentations
        self._update_diversity_score()
        
        # Update visualization
        self.current_frame = 0
        self._visualize_frame()
        
    def _update_diversity_score(self):
        """Update diversity score for current augmentation"""
        # Get existing sequences from dataset
        existing_indices = self.dataset_manager.get_class_sequences(self.gesture_id)
        existing_sequences = [np.array(self.dataset_manager.data[idx]) for idx in existing_indices]
        
        # Combine with all pending sequences (from current session) and saved augmentations
        all_pending = self.pending_sequences.copy()
        for aug_seq in self.saved_augmentations.values():
            all_pending.append(aug_seq)
        
        # Calculate diversity
        self.current_diversity, self.current_similarities = calculate_dataset_diversity(
            self.current_augmented, existing_sequences, all_pending
        )
        
        # Update diversity display
        diversity_text = f"Current Rotation: {self.current_angle:+d}°\n"
        diversity_text += f"Diversity Score: {self.current_diversity:.3f}\n"
        
        if self.current_similarities:
            diversity_text += f"Most Similar: {max(self.current_similarities):.3f}\n"
            diversity_text += f"Least Similar: {min(self.current_similarities):.3f}"
        else:
            diversity_text += "No existing sequences for comparison"
        
        self.diversity_label.config(text=diversity_text)
        
        # Update warning
        if self.current_diversity < 0.3:
            warning_msg = "WARNING: Very low diversity - highly similar to existing data!"
            warning_color = "red"
        elif self.current_diversity < 0.5:
            warning_msg = "CAUTION: Moderate diversity - somewhat similar to existing data"
            warning_color = "orange"
        else:
            warning_msg = "Good diversity - sufficiently different from existing data"
            warning_color = "green"
        
        self.diversity_warning.config(text=warning_msg, fg=warning_color)
        
    def _save_current_augmentation(self):
        """Save the current augmentation"""
        if self.current_angle == 0:
            messagebox.showinfo("Info", "Cannot save the original (0°) as an augmentation.\n"
                              "The original was already saved in the previous step.")
            return
        
        if self.current_angle in self.saved_augmentations:
            result = messagebox.askyesno("Replace?", 
                                        f"Rotation {self.current_angle:+d}° already saved.\n"
                                        f"Do you want to replace it?")
            if not result:
                return
        
        # Save the augmentation
        self.saved_augmentations[self.current_angle] = self.current_augmented.copy()
        
        # Update saved augmentations display
        self._update_saved_display()
        
    def _update_saved_display(self):
        """Update the display of saved augmentations - SINGLE ROW"""
        if not self.saved_augmentations:
            self.saved_label.config(text="None saved yet")
            return
        
        # Display all angles in a single line
        angles_str = ", ".join([f"{angle:+d}°" for angle in sorted(self.saved_augmentations.keys())])
        saved_text = f"Saved {len(self.saved_augmentations)} augmentation(s): {angles_str}"
        
        self.saved_label.config(text=saved_text)
        
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
            self.play_button.config(text="▶ Play")
    
    def _next_frame(self):
        """Go to next frame"""
        self.current_frame = (self.current_frame + 1) % len(self.original_sequence)
        self._visualize_frame()
    
    def _prev_frame(self):
        """Go to previous frame"""
        self.current_frame = (self.current_frame - 1) % len(self.original_sequence)
        self._visualize_frame()
    
    def _visualize_frame(self):
        """Visualize current frame on both canvases"""
        # Draw original
        self._draw_landmarks(self.canvas_original, 
                           self.original_sequence[self.current_frame],
                           self.original_sequence)
        
        # Draw augmented
        if self.current_augmented is not None:
            self._draw_landmarks(self.canvas_augmented,
                               self.current_augmented[self.current_frame],
                               self.current_augmented)
        
        self.frame_label.config(text=f"Frame: {self.current_frame + 1} / {len(self.original_sequence)}")
    
    def _draw_landmarks(self, canvas, landmarks, full_sequence):
        """Draw hand landmarks on the specified canvas"""
        canvas.delete("all")
        
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
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
        
        # Draw ghost frames
        if self.current_frame > 0:
            for frame_idx in range(max(0, self.current_frame - 3), self.current_frame):
                prev_frame = full_sequence[frame_idx].reshape(21, 3)[:, :2]
                prev_scaled = []
                for x, y in prev_frame:
                    canvas_x = center_x + (x - global_center_x) * scale
                    canvas_y = center_y + (y - global_center_y) * scale
                    prev_scaled.append((canvas_x, canvas_y))
                
                alpha = 50 + 50 * (frame_idx - max(0, self.current_frame - 3)) / max(1, self.current_frame - max(0, self.current_frame - 3))
                gray = int(alpha)
                color = f'#{gray:02x}{gray:02x}{gray:02x}'
                
                for x, y in prev_scaled:
                    canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline=color)
        
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
            canvas.create_line(x1, y1, x2, y2, fill='#1E90FF', width=3)
        
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
            
            canvas.create_oval(x - size, y - size, x + size, y + size,
                             fill=color, outline='white', width=2)
    
    def _finish_augmentation(self):
        """Finish augmentation and return all saved sequences"""
        if not self.saved_augmentations:
            result = messagebox.askyesno("No Augmentations", 
                                        "No augmentations were saved.\n"
                                        "Do you want to finish without augmentations?")
            if not result:
                return
        
        # Prepare result: list of tuples (sequence, angle)
        self.result_sequences = [(seq, angle) for angle, seq in self.saved_augmentations.items()]
        self.dialog.destroy()
    
    def _cancel_augmentation(self):
        """Cancel augmentation"""
        if self.saved_augmentations:
            result = messagebox.askyesno("Cancel?", 
                                        f"You have {len(self.saved_augmentations)} saved augmentation(s).\n"
                                        f"Do you want to discard them?")
            if not result:
                return
        
        self.result_sequences = []
        self.dialog.destroy()
    
    def show(self):
        """Show dialog and return saved augmentations"""
        self.dialog.wait_window()
        return self.result_sequences


class SequencePreviewDialog:
    """Dialog to preview and confirm sequence before saving"""
    
    def __init__(self, parent, sequence, gesture_id, diversity_score, similarities):
        self.result = None
        self.current_frame = 0
        self.is_playing = True   
        self.play_speed = 100
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Preview Sequence Before Saving")
        self.dialog.geometry("900x800")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.sequence = sequence
        self.gesture_id = gesture_id
        self.diversity_score = diversity_score
        self.similarities = similarities
        
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
        """Setup preview dialog UI"""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        info_frame = ttk.LabelFrame(main_frame, text="Sequence Information", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        info_text = f"Gesture ID: {self.gesture_id}\n"
        info_text += f"Diversity Score: {self.diversity_score:.3f}\n"
        
        if self.similarities:
            info_text += f"Most Similar Existing Sequence: {max(self.similarities):.3f}\n"
            info_text += f"Least Similar Existing Sequence: {min(self.similarities):.3f}"
        else:
            info_text += "No existing sequences for this gesture"
 
        if self.diversity_score < 0.3:
            warning_msg = "WARNING: Low diversity - very similar to existing data!"
            warning_color = "red"
        elif self.diversity_score < 0.5:
            warning_msg = "CAUTION: Moderate diversity - somewhat similar to existing data"
            warning_color = "orange"
        else:
            warning_msg = "Good diversity - sufficiently different from existing data"
            warning_color = "green"
        
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack()
        warning_label = tk.Label(info_frame, text=warning_msg, 
                                 fg=warning_color, font=('Arial', 9, 'bold'),
                                 pady=10)
        warning_label.pack()
        
        canvas_frame = ttk.LabelFrame(main_frame, text="Sequence Preview", padding="10")
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.canvas = tk.Canvas(canvas_frame, bg='black', height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack()
        
        ttk.Button(btn_frame, text="Prev", command=self._prev_frame).pack(side=tk.LEFT, padx=2)
        self.play_button = ttk.Button(btn_frame, text="Pause", command=self._toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Next", command=self._next_frame).pack(side=tk.LEFT, padx=2)
        
        self.frame_label = ttk.Label(control_frame, text=f"Frame: 1 / {len(self.sequence)}")
        self.frame_label.pack(pady=5)
        
        # Decision buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Save Sequence", 
                  command=self._save_sequence, style="Success.TButton").pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(button_frame, text="Discard Sequence", 
                  command=self._discard_sequence).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.dialog.bind('<Left>', lambda e: self._prev_frame())
        self.dialog.bind('<Right>', lambda e: self._next_frame())
        self.dialog.bind('<space>', lambda e: self._toggle_play())
        
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
            self.play_button.config(text="▶ Play")
    
    def _next_frame(self):
        """Go to next frame"""
        self.current_frame = (self.current_frame + 1) % len(self.sequence)
        self._visualize_frame()
    
    def _prev_frame(self):
        """Go to previous frame"""
        self.current_frame = (self.current_frame - 1) % len(self.sequence)
        self._visualize_frame()
    
    def _visualize_frame(self):
        """Visualize current frame"""
        self.canvas.delete("all")
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        frame_landmarks = self.sequence[self.current_frame].reshape(21, 3)
        landmarks_2d = frame_landmarks[:, :2]
        
        # Calculate global bounds for consistent scaling
        all_frames = self.sequence.reshape(-1, 21, 3)[:, :, :2]
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
                prev_frame = self.sequence[frame_idx].reshape(21, 3)[:, :2]
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
        
        self.frame_label.config(text=f"Frame: {self.current_frame + 1} / {len(self.sequence)}")
    
    def _save_sequence(self):
        """User chose to save"""
        self.result = True
        self.dialog.destroy()
    
    def _discard_sequence(self):
        """User chose to discard"""
        self.result = False
        self.dialog.destroy()
    
    def show(self):
        """Show dialog and return result"""
        self.dialog.wait_window()
        return self.result


class GestureDataCollector:
    
    def __init__(self, root, config):
        self.root = root
        self.root.title("Gesture Dataset Collector")
        self.root.geometry("1000x800")
        
        self.PALM_MODEL = path + "/models/palm_detection_full_quant.tflite"
        self.LANDMARK_MODEL = path + "/models/hand_landmark_full_quant.tflite"
        self.ANCHORS = path + "/models/anchors.csv"
        
        self.cap = None
        self.tracker = None 
        
        self.config = config
        self.sequence_length = config['sequence_length']
        self.dataset_path = config.get('dataset_path', 'gesture_dataset_seq.h5')
        
        self.dataset_manager = DatasetManager(sequence_length=self.sequence_length)
        
        if config['mode'] == 'load' and config['dataset_path']:
            try:
                self.dataset_manager.load_from_h5(config['dataset_path'])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
        
        self.current_frame = None
        self.is_running = False
        
        # Sequence recording state
        self.is_recording = False
        self.current_sequence = []
        self.frames_recorded = 0
        self.sequence_reference_wrist = None
        self.sequence_reference_scale = None
        
        # Tracking for consecutive null detections and interpolation
        self.consecutive_nulls = 0
        self.max_consecutive_nulls = 2
        self.last_valid_landmarks = None
        self.pending_interpolations = []
        
        self.pending_sequences = []
        
        self.current_gesture_id = tk.StringVar(value=str(config['selected_class']))
        self.camera_index = tk.IntVar(value=0)
        self.available_cameras = self._detect_cameras()
        
        self._setup_ui()
        self._initialize_tracker()
        self.root.bind('<space>', self._toggle_recording)
        
    def _detect_cameras(self):
        """Detect available cameras on Windows"""
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
        
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)
        control_frame.columnconfigure(3, weight=1)
        
        input_frame = ttk.LabelFrame(control_frame, text="Input", padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        ttk.Label(input_frame, text="Camera:").grid(row=0, column=0, sticky=tk.W, pady=2)
        camera_combo = ttk.Combobox(input_frame, textvariable=self.camera_index, 
                                     values=self.available_cameras, state="readonly", width=8)
        camera_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
        camera_combo.bind('<<ComboboxSelected>>', self._change_camera)
        
        ttk.Label(input_frame, text="Gesture ID:").grid(row=1, column=0, sticky=tk.W, pady=2)
        gesture_entry = ttk.Entry(input_frame, textvariable=self.current_gesture_id, width=8)
        gesture_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
        
        input_frame.columnconfigure(1, weight=1)
        
        # Recording frame
        record_frame = ttk.LabelFrame(control_frame, text="Recording", padding="10")
        record_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.record_button = ttk.Button(record_frame, text="Start Recording (Space)", 
                  command=self._toggle_recording)
        self.record_button.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(record_frame, variable=self.progress_var,
                                           maximum=self.sequence_length, mode='determinate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.frames_label = ttk.Label(record_frame, text=f"0 / {self.sequence_length} frames", 
                                     font=('Arial', 10, 'bold'))
        self.frames_label.grid(row=2, column=0, pady=2)
        
        self.skip_warning_label = ttk.Label(record_frame, text="", 
                                           foreground='orange', font=('Arial', 8))
        self.skip_warning_label.grid(row=3, column=0, pady=2)
        
        record_frame.columnconfigure(0, weight=1)
        
        button_frame = ttk.LabelFrame(control_frame, text="Actions", padding="10")
        button_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        ttk.Button(button_frame, text="Save & Exit", 
                  command=self._save_and_exit).grid(row=0, column=0, 
                                                   sticky=(tk.W, tk.E), pady=2)
        
        button_frame.columnconfigure(0, weight=1)
        
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding="10")
        stats_frame.grid(row=0, column=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.stats_label = ttk.Label(stats_frame, text="Total sequences: 0\nGesture classes: 0", 
                                     justify=tk.LEFT, anchor=tk.W)
        self.stats_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        stats_frame.columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar(value="Ready - Press Space or button to start recording")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self._update_statistics()
        
    def _initialize_tracker(self):
        """Initialize the hand tracker"""
        try:
            self.tracker = HandTracker(
                palm_detection_model=self.PALM_MODEL,
                hand_landmark_model=self.LANDMARK_MODEL,
                anchors=self.ANCHORS,
                num_hands=1
            )
            self._open_camera()
            self.is_running = True
            self._update_frame()
            self.status_var.set("Tracker initialized - Ready to record")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize tracker: {str(e)}")
            self.root.quit()
    
    def _open_camera(self):
        """Open the camera with high resolution"""
        if self.cap is not None:
            self.cap.release()
        
        camera_idx = self.camera_index.get()
        self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_idx}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width < 1920:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        print(f"Camera resolution: {actual_width}x{actual_height}")
    
    def _change_camera(self, event=None):
        """Change camera source"""
        self._open_camera()
        self.status_var.set(f"Switched to camera {self.camera_index.get()}")
    
    def _toggle_recording(self, event=None):
        """Toggle recording state"""
        if not self.is_recording:
            try:
                gesture_id = int(self.current_gesture_id.get())
                if gesture_id < 0:
                    raise ValueError("Gesture ID must be non-negative")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid gesture ID: {str(e)}")
                return
            
            self.is_recording = True
            self.current_sequence = []
            self.frames_recorded = 0
            self.sequence_reference_wrist = None
            self.sequence_reference_scale = None
            self.consecutive_nulls = 0
            self.last_valid_landmarks = None
            self.pending_interpolations = []
            self.progress_var.set(0)
            self.record_button.config(text="Stop Recording (Space)", style="Recording.TButton")
            self.status_var.set(f"Recording sequence for Gesture {gesture_id}...")
            self.skip_warning_label.config(text="")
        else:
            self._stop_recording()
    
    def _stop_recording(self):
        """Stop recording and preview sequence"""
        self.is_recording = False
        self.record_button.config(text="Start Recording (Space)")
        
        if len(self.current_sequence) == self.sequence_length:
            try:
                gesture_id = int(self.current_gesture_id.get())
                sequence_array = np.array(self.current_sequence)
                
                # Get existing sequences for this gesture
                existing_indices = self.dataset_manager.get_class_sequences(gesture_id)
                existing_sequences = [np.array(self.dataset_manager.data[idx]) for idx in existing_indices]
                
                diversity_score, similarities = calculate_dataset_diversity(
                    sequence_array, existing_sequences, self.pending_sequences
                )
                
                # Show preview dialog
                preview = SequencePreviewDialog(
                    self.root, sequence_array, gesture_id, 
                    diversity_score, similarities
                )
                should_save = preview.show()
                
                if should_save:
                    # Original sequence saved
                    self.pending_sequences.append(sequence_array)
                    self.dataset_manager.add_sequence(self.current_sequence, gesture_id)
                    self._update_statistics()
                    self.status_var.set(f"Sequence saved! Total: {len(self.dataset_manager.data)} (Diversity: {diversity_score:.3f})")
                    
                    # Automatically show augmentation dialog without prompt
                    self._show_augmentation_dialog(sequence_array, gesture_id)
                else:
                    self.status_var.set("Sequence discarded by user")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process sequence: {str(e)}")
        else:
            self.status_var.set(f"Recording stopped - incomplete sequence ({len(self.current_sequence)}/{self.sequence_length} frames)")
        
        self.current_sequence = []
        self.frames_recorded = 0
        self.consecutive_nulls = 0
        self.last_valid_landmarks = None
        self.pending_interpolations = []
        self.progress_var.set(0)
        self.skip_warning_label.config(text="")
    
    def _show_augmentation_dialog(self, sequence, gesture_id):
        """Show augmentation dialog"""
        aug_dialog = AugmentationDialog(
            self.root, sequence, gesture_id, 
            self.dataset_manager, self.pending_sequences
        )
        
        augmented_sequences = aug_dialog.show()
        
        if augmented_sequences:
            # Save all augmented sequences
            for aug_seq, angle in augmented_sequences:
                self.pending_sequences.append(aug_seq)
                self.dataset_manager.add_sequence(aug_seq, gesture_id)
            
            self._update_statistics()
            self.status_var.set(f"Saved {len(augmented_sequences)} augmented sequence(s)! "
                              f"Total: {len(self.dataset_manager.data)}")
            
        else:
            self.status_var.set("No augmentations saved")
    
    def _discard_sequence(self, reason):
        """Discard the current sequence being recorded"""
        self.is_recording = False
        self.record_button.config(text="Start Recording (Space)")
        self.current_sequence = []
        self.frames_recorded = 0
        self.consecutive_nulls = 0
        self.last_valid_landmarks = None
        self.pending_interpolations = []
        self.progress_var.set(0)
        self.skip_warning_label.config(text="")
        
        self.status_var.set(f"Sequence DISCARDED: {reason}")
        messagebox.showwarning("Sequence Discarded", 
                              f"The current sequence was discarded.\n\nReason: {reason}\n\n"
                              f"Please try recording again with steadier hand visibility.")
    
    def _perform_linear_interpolation(self, start_landmarks, end_landmarks, num_steps):
        """Perform linear interpolation between two landmark sets"""
        interpolated = []
        for i in range(1, num_steps + 1):
            t = i / (num_steps + 1)
            interpolated_frame = start_landmarks + t * (end_landmarks - start_landmarks)
            interpolated.append(interpolated_frame)
        
        return interpolated
    
    def _update_frame(self):
        """Update video frame"""
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            
            detections = self.tracker(frame)
            
            has_hand = False
            if detections:
                for landmarks, hand_bbox, handedness in detections:
                    has_hand = True
                    self._draw_landmarks(landmarks, frame)
                    self._draw_bbox(hand_bbox, frame)
                    hand_type = "Right" if handedness > 0.5 else "Left"
                    cv2.putText(frame, f"Hand: {hand_type}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if self.is_recording and len(self.current_sequence) < self.sequence_length:
                        if self.pending_interpolations and self.last_valid_landmarks is not None:
                            current_normalized = self._normalize_landmarks_for_sequence(landmarks)
                            num_missing = len(self.pending_interpolations)
                            
                            interpolated_frames = self._perform_linear_interpolation(
                                self.last_valid_landmarks,
                                current_normalized,
                                num_missing
                            )
                            
                            for idx, interp_frame in zip(self.pending_interpolations, interpolated_frames):
                                self.current_sequence[idx] = interp_frame
                            
                            print(f"Interpolated {num_missing} missing frame(s)")
                            self.pending_interpolations = []
                        
                        self.consecutive_nulls = 0
                        self.skip_warning_label.config(text="")
                        
                        normalized_landmarks = self._normalize_landmarks_for_sequence(landmarks)
                        self.current_sequence.append(normalized_landmarks)
                        self.last_valid_landmarks = normalized_landmarks
                        
                        self.frames_recorded = len(self.current_sequence)
                        self.progress_var.set(self.frames_recorded)
                        self.frames_label.config(text=f"{self.frames_recorded} / {self.sequence_length} frames")
                        
                        if len(self.current_sequence) >= self.sequence_length:
                            self._stop_recording()
            
            if self.is_recording and not has_hand and len(self.current_sequence) < self.sequence_length:
                self.consecutive_nulls += 1
                
                if self.consecutive_nulls == 1:
                    self.skip_warning_label.config(text="Missing detection (1)")
                elif self.consecutive_nulls == 2:
                    self.skip_warning_label.config(text="Missing detection (2)")
                
                if self.consecutive_nulls > self.max_consecutive_nulls:
                    self._discard_sequence(f"Too many consecutive missing detections ({self.consecutive_nulls})")
                else:
                    if self.last_valid_landmarks is not None:
                        placeholder = self.last_valid_landmarks.copy()
                        self.current_sequence.append(placeholder)
                        self.pending_interpolations.append(len(self.current_sequence) - 1)
                        
                        self.frames_recorded = len(self.current_sequence)
                        self.progress_var.set(self.frames_recorded)
                        self.frames_label.config(text=f"{self.frames_recorded} / {self.sequence_length} frames (pending interp)")
                        
                        if len(self.current_sequence) >= self.sequence_length:
                            self._stop_recording()
            
            if self.is_recording:
                if has_hand:
                    color = (0, 0, 255)
                    text = "RECORDING"
                elif self.consecutive_nulls > 0:
                    color = (0, 165, 255)
                    text = f"WAITING FOR HAND ({self.consecutive_nulls}/{self.max_consecutive_nulls})"
                else:
                    color = (0, 165, 255)
                    text = "WAITING FOR HAND"
                
                cv2.circle(frame, (30, 70), 15, color, -1)
                cv2.putText(frame, text, (55, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Frame: {len(self.current_sequence)}/{self.sequence_length}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
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
    
    def _normalize_landmarks_for_sequence(self, landmarks):
        """Normalize landmarks while preserving relative motion and shape (scale-invariant)"""
        if len(self.current_sequence) == 0:
            self.sequence_reference_wrist = landmarks[0].copy()
            
            distances = np.linalg.norm(landmarks - landmarks[0], axis=1)
            hand_span = np.max(distances)
            
            if hand_span < 0.001:
                hand_span = 1.0
            
            self.sequence_reference_scale = hand_span
            print(f"Sequence reference scale (hand span): {hand_span:.4f}")
        
        normalized = landmarks - self.sequence_reference_wrist
        normalized = normalized / self.sequence_reference_scale
        
        wrist_pos = normalized[0]
        if len(self.current_sequence) % 3 == 0:
            print(f"Frame {len(self.current_sequence)}: Wrist position = ({wrist_pos[0]:.3f}, {wrist_pos[1]:.3f}, {wrist_pos[2]:.3f})")
        
        return normalized.flatten().astype(np.float32)
    
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
            cv2.circle(frame, (int(x), int(y)), 8, color, -1)
    
    def _draw_bbox(self, hand_bbox, frame):
        """Draw bounding box"""
        center = hand_bbox.center
        dims = hand_bbox.dims
        rotation = hand_bbox.rotation * 180 / np.pi
        
        rect = (center, dims, rotation)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
    
    def _update_statistics(self):
        """Update statistics display"""
        stats = self.dataset_manager.get_statistics()
        counts = [str(count) for _, count in sorted(stats['sequences_per_class'].items())]
        counts_str = ', '.join(counts) if counts else '0'
        
        stats_text = f"Total sequences: {stats['total_sequences']}\n"
        stats_text += f"Gesture classes: {stats['num_classes']}\n"
        stats_text += f"Sequences per class:\n{counts_str}"
        
        self.stats_label.config(text=stats_text)
    
    def _save_dataset(self):
        """Save dataset"""
        if len(self.dataset_manager.data) == 0:
            messagebox.showwarning("Warning", "No data collected yet")
            return False
        
        filename = self.dataset_path if self.config['mode'] == 'load' else "datasets/gesture_dataset_seq1.h5"
        
        try:
            self.dataset_manager.save_to_h5(filename)
            
            # Clear pending sequences after successful save
            self.pending_sequences = []
             
            messagebox.showinfo("Success", 
                               f"Dataset saved to {filename}\n"
                               f"Total sequences: {len(self.dataset_manager.data)}")
            self.status_var.set(f"Dataset saved to {filename}")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save dataset: {str(e)}")
            return False
    
    def _save_and_exit(self):
        """Save dataset and exit"""
        if len(self.dataset_manager.data) == 0:
            result = messagebox.askyesno("Exit", "No data saved. Exit anyway?")
            if result:
                self._cleanup()
                self.root.quit()
        else:
            if self._save_dataset(): 
                self._cleanup()
                self.root.quit()
    
    def _exit_app(self):
        """Exit application"""
        if len(self.dataset_manager.data) == 0:
            result = messagebox.askyesno("Exit", "No data saved. Exit anyway?")
            if not result:
                return
        else:
            result = messagebox.askyesnocancel("Exit", 
                                               "Do you want to save before exiting?")
            if result is None:
                return
            elif result:
                self._save_dataset()
        
        self._cleanup()
        self.root.quit()
    
    def _cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__": 
    config = show_dataset_selector() 
    
    if config['mode'] is None:
        exit()
    
    root = tk.Tk()
    app = GestureDataCollector(root, config)
    
    def on_closing():
        app._exit_app()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()