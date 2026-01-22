import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import h5py


class DatasetSelector:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Selection")
        self.root.geometry("400x200")
        
        self.selected_mode = None  # 'new' or 'load'
        self.dataset_path = None
        self.selected_class = None
        self.sequence_length = 10
        
        self._setup_ui()
        self._center_window()
    
    def _center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def _setup_ui(self):
        """Setup the user interface"""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, text="Gesture Dataset Collector", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 30))
        
        ttk.Button(main_frame, text="Create New Dataset", 
                  command=self._create_new_dataset).pack(fill=tk.X, pady=5)
        
        ttk.Button(main_frame, text="Load Existing Dataset", 
                  command=self._load_dataset).pack(fill=tk.X, pady=5)
    
    def _create_new_dataset(self):
        """Handle new dataset creation"""
        self.selected_mode = 'new'
        self.sequence_length = 10
        self.selected_class = 0
        self.root.quit()
    
    def _load_dataset(self):
        """Browse and load existing dataset file"""
        filename = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Load dataset info
            with h5py.File(filename, 'r') as hf:
                self.sequence_length = hf.attrs.get('sequence_length', 10)
                labels = hf['labels'][:]
                classes = set(labels)
                self.selected_class = max(classes) + 1 if classes else 0
            
            self.dataset_path = filename
            self.selected_mode = 'load'
            self.root.quit()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")
    
    def get_result(self):
        """Get the user's selection"""
        return {
            'mode': self.selected_mode,
            'dataset_path': self.dataset_path,
            'sequence_length': self.sequence_length,
            'selected_class': self.selected_class,
            'dataset_info': None
        }


def show_dataset_selector():
    """Show dataset selector and return user's choice"""
    root = tk.Tk()
    selector = DatasetSelector(root)
    
    def on_closing():
        root.quit()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
    
    result = selector.get_result()
    root.destroy()
    
    return result


if __name__ == "__main__":
    result = show_dataset_selector()
    print(f"Selected mode: {result['mode']}")
    print(f"Dataset path: {result['dataset_path']}")
    print(f"Sequence length: {result['sequence_length']}")
    print(f"Selected class: {result['selected_class']}")