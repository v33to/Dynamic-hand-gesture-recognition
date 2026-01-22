import os
import numpy as np
import h5py
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from datetime import datetime

path = os.getcwd()

# Configuration
SEQUENCE_LENGTH = 10
MAX_SLIDING_WINDOWS = 7
FONT_CONFIG = {
    'title_size': 14,
    'axis_label_size': 12,
    'tick_label_size': 14,
    'legend_size': 10,
    'annotation_size': 14,
    'suptitle_size': 16
}

# Model configurations
MODEL_CONFIGS = {
    'Baseline Model': {
        'model_path': path + "/trained_models_initial/gesture_conv1d_quant.tflite",
        'labels_path': path + "/trained_models_initial/gesture_labels.txt",
        'has_unknown_class': False,
        'unknown_confidence_threshold': 0.9,
        'unknown_entropy_threshold': 0.5,
        'description': "Baseline model (pre-active learning, no explicit unknown class)"
    },
    'Enhanced Model': {
        'model_path': path + "/trained_models/gesture_conv1d_quant.tflite",
        'labels_path': path + "/trained_models/gesture_labels.txt",
        'has_unknown_class': True,
        'unknown_confidence_threshold': 0.9,
        'unknown_entropy_threshold': 0.5,
        'description': "Enhanced model (post-active learning, explicit unknown class)"
    }
}

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


class GestureClassifier:
    """Wrapper for TFLite gesture classifier"""
    
    def __init__(self, model_path, labels_path, has_unknown_class,
                 unknown_confidence_threshold, unknown_entropy_threshold):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.has_unknown_class = has_unknown_class
        self.unknown_confidence_threshold = unknown_confidence_threshold
        self.unknown_entropy_threshold = unknown_entropy_threshold
        
        self.sequence_length = self.input_details[0]['shape'][1]
        self.feature_size = self.input_details[0]['shape'][2]
        
        self.labels = []
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        
        self.num_classes = len(self.labels)
        
    def predict_window(self, sequence):
        """Predict a single window"""
        if len(sequence.shape) == 2:
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
        else:
            input_data = sequence.astype(np.float32)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        predictions = output_data[0]
        max_confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)
        
        # Calculate entropy
        entropy = -np.sum(predictions * np.log(predictions + 1e-10))
        normalized_entropy = entropy / np.log(len(predictions))
        
        # Determine if unknown based on model type
        if self.has_unknown_class:
            # Model has explicit unknown class (class 8)
            # Check if it predicted unknown OR meets uncertainty thresholds
            is_unknown_by_prediction = (predicted_class == 8)
            is_unknown_by_uncertainty = (
                (max_confidence < self.unknown_confidence_threshold) or 
                (normalized_entropy > self.unknown_entropy_threshold)
            )
            is_unknown = is_unknown_by_prediction or is_unknown_by_uncertainty
            
            # If model predicts known class but uncertainty is high, override to unknown
            if not is_unknown_by_prediction and is_unknown_by_uncertainty:
                predicted_class = 8
        else:
            # Model without explicit unknown class - use only thresholds
            is_unknown = (
                (max_confidence < self.unknown_confidence_threshold) or 
                (normalized_entropy > self.unknown_entropy_threshold)
            )
            # Map to unknown class ID (8) if unknown
            if is_unknown:
                predicted_class = 8
        
        return {
            'predictions': predictions,
            'predicted_class': predicted_class,
            'max_confidence': max_confidence,
            'entropy': normalized_entropy,
            'is_unknown': is_unknown,
            'raw_predicted_class': np.argmax(predictions)
        }
    
    def predict_gesture_with_voting(self, windows):
        """Predict using majority voting across windows"""
        all_results = []
        valid_predictions = []
        
        for window in windows:
            result = self.predict_window(window)
            all_results.append(result)
            
            # Only consider non-unknown predictions for voting
            if not result['is_unknown']:
                valid_predictions.append({
                    'class': result['predicted_class'],
                    'confidence': result['max_confidence'],
                    'entropy': result['entropy']
                })
        
        # If no valid predictions, return best unknown result
        if not valid_predictions:
            best_result = max(all_results, key=lambda x: x['max_confidence'])
            return {
                'predicted_class': 8,
                'max_confidence': best_result['max_confidence'],
                'entropy': best_result['entropy'],
                'is_unknown': True,
                'num_windows': len(windows),
                'num_valid': 0,
                'voting_results': None,
                'all_predictions': [r['predicted_class'] for r in all_results]
            }
        
        # Majority voting among valid predictions
        class_votes = [p['class'] for p in valid_predictions]
        vote_counts = Counter(class_votes)
        
        max_votes = max(vote_counts.values())
        top_classes = [cls for cls, count in vote_counts.items() if count == max_votes]
        
        # Break ties by confidence
        if len(top_classes) > 1:
            tied_predictions = [p for p in valid_predictions if p['class'] in top_classes]
            final_prediction = max(tied_predictions, key=lambda x: x['confidence'])
        else:
            class_predictions = [p for p in valid_predictions if p['class'] == top_classes[0]]
            final_prediction = max(class_predictions, key=lambda x: x['confidence'])
        
        return {
            'predicted_class': final_prediction['class'],
            'max_confidence': final_prediction['confidence'],
            'entropy': final_prediction['entropy'],
            'is_unknown': False,
            'num_windows': len(windows),
            'num_valid': len(valid_predictions),
            'voting_results': dict(vote_counts),
            'all_predictions': [r['predicted_class'] for r in all_results]
        }


class ModelBenchmark:
    """Benchmark and compare two gesture recognition models"""
    
    def __init__(self, dataset_path, output_dir='benchmark_results'):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.gestures = []
        self.labels = []
        self.num_gestures = 0
        
        self.models = {}
        self.results = {}
        
        self.report_file = os.path.join(output_dir, f'benchmark_report.txt')
        
    def load_dataset(self):
        """Load continuous gesture dataset"""
        print(f"Loading dataset from {self.dataset_path}")
        
        with h5py.File(self.dataset_path, 'r') as f:
            gesture_grp = f['gestures']
            self.num_gestures = f.attrs['num_gestures']
            
            for idx in range(self.num_gestures):
                gesture_id = f'gesture_{idx}'
                g = gesture_grp[gesture_id]
                
                windows = g['windows'][:]
                label = g.attrs['label']
                num_windows = g.attrs['num_windows']
                
                self.gestures.append({
                    'windows': windows,
                    'label': label,
                    'num_windows': num_windows,
                    'gesture_id': idx
                })
                self.labels.append(label)
        
        print(f"Loaded {self.num_gestures} gestures")
        print(f"Label distribution: {Counter(self.labels)}")
        print()
    
    def load_models(self, model_configs):
        """Load all models to benchmark"""
        for model_name, config in model_configs.items():
            print(f"Loading model: {model_name}")
            print(f"  Description: {config['description']}")
            print(f"  Model: {config['model_path']}")
            
            classifier = GestureClassifier(
                model_path=config['model_path'],
                labels_path=config['labels_path'],
                has_unknown_class=config['has_unknown_class'],
                unknown_confidence_threshold=config['unknown_confidence_threshold'],
                unknown_entropy_threshold=config['unknown_entropy_threshold']
            )
            
            self.models[model_name] = {
                'classifier': classifier,
                'config': config
            }
            
            print(f"  Loaded successfully: {classifier.num_classes} classes")
            print()
    
    def evaluate_model(self, model_name):
        """Evaluate a single model on the dataset"""
        print(f"Evaluating model: {model_name}")
        print("=" * 60)
        
        model_info = self.models[model_name]
        classifier = model_info['classifier']
        
        predictions = []
        confidences = []
        entropies = []
        voting_details = []
        all_window_predictions = []
        
        for gesture_data in self.gestures:
            windows = gesture_data['windows']
            true_label = gesture_data['label']
            
            # Predict using majority voting
            result = classifier.predict_gesture_with_voting(windows)
            
            predictions.append(result['predicted_class'])
            confidences.append(result['max_confidence'])
            entropies.append(result['entropy'])
            voting_details.append(result)
            all_window_predictions.append(result['all_predictions'])
        
        # Calculate metrics
        true_labels = np.array(self.labels)
        pred_labels = np.array(predictions)
        
        accuracy = accuracy_score(true_labels, pred_labels)
        
        # Calculate overall precision and recall (weighted)
        precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
        
        # Per-class accuracy
        per_class_accuracy = {}
        per_class_counts = {}
        for class_id in np.unique(true_labels):
            mask = true_labels == class_id
            class_acc = accuracy_score(true_labels[mask], pred_labels[mask])
            per_class_accuracy[int(class_id)] = class_acc
            per_class_counts[int(class_id)] = int(np.sum(mask))
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        
        # Classification report
        class_report = classification_report(
            true_labels, pred_labels,
            labels=list(range(9)),
            target_names=[GESTURE_NAMES.get(i, f'Class {i}') for i in range(9)],
            zero_division=0,
            output_dict=True
        )
        
        # Unknown detection analysis
        true_unknowns = true_labels == 8
        pred_unknowns = pred_labels == 8
        
        unknown_stats = {
            'true_unknown_count': int(np.sum(true_unknowns)),
            'predicted_unknown_count': int(np.sum(pred_unknowns)),
            'true_positives': int(np.sum(true_unknowns & pred_unknowns)),
            'false_positives': int(np.sum(~true_unknowns & pred_unknowns)),
            'false_negatives': int(np.sum(true_unknowns & ~pred_unknowns)),
            'true_negatives': int(np.sum(~true_unknowns & ~pred_unknowns))
        }
        
        if unknown_stats['true_unknown_count'] > 0:
            unknown_stats['recall'] = unknown_stats['true_positives'] / unknown_stats['true_unknown_count']
        else:
            unknown_stats['recall'] = None
        
        if unknown_stats['predicted_unknown_count'] > 0:
            unknown_stats['precision'] = unknown_stats['true_positives'] / unknown_stats['predicted_unknown_count']
        else:
            unknown_stats['precision'] = None
        
        # Confidence analysis
        confidence_stats = {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        }
        
        entropy_stats = {
            'mean': float(np.mean(entropies)),
            'std': float(np.std(entropies)),
            'min': float(np.min(entropies)),
            'max': float(np.max(entropies))
        }
        
        self.results[model_name] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'per_class_accuracy': per_class_accuracy,
            'per_class_counts': per_class_counts,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'unknown_stats': unknown_stats,
            'confidence_stats': confidence_stats,
            'entropy_stats': entropy_stats,
            'predictions': predictions,
            'confidences': confidences,
            'entropies': entropies,
            'voting_details': voting_details,
            'all_window_predictions': all_window_predictions
        }
        
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall: {recall:.4f}")
        print()
    
    def compare_models(self):
        """Compare results between models"""
        print("=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print()
        
        # Overall metrics comparison
        print("Overall Metrics:")
        for model_name in self.results.keys():
            acc = self.results[model_name]['accuracy']
            prec = self.results[model_name]['precision']
            rec = self.results[model_name]['recall']
            print(f"  {model_name}:")
            print(f"    Accuracy:  {acc:.4f}")
            print(f"    Precision: {prec:.4f}")
            print(f"    Recall:    {rec:.4f}")
        
        if len(self.results) == 2:
            model_names = list(self.results.keys())
            acc_diff = self.results[model_names[1]]['accuracy'] - self.results[model_names[0]]['accuracy']
            prec_diff = self.results[model_names[1]]['precision'] - self.results[model_names[0]]['precision']
            rec_diff = self.results[model_names[1]]['recall'] - self.results[model_names[0]]['recall']
            print(f"  Difference:")
            print(f"    Accuracy:  {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")
            print(f"    Precision: {prec_diff:+.4f} ({prec_diff*100:+.2f}%)")
            print(f"    Recall:    {rec_diff:+.4f} ({rec_diff*100:+.2f}%)")
        print()
        
        # Per-class comparison
        print("Per-Class Accuracy:")
        all_classes = set()
        for result in self.results.values():
            all_classes.update(result['per_class_accuracy'].keys())
        
        for class_id in sorted(all_classes):
            class_name = GESTURE_NAMES.get(class_id, f'Class {class_id}')
            print(f"  {class_name} (Class {class_id}):")
            
            for model_name in self.results.keys():
                if class_id in self.results[model_name]['per_class_accuracy']:
                    acc = self.results[model_name]['per_class_accuracy'][class_id]
                    count = self.results[model_name]['per_class_counts'][class_id]
                    print(f"    {model_name}: {acc:.4f} ({count} samples)")
                else:
                    print(f"    {model_name}: N/A (0 samples)")
            
            if len(self.results) == 2:
                model_names = list(self.results.keys())
                if (class_id in self.results[model_names[0]]['per_class_accuracy'] and 
                    class_id in self.results[model_names[1]]['per_class_accuracy']):
                    diff = (self.results[model_names[1]]['per_class_accuracy'][class_id] - 
                           self.results[model_names[0]]['per_class_accuracy'][class_id])
                    print(f"    Difference: {diff:+.4f}")
            print()
        
        # Unknown detection comparison
        print("Unknown Detection Performance:")
        for model_name in self.results.keys():
            stats = self.results[model_name]['unknown_stats']
            print(f"  {model_name}:")
            print(f"    True unknowns in dataset: {stats['true_unknown_count']}")
            print(f"    Predicted as unknown: {stats['predicted_unknown_count']}")
            print(f"    True positives: {stats['true_positives']}")
            print(f"    False positives: {stats['false_positives']}")
            print(f"    False negatives: {stats['false_negatives']}")
            if stats['precision'] is not None:
                print(f"    Precision: {stats['precision']:.4f}")
            if stats['recall'] is not None:
                print(f"    Recall: {stats['recall']:.4f}")
            if stats['precision'] is not None and stats['recall'] is not None:
                f1 = 2 * (stats['precision'] * stats['recall']) / (stats['precision'] + stats['recall'] + 1e-10)
                print(f"    F1-Score: {f1:.4f}")
        print()
        
        # Confidence and entropy comparison
        print("Confidence Statistics:")
        for model_name in self.results.keys():
            stats = self.results[model_name]['confidence_stats']
            print(f"  {model_name}:")
            print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print()
        
        print("Entropy Statistics:")
        for model_name in self.results.keys():
            stats = self.results[model_name]['entropy_stats']
            print(f"  {model_name}:")
            print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print()
    
    def generate_visualizations(self):
        """Generate confusion matrix visualization"""
        print("Generating confusion matrix visualization...")
        
        num_models = len(self.results)
        model_names = list(self.results.keys())
        
        # Confusion matrices side by side with precision/recall
        fig, axes = plt.subplots(1, num_models, figsize=(8*num_models, 6))
        if num_models == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(model_names):
            conf_matrix = np.array(self.results[model_name]['confusion_matrix'])
            acc = self.results[model_name]['accuracy']
            prec = self.results[model_name]['precision']
            rec = self.results[model_name]['recall']
            
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar=False,
                       xticklabels=[GESTURE_NAMES.get(i, f'{i}')[:10] for i in range(9)],
                       yticklabels=[GESTURE_NAMES.get(i, f'{i}')[:10] for i in range(9)],
                       annot_kws={'size': FONT_CONFIG['annotation_size']})
            
            title_text = f'{model_name}\nAcc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}'
            axes[idx].set_title(title_text, fontweight='bold', fontsize=FONT_CONFIG['title_size'])
            axes[idx].set_xlabel('Predicted', fontweight='bold', fontsize=FONT_CONFIG['axis_label_size'])
            axes[idx].set_ylabel('True', fontweight='bold', fontsize=FONT_CONFIG['axis_label_size'])
            axes[idx].tick_params(axis='both', labelsize=FONT_CONFIG['tick_label_size'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices.png'), dpi=150)
        print(f"  Saved: confusion_matrices.png")
        plt.close()
        print()
    
    def save_report(self):
        """Save detailed text report"""
        with open(self.report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GESTURE RECOGNITION MODEL BENCHMARK REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Total gestures: {self.num_gestures}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model configurations
            f.write("=" * 80 + "\n")
            f.write("MODEL CONFIGURATIONS\n")
            f.write("=" * 80 + "\n\n")
            
            for model_name, model_info in self.models.items():
                config = model_info['config']
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  Description: {config['description']}\n")
                f.write(f"  Model path: {config['model_path']}\n")
                f.write(f"  Has unknown class: {config['has_unknown_class']}\n")
                f.write(f"  Confidence threshold: {config['unknown_confidence_threshold']}\n")
                f.write(f"  Entropy threshold: {config['unknown_entropy_threshold']}\n\n")
            
            # Results for each model
            for model_name in self.results.keys():
                result = self.results[model_name]
                
                f.write("=" * 80 + "\n")
                f.write(f"RESULTS: {model_name.upper()}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("Overall Metrics:\n")
                f.write(f"  Accuracy:  {result['accuracy']:.4f}\n")
                f.write(f"  Precision: {result['precision']:.4f}\n")
                f.write(f"  Recall:    {result['recall']:.4f}\n\n")
                
                f.write("Per-Class Performance:\n")
                for class_id in sorted(result['per_class_accuracy'].keys()):
                    class_name = GESTURE_NAMES.get(class_id, f'Class {class_id}')
                    acc = result['per_class_accuracy'][class_id]
                    count = result['per_class_counts'][class_id]
                    f.write(f"  {class_name} (Class {class_id}): {acc:.4f} ({count} samples)\n")
                f.write("\n")
                
                f.write("Unknown Detection:\n")
                stats = result['unknown_stats']
                f.write(f"  True unknowns: {stats['true_unknown_count']}\n")
                f.write(f"  Predicted unknowns: {stats['predicted_unknown_count']}\n")
                f.write(f"  True positives: {stats['true_positives']}\n")
                f.write(f"  False positives: {stats['false_positives']}\n")
                f.write(f"  False negatives: {stats['false_negatives']}\n")
                if stats['precision'] is not None:
                    f.write(f"  Precision: {stats['precision']:.4f}\n")
                if stats['recall'] is not None:
                    f.write(f"  Recall: {stats['recall']:.4f}\n")
                f.write("\n")
                
                f.write("Confidence Statistics:\n")
                conf = result['confidence_stats']
                f.write(f"  Mean: {conf['mean']:.4f} ± {conf['std']:.4f}\n")
                f.write(f"  Range: [{conf['min']:.4f}, {conf['max']:.4f}]\n\n")
                
                f.write("Entropy Statistics:\n")
                ent = result['entropy_stats']
                f.write(f"  Mean: {ent['mean']:.4f} ± {ent['std']:.4f}\n")
                f.write(f"  Range: [{ent['min']:.4f}, {ent['max']:.4f}]\n\n")
            
            # Comparison
            if len(self.results) > 1:
                f.write("=" * 80 + "\n")
                f.write("MODEL COMPARISON SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                
                model_names = list(self.results.keys())
                
                f.write("Overall Metrics:\n")
                for model_name in model_names:
                    f.write(f"  {model_name}:\n")
                    f.write(f"    Accuracy:  {self.results[model_name]['accuracy']:.4f}\n")
                    f.write(f"    Precision: {self.results[model_name]['precision']:.4f}\n")
                    f.write(f"    Recall:    {self.results[model_name]['recall']:.4f}\n")
                
                if len(model_names) == 2:
                    acc_diff = self.results[model_names[1]]['accuracy'] - self.results[model_names[0]]['accuracy']
                    prec_diff = self.results[model_names[1]]['precision'] - self.results[model_names[0]]['precision']
                    rec_diff = self.results[model_names[1]]['recall'] - self.results[model_names[0]]['recall']
                    f.write(f"  Improvement:\n")
                    f.write(f"    Accuracy:  {acc_diff:+.4f} ({acc_diff*100:+.2f}%)\n")
                    f.write(f"    Precision: {prec_diff:+.4f} ({prec_diff*100:+.2f}%)\n")
                    f.write(f"    Recall:    {rec_diff:+.4f} ({rec_diff*100:+.2f}%)\n")
                f.write("\n")
        
        print(f"Report saved to: {self.report_file}")
    
    
    def run_benchmark(self):
        """Run complete benchmark pipeline"""
        print("\n" + "=" * 80)
        print("GESTURE RECOGNITION MODEL BENCHMARK")
        print("=" * 80 + "\n")
        
        self.load_dataset()
        self.load_models(MODEL_CONFIGS)
        
        # Evaluate each model
        for model_name in self.models.keys():
            self.evaluate_model(model_name)
        
        self.compare_models()
        self.generate_visualizations()
        self.save_report()
        
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {self.output_dir}/")
        print(f"  - Text report: {os.path.basename(self.report_file)}")
        print(f"  - Confusion matrix: confusion_matrices.png")
        print()


if __name__ == "__main__":
    # Configuration
    dataset_path = path + "/datasets/continuous_gestures_labeled.h5"
    output_dir = path + "/benchmark_results"
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please run bench.py first to create the labeled dataset.")
        exit(1)
    
    for model_name, config in MODEL_CONFIGS.items():
        if not os.path.exists(config['model_path']):
            print(f"ERROR: Model not found for {model_name}: {config['model_path']}")
            exit(1)
    
    # Run benchmark
    benchmark = ModelBenchmark(dataset_path, output_dir)
    benchmark.run_benchmark()