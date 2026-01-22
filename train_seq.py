import os
import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import json

path = os.getcwd()

class GestureConv1DTrainer:
    
    def __init__(self, dataset_path, output_dir='trained_models'):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.num_classes = None
        self.sequence_length = None
        self.feature_size = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.training_metrics = {}
        
    def load_dataset(self, val_split=0.15, test_split=0.15):
        """Load and split sequence dataset"""
        print(f"Loading dataset from {self.dataset_path}")
        
        with h5py.File(self.dataset_path, 'r') as f:
            sequences = f['sequences'][:]
            labels = f['labels'][:]
            self.sequence_length = f.attrs['sequence_length']
            self.feature_size = f.attrs['feature_size']
            
        print(f"Loaded {len(sequences)} sequences")
        print(f"Sequence length: {self.sequence_length} frames")
        print(f"Feature size: {self.feature_size}")
        print(f"Unique classes: {np.unique(labels)}")
        
        labels_encoded = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
        
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            sequences, labels_encoded, test_size=test_split, random_state=42, stratify=labels_encoded
        )
        
        val_size = val_split / (1 - test_split)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
        
        print(f"\nTrain samples: {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_val)}")
        print(f"Test samples: {len(self.X_test)}")
        
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_val = keras.utils.to_categorical(self.y_val, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        
    def build_model(self, conv_filters=[64, 128, 128], dense_units=[64, 32], dropout_rate=0.3):
        """Build 1D CNN model for sequence classification"""
        model = keras.Sequential()
        
        model.add(layers.Input(shape=(self.sequence_length, self.feature_size)))
        
        for i, filters in enumerate(conv_filters):
            model.add(layers.Conv1D(filters, kernel_size=3, padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            model.add(layers.Dropout(dropout_rate * 0.5))
            
            if i > 0:
                model.add(layers.MaxPooling1D(pool_size=2))
        
        # Global pooling to aggregate temporal features
        model.add(layers.GlobalAveragePooling1D())
        
        for units in dense_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        self.model = model
        print("\nModel Architecture:")
        model.summary()
        
        total_params = model.count_params()
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Estimated model size: {total_params * 4 / 1024:.2f} KB (float32)")
        
    def train(self, epochs=100, batch_size=32, learning_rate=0.001):
        """Train the model"""
        print(f"\nTraining for {epochs} epochs with batch size {batch_size}...")
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.output_dir, 'best_conv1d_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Per-class accuracy
        predictions = self.model.predict(self.X_test, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(self.y_test, axis=1)
        
        per_class_metrics = {}
        print("\nPer-class accuracy:")
        for cls in range(self.num_classes):
            mask = true_classes == cls
            if np.sum(mask) > 0:
                cls_accuracy = np.mean(pred_classes[mask] == true_classes[mask])
                cls_name = str(self.label_encoder.classes_[cls])
                num_samples = int(np.sum(mask))
                print(f"  Class {cls_name}: {cls_accuracy:.4f} ({num_samples} samples)")
                per_class_metrics[cls_name] = {
                    'accuracy': float(cls_accuracy),
                    'num_samples': num_samples
                }
        
        # Store test metrics
        self.training_metrics['test'] = {
            'loss': float(test_loss),
            'accuracy': float(test_accuracy),
            'per_class': per_class_metrics
        }
        
        return test_loss, test_accuracy
    
    def save_training_metrics(self, history):
        """Save all training metrics to a single JSON file"""
        print("\nSaving training metrics...")
        
        # Build complete metrics dictionary
        all_metrics = {
            'model_config': {
                'num_classes': int(self.num_classes),
                'sequence_length': int(self.sequence_length),
                'feature_size': int(self.feature_size),
                'class_names': self.label_encoder.classes_.tolist(),
                'model_type': 'Conv1D'
            },
            'epoch_metrics': {
                'epoch': list(range(1, len(history.history['loss']) + 1)),
                'train_loss': [float(x) for x in history.history['loss']],
                'train_accuracy': [float(x) for x in history.history['accuracy']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']]
            },
            'summary': {
                'epochs_completed': len(history.history['loss']),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_train_accuracy': float(history.history['accuracy'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'best_val_accuracy': float(max(history.history['val_accuracy'])),
                'best_val_loss': float(min(history.history['val_loss']))
            }
        }
        
        # Add test metrics if available
        if hasattr(self, 'training_metrics') and 'test' in self.training_metrics:
            all_metrics['test_metrics'] = self.training_metrics['test']
        
        # Save to single JSON file
        metrics_path = os.path.join(self.output_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"All training metrics saved to {metrics_path}")
    
    def plot_training_history(self, history=None, json_path=None, save_path=None):
        """Plot training history from history object or JSON file"""
        if json_path:
            # Load from JSON
            print(f"Loading metrics from {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
            history_dict = {
                'accuracy': data['epoch_metrics']['train_accuracy'],
                'val_accuracy': data['epoch_metrics']['val_accuracy'],
                'loss': data['epoch_metrics']['train_loss'],
                'val_loss': data['epoch_metrics']['val_loss']
            }
        elif history:
            # Use history object
            history_dict = history.history
        else:
            print("Error: Either history object or json_path must be provided")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(history_dict['accuracy'], label='Train', linewidth=2)
        ax1.plot(history_dict['val_accuracy'], label='Validation', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history_dict['loss'], label='Train', linewidth=2)
        ax2.plot(history_dict['val_loss'], label='Validation', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def convert_to_tflite(self, quantize=True):
        """Convert model to TFLite format"""
        print("\nConverting to TFLite format...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            print("Applying dynamic range quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            def representative_dataset():
                for i in range(min(100, len(self.X_train))):
                    data = self.X_train[i:i+1].astype(np.float32)
                    yield [data]
            
            converter.representative_dataset = representative_dataset
            
            tflite_model = converter.convert()
            output_path = os.path.join(self.output_dir, 'gesture_conv1d_quant.tflite')
        else:
            tflite_model = converter.convert()
            output_path = os.path.join(self.output_dir, 'gesture_conv1d.tflite')
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to {output_path}")
        print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
        
        return output_path
    
    def verify_tflite_model(self, tflite_path, num_samples=20):
        """Verify TFLite model accuracy"""
        print(f"\nVerifying TFLite model: {tflite_path}")
        
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input type: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output type: {output_details[0]['dtype']}")
        
        num_samples = min(num_samples, len(self.X_test))
        correct = 0
        indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        
        print(f"\nTesting on {num_samples} samples...")
        for idx in indices:
            input_data = self.X_test[idx:idx+1].astype(np.float32)
            true_label = np.argmax(self.y_test[idx])
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            pred_label = np.argmax(output_data[0])
            
            if pred_label == true_label:
                correct += 1
        
        accuracy = correct / num_samples
        print(f"TFLite Model Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def save_class_labels(self):
        """Save class labels mapping"""
        labels_path = os.path.join(self.output_dir, 'gesture_labels.txt')
        with open(labels_path, 'w') as f:
            for label in self.label_encoder.classes_:
                f.write(f"{label}\n")
        print(f"Class labels saved to {labels_path}")
    
    def save_metadata(self):
        """Save model metadata"""
        metadata = {
            'num_classes': self.num_classes,
            'sequence_length': self.sequence_length,
            'feature_size': self.feature_size,
            'class_names': self.label_encoder.classes_.tolist(),
            'model_type': 'Conv1D'
        }
        
        metadata_path = os.path.join(self.output_dir, 'model_metadata.txt')
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"Model metadata saved to {metadata_path}")


if __name__ == "__main__":
    dataset_path = path + "/datasets/gesture_dataset_seq_diverse.h5"
    # dataset_path = path + "/datasets/gesture_dataset_seq_enhanced.h5"
    output_dir = path + "/trained_models1"
    
    epochs = 200
    batch_size = 32
    learning_rate = 0.001
    conv_filters = [32, 64, 128]
    dense_units = [64, 32]
    dropout = 0.3
    quantize = True
    
    trainer = GestureConv1DTrainer(dataset_path, output_dir)
    trainer.load_dataset(val_split=0.15, test_split=0.15)
    trainer.build_model(conv_filters=conv_filters, dense_units=dense_units, dropout_rate=dropout)
    
    history = trainer.train(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    trainer.evaluate()
    
    # Save all metrics to CSV files
    trainer.save_training_metrics(history)
    
    # Plot from history object
    plot_path = os.path.join(output_dir, 'conv1d_training_history.png')
    trainer.plot_training_history(history, save_path=plot_path)
    
    tflite_path = trainer.convert_to_tflite(quantize=quantize)
    trainer.verify_tflite_model(tflite_path, num_samples=50)
    
    trainer.save_class_labels()
    trainer.save_metadata()