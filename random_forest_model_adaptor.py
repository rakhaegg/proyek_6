import tensorflow as tf
import pickle
import numpy as np
class RandomForestModelAdaptor(tf.Module):
    def __init__(self, model_path):
        super().__init__()
        print(model_path)
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Perbaiki tipe data sesuai input
        self.feature_description = {
            'Received Packets': tf.io.FixedLenFeature([], tf.float32),
            'Sent Packets': tf.io.FixedLenFeature([], tf.float32),
            'Received Bytes': tf.io.FixedLenFeature([], tf.float32),
            'Sent Bytes': tf.io.FixedLenFeature([], tf.float32),
            'Delta Received Packets': tf.io.FixedLenFeature([], tf.float32),
            'Delta Sent Packets': tf.io.FixedLenFeature([], tf.float32),
            'Delta Received Bytes': tf.io.FixedLenFeature([], tf.float32),
            'Delta Sent Bytes': tf.io.FixedLenFeature([], tf.float32),
            'Connection Point_index': tf.io.FixedLenFeature([], tf.int64),  # Perbaikan tipe data
        }

    def _decode_and_preprocess(self, serialized_examples):
        # Decode serialized tf.Example
        parsed_features = tf.io.parse_example(serialized_examples, self.feature_description)
        # Konversi semua fitur ke float32 untuk digunakan oleh scikit-learn
        features = tf.stack(
            [tf.cast(parsed_features[key], tf.float32) for key in self.feature_description.keys()],
            axis=1
        )
        return features

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def predict(self, serialized_examples):
        # Decode serialized examples
        inputs = self._decode_and_preprocess(serialized_examples)
        
        # Gunakan tf.py_function untuk memanggil model scikit-learn
        def _predict(inputs_np):
            
            predictions = self.model.predict(inputs_np)
            return np.array(predictions, dtype=np.float32)
        
        predictions = tf.py_function(
            func=_predict,
            inp=[inputs],  # Tensor numerik
            Tout=tf.float32
        )
        return predictions
