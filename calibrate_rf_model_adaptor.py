import tensorflow as tf
import numpy as np
import pickle

class CalibratedModelAdaptor(tf.Module):
    def __init__(self, model_path):
        super().__init__()
        # Muat model scikit-learn terkalibrasi
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Definisikan deskripsi fitur
        self.feature_description = {
            'Connection Point_index': tf.io.FixedLenFeature([], tf.int64),  # Awalnya int64
            'Received Bytes': tf.io.FixedLenFeature([], tf.float32),
            'Received Packets': tf.io.FixedLenFeature([], tf.float32),
            'Delta Received Bytes': tf.io.FixedLenFeature([], tf.float32),
            'Sent Bytes': tf.io.FixedLenFeature([], tf.float32),
            'Delta Sent Bytes': tf.io.FixedLenFeature([], tf.float32),
            'Sent Packets': tf.io.FixedLenFeature([], tf.float32),
            'Delta Received Packets': tf.io.FixedLenFeature([], tf.float32),
            'Delta Sent Packets': tf.io.FixedLenFeature([], tf.float32),
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def predict(self, serialized_inputs):
        def _predict(inputs_np):
            # Prediksi probabilitas dengan model scikit-learn
            predictions = self.model.predict_proba(inputs_np)[:, 1]
            return np.array(predictions, dtype=np.float32)

        # Deserialisasi serialized_inputs
        parsed_features = tf.io.parse_example(serialized_inputs, self.feature_description)

        # Konversi semua fitur menjadi float32
        float_features = {
            key: tf.cast(value, tf.float32)
            for key, value in parsed_features.items()
        }

        # Gabungkan fitur menjadi satu tensor
        inputs = tf.stack(list(float_features.values()), axis=1)

        # Gunakan tf.py_function untuk prediksi dengan scikit-learn
        probabilities = tf.py_function(func=_predict, inp=[inputs], Tout=tf.float32)
        return probabilities
