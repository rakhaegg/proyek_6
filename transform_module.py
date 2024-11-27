import tensorflow as tf
import tensorflow_transform as tft

# Daftar fitur numerik dan kategori
NUMERIC_FEATURES = [
    'Received Packets', 'Sent Packets', 
    'Received Bytes', 'Sent Bytes', 
    'Delta Received Packets', 'Delta Sent Packets', 
    'Delta Received Bytes', 'Delta Sent Bytes'
]
CATEGORICAL_FEATURES = [
    'Connection Point'  # Contoh fitur kategori
]

# Fungsi untuk mengganti nilai yang hilang secara manual
def handle_missing_value(tensor, default_value):
    # Konversi ke tipe float jika diperlukan untuk mendukung tf.math.is_nan
    tensor = tf.cast(tensor, tf.float32)
    filled_tensor = tf.where(
        tf.math.is_nan(tensor),
        tf.constant(default_value, dtype=tf.float32),
        tensor
    )
    return tf.cast(filled_tensor, tensor.dtype)  # Kembalikan ke tipe asli (int jika diperlukan)

# Fungsi preprocessing untuk Random Forest
def preprocessing_fn(inputs):
    outputs = {}

    # 1. Mengolah fitur numerik
    for feature in NUMERIC_FEATURES:
        # Isi nilai yang hilang dengan 0
        outputs[feature] = handle_missing_value(inputs[feature], default_value=0)
    
    # 2. Mengolah fitur kategori
    for feature in CATEGORICAL_FEATURES:
        # Konversi ke string jika tipe data bukan string
        categorical_feature = tf.strings.as_string(inputs[feature])
        outputs[feature + '_index'] = tft.compute_and_apply_vocabulary(
            categorical_feature
        )

    # 3. Menangani label
    # Anggap label adalah fitur biner "Binary Label" (Attack = 1, Normal = 0)
    outputs['label'] = tf.cast(tf.equal(inputs['Binary Label'], 'Attack'), tf.int64)

    return outputs
