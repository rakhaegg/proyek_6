import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from random_forest_model_adaptor import RandomForestModelAdaptor


def _parse_tfrecord(example, schema):
    # Dekode fitur sesuai skema
    feature_description = {
        column: tf.io.FixedLenFeature([], tf.float32) for column in schema
    }
    feature_description['Connection Point_index'] = tf.io.FixedLenFeature([], tf.int64)  # Tipe data int64

    feature_description['label'] = tf.io.FixedLenFeature([], tf.int64)
    return tf.io.parse_single_example(example, feature_description)

def _load_dataset(file_pattern, schema):
    # Membaca TFRecord
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(file_pattern), compression_type="GZIP")
    dataset = dataset.map(lambda x: _parse_tfrecord(x, schema))
    
    # Konversi dataset ke DataFrame pandas
    records = []
    labels = []
    for record in dataset:
        record_dict = {k: v.numpy() for k, v in record.items()}
        labels.append(record_dict.pop('label'))  # Pisahkan label
        records.append(record_dict)
    
    # Konversi records menjadi DataFrame pandas
    features_df = pd.DataFrame(records)
    return features_df, labels



# Fungsi utama untuk melatih model
def run_fn(fn_args):
    # Skema fitur dari transform
    schema = [
        'Received Packets', 'Sent Packets', 'Received Bytes', 'Sent Bytes',
        'Delta Received Packets', 'Delta Sent Packets',
        'Delta Received Bytes', 'Delta Sent Bytes',
        'Connection Point_index'
    ]
    train_data_path = os.path.join(fn_args.train_files[0])
    eval_data_path = os.path.join(fn_args.eval_files[0])
    
    train_x, train_y = _load_dataset(train_data_path, schema)
    eval_x, eval_y = _load_dataset(eval_data_path, schema)
    assert 'label' not in train_x.columns, "Kolom 'label' masih ada di train_x!"
    assert 'label' not in eval_x.columns, "Kolom 'label' masih ada di eval_x!"
    print("Validasi sukses: Tidak ada kolom 'label' di train_x dan eval_x.")

    # Buat dan latih model Random Forest
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    )
    model.fit(train_x, train_y)
    
    # Evaluasi model
    eval_predictions = model.predict(eval_x)
    eval_accuracy = accuracy_score(eval_y, eval_predictions)
    
    print(f"Evaluation Accuracy: {eval_accuracy:.4f}")
    
   # Pastikan lokasi direktori model dibuat
    model_output_dir = fn_args.serving_model_dir
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Simpan model Random Forest sebagai file pickle
    model_pkl_path = os.path.join(model_output_dir, 'model.pkl')
    with open(model_pkl_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model disimpan sebagai pickle di {model_pkl_path}")
    
    # Gunakan adaptor untuk menyimpan model sebagai TensorFlow SavedModel
    adaptor = RandomForestModelAdaptor(model_pkl_path)
    tf.saved_model.save(adaptor, model_output_dir)