import numpy as np
import tensorflow as tf

tag_maps = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

def masked_loss(y_true, y_pred):
  # Mendefinisikan fungsi kerugian dengan SparseCategoricalCrossentropy
  # Parameter from_logits=True menunjukkan pada loss function bahwa nilai output yang dihasilkan oleh model tidak dinormalisasi, alias logit.
  # Dengan kata lain, fungsi softmax belum diterapkan pada mereka untuk menghasilkan distribusi probabilitas.
  # Oleh karena itu, lapisan output dalam kasus ini tidak memiliki fungsi aktivasi softmax:
  # ignore_class=-1 mengabaikan kelas dengan nilai -1 saat menghitung kerugian
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=-1)

  # Menghitung kerugian menggunakan fungsi kerugian yang telah didefinisikan
  loss = loss_fn(y_true, y_pred)

  return loss

def masked_accuracy(y_true, y_pred):
  # Mengonversi tensor label sebenarnya ke tipe data float32
  y_true = tf.cast(y_true, tf.float32)

  # Membuat masker untuk nilai yang akan diabaikan yaitu -1 dan mengonversi ke tipe data float32
  mask = tf.not_equal(y_true, -1)
  mask = tf.cast(mask, tf.float32)

  # Menerapkan argmax untuk mendapatkan nilai prediksi, dan mengonversi ke tipe data float32
  y_pred_class = tf.math.argmax(y_pred, axis=-1)
  y_pred_class = tf.cast(y_pred_class, tf.float32)

  # Membandingkan label sebenarnya dengan kelas prediksi, dan mengonversi ke tipe data float32
  matches_true_pred  = tf.equal(y_true, y_pred_class)
  matches_true_pred = tf.cast(matches_true_pred, tf.float32)

  # Mengalikan hasil perbandingan dengan masker, sehingga hanya menghitung akurasi untuk label yang tidak di-mask
  matches_true_pred *= mask

  # Menghitung akurasi dengan jumlah prediksi yang benar dibagi dengan jumlah label yang tidak di-mask
  masked_acc = tf.reduce_sum(matches_true_pred) / tf.reduce_sum(mask)

  return masked_acc

def predict(sentence, model, sentence_vectorizer, tag_map=tag_maps):
    # Mengonversi kalimat menjadi vektor fitur dengan menggunakan sentence_vectorizer
    sentence_vectorized = sentence_vectorizer(sentence)
    # Menambahkan dimensi tambahan pada vektor fitur untuk sesuai untuk di fed ke model
    sentence_vectorized = tf.expand_dims(sentence_vectorized, 0)
    # Melakukan prediksi menggunakan model NER
    outputs = model(sentence_vectorized)
    # Dapatkan label yang diprediksi untuk setiap token, menggunakan fungsi argmax dan menentukan axis yang benar untuk melakukan argmax
    output = np.argmax(outputs, axis=-1)
    # Baris berikutnya hanya untuk menyesuaikan dimensi keluaran. Karena fungsi ini hanya mengharapkan satu masukan untuk mendapatkan prediksi, keluarannya akan menjadi seperti [[1,2,3]]
    # jadi untuk menghindari notasi yang berat di bawah ini, mari kita ubah menjadi [1,2,3]
    output = output[0]
    # Mendapatkan daftar label yang sesuai dengan tag_map
    labels = list(tag_map.keys())
    pred = []
    # Iterasi melalui setiap indeks label hasil prediksi
    for tag_idx in output:
        pred_label = labels[tag_idx]
        pred.append(pred_label)

    return pred