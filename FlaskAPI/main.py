
import tensorflow as tf
loaded_model = tf.keras.models.load_model('FinishedAIMODEL.keras')
loaded_model.save("FinishedModel.h5")