""" Create a model that learns to multiply by seven.

    Based on https://www.kaggle.com/jruizvar/toy-model-of-a-neural-network-in-tensorflow
"""
import numpy as np
import tensorflow as tf

COUNT_NEURONS = 1
COUNT_EPOCHS = 1000
EARLY_STOP = 1.e-5
LEARNING_RATE = 0.1

x_train = np.array([[1.]])
y_train = np.array([[7.]])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(COUNT_NEURONS, use_bias=False, input_shape=[1]))

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

for i in range(COUNT_EPOCHS):
    with tf.GradientTape() as model_tape:
        y_pred = model(x_train)        
        model_loss = tf.losses.mean_squared_error(y_train, y_pred)

    gradients_of_model = model_tape.gradient(model_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))

    if i % 10 == 0:
            print("Epoch:", i)
            print("Loss value:", model_loss)
            print("Current predicted value:", y_pred)
            print()
    if model_loss < EARLY_STOP:
        break


""" Test trained model
"""
x_test = np.array([[2.],
                    [3.],
                    [4.],
                    [5.],
                    [6.],
                    [7.],
                    [8.],
                    [9.]])

y_test = model(x_test)

print("Predictions on test set:")
print(np.c_[x_test, y_test])

# Optionnal: Save the model on disk to serve with Tensorflow Serving
#tf.keras.experimental.export_saved_model(model, 'multiply_seven/saved_model', serving_only=True)