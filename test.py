import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

tfd = tfp.distributions
heart_disease = pd.read_csv("dataset/heart_statlog_cleveland_hungary_final.csv")
X = heart_disease.drop(columns=['target'])
y = heart_disease['target']
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)
print(X_test.shape)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 1), activation='relu', input_shape=(X_train.shape[1], 1, 1)),
    tf.keras.layers.MaxPooling2D((2, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


distribution_layer = tfp.layers.DistributionLambda(lambda t: tfd.Bernoulli(probs=tf.sigmoid(t)))
print(distribution_layer)
def negative_log_likelihood(y_true, y_pred):
    dist = distribution_layer(y_pred)
    return -tf.reduce_mean(dist.log_prob(y_true))


model.compile(tf.keras.optimizers.Adam(learning_rate=0.01), loss=negative_log_likelihood)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=50)
test_loss = model.evaluate(X_test, y_test)
print(test_loss)


y_pred = model.predict(X_test)
predicted_probabilities = tf.sigmoid(y_pred)
print(predicted_probabilities)

predicted_labels = tfd.Bernoulli(probs=predicted_probabilities).sample()
predicted_labels = tf.reshape(predicted_labels, [-1])
accuracy = accuracy_score(y_test, predicted_labels)
print("Accuracy:", accuracy)