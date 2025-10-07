import tensorflow as tf
import numpy as np

class RBM(tf.keras.Model):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = tf.Variable(tf.random.normal([n_visible, n_hidden], stddev=0.01))
        self.bh = tf.Variable(tf.zeros([n_hidden]))
        self.bv = tf.Variable(tf.zeros([n_visible]))
   
    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

    def call(self, v):
       
        h_prob = tf.nn.sigmoid(tf.matmul(v, self.W) + self.bh)
        h_sample = self.sample_prob(h_prob)
        v_prob = tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(self.W)) + self.bv)
        return v_prob, h_prob
   
    def train_step(self, v0, learning_rate=0.01):
        h0_prob = tf.nn.sigmoid(tf.matmul(v0, self.W) + self.bh)
        h0_sample = self.sample_prob(h0_prob)
        v1_prob = tf.nn.sigmoid(tf.matmul(h0_sample, tf.transpose(self.W)) + self.bv)
        v1_sample = self.sample_prob(v1_prob)
        h1_prob = tf.nn.sigmoid(tf.matmul(v1_sample, self.W) + self.bh)
       
        positive_grad = tf.matmul(tf.transpose(v0), h0_prob)
        negative_grad = tf.matmul(tf.transpose(v1_sample), h1_prob)
       
        self.W.assign_add(learning_rate * (positive_grad - negative_grad) / tf.cast(tf.shape(v0)[0], tf.float32))
        self.bv.assign_add(learning_rate * tf.reduce_mean(v0 - v1_sample, axis=0))
        self.bh.assign_add(learning_rate * tf.reduce_mean(h0_prob - h1_prob, axis=0))


num_users = 10
num_movies = 6
np.random.seed(0)
data = np.random.randint(2, size=(num_users, num_movies)).astype(np.float32)

rbm = RBM(n_visible=num_movies, n_hidden=3)

epochs = 500
for epoch in range(epochs):
    rbm.train_step(data)
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1} done.")

user_0 = data[0:1]
v_prob, _ = rbm.call(user_0)
print("User 0 original ratings:", user_0)
print("Predicted ratings probabilities:", v_prob.numpy())