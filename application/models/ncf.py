import tensorflow as tf

loss_tracker = tf.keras.metrics.Mean(name="loss")
mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")


# import concurrent.futures
# Default Parameters of NeuMF in X.He Paper
# python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1

# with concurrent.futures.ThreadPoolExecutor() as executor:
#     output = executor.map(self.calculate_beta_alpha, all_pos_u)

class NeuralCF(tf.keras.Model):
    """
    This class implements matrix factorization using the tf2 api
    """

    def __init__(self, n_users, n_items, user_dim, item_dim, hidden1_dim,
                 hidden2_dim,
                 hidden3_dim,
                 hidden4_dim):
        super(NeuralCF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.hidden3_dim = hidden3_dim
        self.hidden4_dim = hidden4_dim

        self.user_embeddings = tf.keras.layers.Embedding(self.n_users, self.user_dim, embeddings_initializer='normal',
                                                         input_length=1)
        self.item_embeddings = tf.keras.layers.Embedding(self.n_items, self.item_dim, embeddings_initializer='normal',
                                                         input_length=1)

        self.concat = tf.keras.layers.Concatenate()

        self.dense1 = tf.keras.layers.Dense(self.hidden1_dim, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.1)

        self.dense2 = tf.keras.layers.Dense(self.hidden2_dim, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.1)

        self.dense3 = tf.keras.layers.Dense(self.hidden3_dim, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.1)

        self.dense4 = tf.keras.layers.Dense(self.hidden4_dim, activation='relu')

        self.predicted = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, training=True):
        user_input, item_input = inputs

        input_user_vector = self.user_embeddings(user_input)
        input_item_vector = self.item_embeddings(item_input)

        x = self.concat([input_item_vector, input_user_vector])
        x = self.dense1(x)
        if training:
            x = self.dropout1(x)
        x = self.dense2(x)
        if training:
            x = self.dropout2(x)
        x = self.dense3(x)
        if training:
            x = self.dropout3(x)
        x = self.dense4(x)
        x = self.predicted(x)
        return x

    def predict_all(self):
        U = self.user_embeddings.weights[0].numpy()
        I = self.item_embeddings.weights[0].numpy()
        U_ = tf.reshape(tf.tile(U, [1, I.shape[0]]), (U.shape[0] * I.shape[0], U.shape[1]))
        I_ = tf.tile(I, [U.shape[0], 1])
        UI = tf.concat([U_, I_], 1)
        x = self.dense1(tf.Variable(UI))
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        full_predictions = self.predicted(x)
        return full_predictions
