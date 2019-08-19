import tensorflow as tf
from tqdm import tqdm
import numpy as np


class ANFIS:
    def __init__(self, n_inputs, n_rules, batch_size, learning_rate=0.001, mf='gaussian'):
        tf.reset_default_graph()
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        self.batch_size = batch_size

        # make_initializable_iterator
        self.X = tf.placeholder(tf.float64, shape=(None, n_inputs), name='inputs')
        self.y = tf.placeholder(tf.float64, shape=None, name='target')
        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.y)).batch(batch_size)
        self.iterator = dataset.make_initializable_iterator()
        self.inputs, self.target = self.iterator.get_next()

        if mf == 'gaussian':
            mu = tf.get_variable(name="mu", shape=[n_rules * n_inputs],
                                 initializer=tf.random_normal_initializer(0, 1), dtype=tf.float64)
            sigma = tf.get_variable(name="sigma", shape=[n_rules * n_inputs],
                                    initializer=tf.random_normal_initializer(0, 1), dtype=tf.float64)

            rule = tf.reshape(
                tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(self.inputs, (1, n_rules)), mu)) / tf.square(sigma)),
                (-1, n_rules, n_inputs))
        elif mf == 'bell':
            a = tf.get_variable(name='a', shape=[n_rules * n_inputs],
                                initializer=tf.random_normal_initializer(0, 1),
                                dtype=tf.float64)
            b = tf.get_variable(name='b', shape=[n_rules * n_inputs],
                                initializer=tf.random_normal_initializer(0, 1),
                                dtype=tf.float64)
            c = tf.get_variable(name='c', shape=[n_rules * n_inputs],
                                initializer=tf.random_normal_initializer(0, 1),
                                dtype=tf.float64)
            rule = tf.reshape(1 / (1 + tf.square((tf.tile(self.inputs, (1, n_rules)) - c) / a) ** tf.square(b)),
                              (-1, n_rules, n_inputs))

        for i in range(n_inputs - 1):
            if i == 0:
                rule_1 = tf.reshape(rule[:, :, i], shape=(-1, n_rules, 1))
            else:
                rule_1 = tf.identity(firing_strength)

            rule_2 = tf.reshape(rule[:, :, i + 1], shape=(-1, 1, n_rules))
            firing_strength = tf.reshape(tf.matmul(rule_1, rule_2), shape=(-1, n_rules ** (i + 2), 1),
                                         name='firing_strength')

        firing_strength = tf.reshape(firing_strength, shape=[-1, n_rules ** n_inputs],
                                     name='firing_strength')
        weights_sum = tf.reshape(tf.reduce_sum(firing_strength, 1), shape=[-1, 1])
        normalized_firing_strength = tf.divide(firing_strength, weights_sum, name='layer_3')
        layer_4 = tf.layers.dense(inputs=self.inputs, units=n_rules ** n_inputs, name='layer_4')

        self.prediction = tf.reduce_sum(tf.multiply(normalized_firing_strength, layer_4), 1, name='prediction')

        # loss function and optimizer
        self.loss = tf.sqrt(tf.losses.mean_squared_error(self.target, self.prediction))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train = self.optimizer.minimize(self.loss)
        self.saver = tf.train.Saver()
        self.trainable_variables = tf.trainable_variables

    def fit(self, X, y, epochs=1, validation_split=0.0, patience=0):
        init_op = tf.global_variables_initializer()

        if validation_split > 0.0:
            split_point = int(len(y) * (1 - validation_split))
            X_val, y_val = X[split_point:], y[split_point:]
            X, y = X[:split_point], y[:split_point]
            print("Train_size: {}, validation_size: {}".format(len(y), len(y_val)))
        else:
            pass

        with tf.Session() as sess:

            # merged = tf.summary.merge_all()
            # writer = tf.summary.FileWriter("ANFIS_logs", sess.graph)
            sess.run(init_op)

            train_loss_history = []
            val_loss_history = []
            # run train data
            for e in range(epochs):
                # Initialize iterator with train data
                sess.run(self.iterator.initializer,
                         feed_dict={self.X: X.reshape(-1, self.n_inputs), self.y: y.flatten()})
                train_loss = 0
                val_loss = 0
                try:
                    with tqdm(total=len(y)) as pbar:
                        while True:
                            _, loss_value = sess.run([self.train, self.loss])
                            train_loss += loss_value
                            pbar.update(self.batch_size)
                except tf.errors.OutOfRangeError:
                    pass

                # Initialize iterator with validation data
                sess.run(self.iterator.initializer,
                         feed_dict={self.X: X_val.reshape(-1, self.n_inputs), self.y: y_val.flatten()})
                try:
                    while True:
                        loss = sess.run(self.loss)
                        val_loss += loss
                except tf.errors.OutOfRangeError:
                    pass
                print('epoch: {}/{}, train_loss:{}, val_loss: {}'.format(e, epochs, train_loss / len(y),
                                                                         val_loss / len(y_val)))
                train_loss_history.append(train_loss / len(y))
                val_loss_history.append(val_loss / len(y_val))

                if patience > 0:
                    if e == 0:
                        patience_count = 0
                        best_val_loss = val_loss / len(y_val)
                    else:
                        new_val_loss = val_loss / len(y_val)
                        if best_val_loss > new_val_loss:
                            patience_count = 0
                            print('val_loss improved from {} to {}, patience_count: {}/{}'.format(best_val_loss,
                                                                                                  new_val_loss,
                                                                                                  patience_count,
                                                                                                  patience))
                            save_path = self.saver.save(sess, "ANFIS/trained/model.ckpt")
                            print("Model saved in path: %s" % save_path)
                        else:
                            patience_count += 1
                            print('val_loss did not improve from {}, patience_count: {}/{}'.format(best_val_loss,
                                                                                                   patience_count,
                                                                                                   patience))

                        best_val_loss = min(best_val_loss, new_val_loss)
                    if patience_count >= patience:
                        break

            history = np.column_stack((train_loss_history, val_loss_history))
            return history

    def predict(self, X, y, pred_steps=5):
        with tf.Session() as sess:

            X = X.reshape(-1, self.n_inputs)
            y = y.flatten()
            self.saver.restore(sess, "ANFIS/trained/model.ckpt")
            # run predict

            all_preds = []
            for step in range(pred_steps):

                print('predicting step:{}'.format(step + 1))
                if step > 0:
                    X = np.concatenate([X[:, 1:], one_step_preds.reshape(-1, 1)], axis=1)
                else:
                    pass

                one_step_preds = []
                sess.run(self.iterator.initializer,
                         feed_dict={self.X: X, self.y: y})

                try:
                    with tqdm(total=len(y)) as pbar:
                        while True:
                            pred = sess.run(self.prediction)
                            one_step_preds.append(pred)
                            pbar.update(self.batch_size)
                except tf.errors.OutOfRangeError:
                    pass
                # final batch is not the same size
                one_step_preds = np.concatenate([np.array(one_step_preds[:-1]).flatten(), one_step_preds[-1]])
                all_preds.append(one_step_preds)
            all_preds = np.array(all_preds)
        return all_preds
