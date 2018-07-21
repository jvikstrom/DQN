
def trainable_weigth(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, dtype=tf.float32)

class DQN(object):

    def __init__(self, session, learning_rate, action_space, state_space):
        self.session       = session
        self.learning_rate = learning_rate
        self.action_space  = action_space
        self.state_space   = state_space

    def create_network(self):

        return None

    def create_training(self, out):

        return None

    def predict(self, state):
        pass

    def Q_value(self, state):
        pass

    def train(self, state, action, label):
        pass

