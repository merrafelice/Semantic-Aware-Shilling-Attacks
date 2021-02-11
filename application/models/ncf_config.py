class NcfTrainingArguments(object):
    def __init__(self, user_dim, item_dim, batch_size, num_epochs, hidden1_dim,
                 hidden2_dim, hidden3_dim, hidden4_dim):
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.hidden3_dim = hidden3_dim
        self.hidden4_dim = hidden4_dim


config = {
    'training_args':
    NcfTrainingArguments(
        user_dim=16,
        item_dim=16,
        batch_size=256,
        num_epochs=150,
        hidden1_dim=64,
        hidden2_dim=32,
        hidden3_dim=16,
        hidden4_dim=8)
}
