import numpy as np
import tensorflow as tf

from data_loader import EllipticDatasetLoader
from models import EvolveGCN
from models.layers import EGCUH, HGRUCell, SummarizeLayer, GCNLayer



DATADIR = "../../data_raw"
FILTER_UNKNOWN = False
ONLY_LOCAL_FEATURE = False
CLASS_WEIGTHS = [0.7,0.29,0.01]
NUM_ROLLS = 4
TEST_SHARE = 0.3
NUM_EPOCH = 500
LEARNING_RATE = 1e-3


def reset_metrics(list_of_metrics):
    for m in list_of_metrics:
        m.reset_states()


dl = EllipticDatasetLoader(DATADIR, TEST_SHARE, FILTER_UNKNOWN,local_features_only=ONLY_LOCAL_FEATURE)

model = EvolveGCN([
    EGCUH(HGRUCell(64),SummarizeLayer(),activation="relu"),
    EGCUH(HGRUCell(dl.num_classes),SummarizeLayer())
])

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def run_model(adj,nodes,targets,training=False):
    weigths = tf.reduce_sum(CLASS_WEIGTHS * targets, axis=-1)
    states = model.get_initial_weigths(tf.shape(nodes))
    logits = []
    for i in range(NUM_ROLLS):
        l, states = model([adj, nodes, states], training=training)
        logits.append(l)

    loss = sum(loss_func(targets, l, sample_weight=weigths) for l in logits)
    return logits[-1], loss, weigths


train_loss_metric = tf.keras.metrics.Mean()
train_accuracy_metric = tf.keras.metrics.Accuracy()
train_precision_metric = tf.keras.metrics.Precision()
train_recall_metric = tf.keras.metrics.Recall()
test_loss_metric = tf.keras.metrics.Mean()
test_accuracy_metric = tf.keras.metrics.Accuracy()
test_precision_metric = tf.keras.metrics.Precision()
test_recall_metric = tf.keras.metrics.Recall()
metrics = [train_loss_metric,train_accuracy_metric,train_precision_metric,train_recall_metric
            ,test_loss_metric,test_accuracy_metric,test_precision_metric,test_recall_metric]

for epoch in range(NUM_EPOCH):
    reset_metrics(metrics)

    for _, n, t, adj in dl.train_batch_iterator():
        with tf.GradientTape() as tape:
            logits, loss, weigths = run_model(adj, n, t, training=True)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads,model.trainable_weights))

        y_true = tf.cast(tf.argmax(t,axis=-1) == 0,tf.float32)
        y_pred = tf.cast(tf.argmax(logits,axis=-1) == 0,tf.float32)
        train_loss_metric(loss)
        train_accuracy_metric(tf.argmax(t,axis=-1), tf.argmax(logits,axis=-1), sample_weight=weigths)
        train_precision_metric(y_true, y_pred)
        train_recall_metric(y_true, y_pred)

    for _, n, t, adj in dl.test_batch_iterator():
        logits, loss, weigths = run_model(adj, n, t)

        y_true = tf.cast(tf.argmax(t, axis=-1) == 0, tf.float32)
        y_pred = tf.cast(tf.argmax(logits, axis=-1) == 0, tf.float32)
        test_loss_metric(loss)
        test_accuracy_metric(tf.argmax(t,axis=-1), tf.argmax(logits,axis=-1), sample_weight=weigths)
        test_precision_metric(y_true, y_pred)
        test_recall_metric(y_true, y_pred)

    print("Epoch: {}\nTRAIN Loss: {:.5}| Accuracy: {:.4}| Precision: {:.4}| Recall: {:.4}\nTEST Loss: {:.4}| Accuracy: {:.4}| Precision: {:.4}| Recall: {:.4}".format(
        epoch, train_loss_metric.result().numpy(), train_accuracy_metric.result().numpy(),
        train_precision_metric.result().numpy(),train_recall_metric.result().numpy(),
        test_loss_metric.result().numpy(), test_accuracy_metric.result().numpy(),
        test_precision_metric.result().numpy(),test_recall_metric.result().numpy()
    ))

#Save the model, change file name as per modifications
try:
    model.save('weights/evolve_default_train.h5', save_formate = 'tf')
except Exception:
    model.save_weights('weights/evolve_default_train.h5')
finally:
    print(model.summary())