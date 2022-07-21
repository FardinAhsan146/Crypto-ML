import numpy as np
from data_loader import EllipticDatasetLoader

# import tensorflow as tf
# from models import EvolveGCN
# from models.layers import EGCUH, HGRUCell, SummarizeLayer, GCNLayer

"""
Gonna test baseline E-GCN as per the github repository here

"""

# Constants
DATADIR = "../../data_raw"
FILTER_UNKNOWN = False
ONLY_LOCAL_FEATURE = False
LEARNING_RATE = 1e-3
TEST_SHARE = 0.3

# optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)
# loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits = True)

# Load data
dl = EllipticDatasetLoader(DATADIR, TEST_SHARE, FILTER_UNKNOWN,local_features_only=ONLY_LOCAL_FEATURE)

# Load model
# model = EvolveGCN([
#     EGCUH(HGRUCell(64),SummarizeLayer(),activation="relu"),
#     EGCUH(HGRUCell(dl.num_classes),SummarizeLayer())
# ])

# model.compile(optimizer = optimizer, loss = loss_func)

# model.load_weights('weights/evolve_default_train.h5')

# print(model.summary())

print(dl)