import torch
import tensorflow as tf
from evolution.gan_train_surr import GanTrain
# import better_exceptions; better_exceptions.hook()
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    logger.info("CUDA device detected for torch!")
else:
    logger.info("CUDA device not detected for torch!")
if len(tf.config.list_physical_devices('GPU'))>=1:
    logger.info("CUDA device detected for TF!")
else:
    logger.info("CUDA device not detected for TF!")
if __name__ == "__main__":
    GanTrain().start()
