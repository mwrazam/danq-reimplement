import os, sys, time
import json
import numpy as np
import h5py
import scipy.io
import tensorflow.keras.layers as L
import tensorflow.keras.metrics as M
import tensorflow as tf
from danq_model import DanQ

# Utility function for simulating multiple GPU availability to Tensorflow, only used for testing purposes
# Execute this function BEFORE running any distributed strategy
def simulate_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Create 2 virtual GPUs with 1GB memory each
        # This is for simulating multiple GPUs for testing purposes only
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

# Load data from files, use fraction to load only a subset of the data during testing
# training_mode=True loads all data, training_mode=False only returns test data
def load_deepsea_data(paths, fraction=1.0, training_mode=True):
    start = time.time()
    print("Loading data...")
    if training_mode:
        # Load training data
        try:
            train_data = h5py.File(os.path.join(paths['DATA_DIR'], paths['TRAINING_DATA_FILE']), 'r')
            train_x = np.transpose(np.take(train_data['trainxdata'], range(0, int(train_data['trainxdata'].shape[2] * fraction)), axis=2), axes=(2,0,1))
            train_y = np.take(train_data['traindata'], range(0, int(train_data['traindata'].shape[1] * fraction)), axis=1).T
            assert train_x.shape[0] == train_y.shape[0], "Input and output training data must have the same number of samples"
            print(f"Training data loaded sucessfully with shape x:{train_x.shape}, y:{train_y.shape}")
        except:
            sys.exit(f"Could not load training data file {os.path.join(paths['DATA_DIR'], paths['TRAINING_DATA_FILE'])}")
        
        # Load validation data
        try:
            valid_data = scipy.io.loadmat(os.path.join(paths['DATA_DIR'], paths['VALIDATION_DATA_FILE']))
            valid_x = np.transpose(np.take(valid_data['validxdata'], range(0, int(len(valid_data['validxdata']) * fraction)), axis=0), axes=(0,2,1))
            valid_y = np.take(valid_data['validdata'], range(0, int(len(valid_data['validdata']) * fraction)), axis=0)
            assert valid_x.shape[0] == valid_y.shape[0], "Input and output validation data must have the same number of samples"
            print(f"Validation data loaded succesfully with shape x:{valid_x.shape}, y:{valid_y.shape}")
        except:
            sys.exit(f"Could not load validation data file {os.path.join(paths['DATA_DIR'], paths['VALIDATION_DATA_FILE'])}")

    # Load test data set
    try:
        test_data = scipy.io.loadmat(os.path.join(paths['DATA_DIR'], paths['TEST_DATA_FILE']))
        test_x = np.transpose(np.take(test_data['testxdata'], range(0, int(len(test_data['testxdata']) * fraction)), axis=0), axes=(0,2,1))
        test_y = np.take(test_data['testdata'], range(0, int(len(test_data['testdata']) * fraction)), axis=0)
        assert test_x.shape[0] == test_y.shape[0], "Input and output test data must have the same number of samples"
        print(f"Test data loaded succesfully with shape x:{test_x.shape}, y:{test_y.shape}")
    except:
         sys.exit(f"Could not load test data file {os.path.join(paths['DATA_DIR'], paths['TEST_DATA_FILE'])}")

    print(f"Done loading data. Took {int(time.time() - start)}s.")

    if not training_mode:
        train_x, train_y, valid_x, valid_y = None, None, None, None

    return train_x, train_y, valid_x, valid_y, test_x, test_y

# Load configuration file
def load_config(filename):
    try:
        with open(filename) as config_file:
            return json.loads(config_file.read())
    except:
        sys.exit(f"Could not load config file {filename}")

if __name__ == "__main__":
    
    # Check whether to run in DEBUG mode
    # DEBUG mode will load only a subset of the data use mult-GPU simulation for distributed environments
    # TODO: Implement click for better CLI, and also allow for custom configuration files containing value ranges, and option to load custom fraction of the data
    if len(sys.argv) == 2 and sys.argv[1] == "DEBUG":
        DEBUG_MODE = True
        DATA_PORTION = 0.001
    elif len(sys.argv) == 1:
        DEBUG_MODE = False
        DATA_PORTION = 1.0
    else:
        sys.exit("Only one argument 'DEBUG' is accepted for this program")

    # get configuration
    config = load_config("config.json")

    # get data
    train_data_x, train_data_y, valid_data_x, valid_data_y, test_data_x, test_data_y = load_deepsea_data(config['PATHS'], fraction=DATA_PORTION, training_mode=True)

    output_model_file = os.path.join(config['PATHS']['MODEL_DIR'], config['PATHS']['BEST_MODEL_FILE'])
    danq_net = DanQ(config['MODEL'], config['OUTPUT'], auto_build=True)    
    danq_net.train(train_data_x, train_data_y, valid_data_x, valid_data_y, output_model_file, distributed=True, save_at_checkpoints=True, early_stopping=True, use_existing=False)
    danq_net.test(test_data_x, test_data_y)