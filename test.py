"""
DUNE CVN test module.
"""
__version__ = '1.0'
__author__ = 'Saul Alonso-Monsalve, Leigh Howard Whitehead'
__email__ = "saul.alonso.monsalve@cern.ch, leigh.howard.whitehead@cern.ch"

import shutil
import numpy as np
import pickle as pk
import sys
import os

sys.path.append(os.path.join(sys.path[0], 'modules'))

from tensorflow.keras.models import model_from_json
from sklearn.metrics import classification_report, confusion_matrix
from data_generator import DataGenerator
from opts import get_args
from keras.models import load_model
import my_losses
from dune_cvn import CustomTrainStep

# manually specify the GPUs to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

args = get_args()

#  def test():
''' Test the DUNE CVN on input dataset.
'''
# parameters
test_values = []
TEST_PARAMS = {'batch_size':args.batch_size,
               'images_path':args.dataset,
               'shuffle':args.shuffle,
               'test_values':test_values}
# load dataset
print('Reading dataset from serialized file...')
with open('dataset/partition.p', 'rb') as partition_file:
    IDs, labels = pk.load(partition_file)
print('Loaded. Number of test examples: %d', len(IDs))

# generator
prediction_generator = DataGenerator(**TEST_PARAMS).generate(labels, IDs)

# load model
print('Loading model from disk...')

if args.isvd:
    model = load_model('saved_model/model_vd.h5',
                       custom_objects={'masked_loss':my_losses.masked_loss,
                                       'multitask_loss': my_losses.multitask_loss,
                                       'masked_loss_binary': my_losses.masked_loss_binary,
                                       'masked_loss_categorical': my_losses.masked_loss_categorical,
                                       'CustomTrainStep': CustomTrainStep})

else:
    with open('saved_model/model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('saved_model/weights.h5')

if(args.print_model):
    model.summary()

# labels
is_antinu_labels = ['nu', 'antinu']
flav_labels = ['CC Numu', 'CC Nue', 'CC Nutau', 'NC']
inte_labels = ['CC QE', 'CC Res', 'CC DIS', 'CC other']
prot_labels = ['0 protons', '1 protons', '2 protons', '>2 protons']
chpi_labels = ['0 ch. pions', '1 ch. pions', '2 ch. pions', '>2 ch. pions']
nepi_labels = ['0 neu. pions', '1 neu. pions', '2 neu. pions', '>2 neu. pions']
neut_labels = ['0 neutrons', '1 neutrons', '2 neutrons', '>2 neutrons']

# prediction
print('Performing test...')
Y_pred = model.predict_generator(generator = prediction_generator,
                       steps = len(IDs)//args.batch_size,
                       verbose = 1)

# predicted values
y_pred_is_antinu = np.around(Y_pred[0]).reshape((Y_pred[0].shape[0], 1)).astype(int)
y_pred_flav = np.argmax(Y_pred[1], axis=1).reshape((Y_pred[1].shape[0], 1))
y_pred_inte = np.argmax(Y_pred[2], axis=1).reshape((Y_pred[2].shape[0], 1))
y_pred_prot = np.argmax(Y_pred[3], axis=1).reshape((Y_pred[3].shape[0], 1))
y_pred_chpi = np.argmax(Y_pred[4], axis=1).reshape((Y_pred[4].shape[0], 1))
y_pred_nepi = np.argmax(Y_pred[5], axis=1).reshape((Y_pred[5].shape[0], 1))
y_pred_neut = np.argmax(Y_pred[6], axis=1).reshape((Y_pred[6].shape[0], 1))

# actual values
test_values = np.array(test_values[0:Y_pred[0].shape[0]]) # array with y true values
y_test_is_antinu = np.array([aux[0] for aux in test_values]).reshape(y_pred_is_antinu.shape)
y_test_flav = np.array([aux[1] for aux in test_values]).reshape(y_pred_flav.shape)
y_test_inte = np.array([aux[2] for aux in test_values]).reshape(y_pred_inte.shape)
y_test_prot = np.array([aux[3] for aux in test_values]).reshape(y_pred_prot.shape)
y_test_chpi = np.array([aux[4] for aux in test_values]).reshape(y_pred_chpi.shape)
y_test_nepi = np.array([aux[5] for aux in test_values]).reshape(y_pred_nepi.shape)
y_test_neut = np.array([aux[6] for aux in test_values]).reshape(y_pred_neut.shape)

for i in range(y_pred_flav.shape[0]):
    # NC exception
    if y_pred_flav[i] == 3:
        y_pred_is_antinu[i] = 2
        y_pred_inte[i] = 4
    if y_test_flav[i] == 3:
        y_test_is_antinu[i] = 2
        y_test_inte[i] = 4
