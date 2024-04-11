# imports
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Input, Dot, Concatenate
from keras.callbacks import TensorBoard
from time import strftime

# numpy rng -- random number generator
rng = np.random.default_rng(747)

# Constants

PATH_MOVIES = 'dataset/ml-1m/movies.dat'
PATH_RATINGS = 'dataset/ml-1m/ratings.dat'
PATH_USERS = 'dataset/ml-1m/users.dat'

LOG_DIR = 'tensorboard_ncf_logs/'

# due to computational limitations, I had to make some changes to the training data, so you may wanna change these if you want
MIN_INTERACTIONS = 700
ITEM_BOUND = 3500
TEST_BOUND = 10
IDX_OFFSET = 1
INTERACTION_SIZE_TEST = 1
ALPHA = 0.5
NUM_USERS = 20
BATCH_SIZE = 1000
EPOCHS = 50

# file imports

# movies
df_movies = pd.read_csv(PATH_MOVIES, delimiter='::', header=None, encoding='latin-1', engine='python', 
                        names=['MOVIE_ID', 'NAME', 'GENRES'])
df_movies.set_index('MOVIE_ID', inplace=True)

# users
df_users = pd.read_csv(PATH_USERS, delimiter='::', header=None, encoding='latin-1', engine='python', 
                        names=['USER_ID', 'GENDER', 'AGE', 'OCCUPATION', 'ZIP'])
df_users.set_index('USER_ID', inplace=True)

# ratings
df_ratings = pd.read_csv(PATH_RATINGS, delimiter='::', header=None, encoding='latin-1', engine='python', 
                        names=['USER_ID', 'MOVIE_ID', 'RATINGS', 'TIMESTAMP'])


# dropping timestamp as it's not needed

sparse_df = df_ratings.drop('TIMESTAMP',axis=1)

# as the original paper, studies implicit feedback between users and items, the ratings must be converted to 1 denoting the
# interaction between users and items

sparse_df.loc[:, 'RATINGS'] = 1

# now, the original Movielens dataset has some errors which we need to fix, eg. sometimes interacted movie_id go beyond the
# actual len of the movie dataset 

f_index = sparse_df.loc[sparse_df.MOVIE_ID > df_movies.shape[0], 'USER_ID'].index
sparse_df.drop(f_index, inplace=True)

# the original paper uses 1:100 (interacted : non interacted) ratio for testing data for each user, i've used 1:10
# again due to computational limitations, again you may wanna tweek ITEM BOUND and TEST BOUND constants if you want.

# this drops all the duplicated user ids, leaving 1 interacted item for each user which we'll use for testing.

test_sparse_df = sparse_df.drop_duplicates('USER_ID') 
sparse_df.drop(test_sparse_df.index, axis=0, inplace=True)
sparse_df.reset_index(inplace=True)
sparse_df.drop('index', axis=1, inplace=True)

test_sparse_df.set_index('USER_ID', inplace=True) 

# user and item identity/unit matrices which we'll use for one-hot encoding

identity_item = np.identity(df_movies.shape[0])
identity_user = np.identity(df_users.shape[0])

# Pre-Processing

# for trainning data, the original paper uses 1:4 [interacted items : non interacted items] for each user, 
# so as we are using about 700 interacted items, therefore there there should 2800 non interacted items, 
# which means in total 3500 items for each user

num_interaction_all = sparse_df.USER_ID.value_counts()
req_num_interaction = num_interaction_all.loc[num_interaction_all>=MIN_INTERACTIONS].sort_values()[:NUM_USERS]

req_user_ids = req_num_interaction.index.to_numpy()
req_num_interaction = req_num_interaction.to_numpy()

# time to tune our test df accroding to our selected user ids

req_test_sparse = test_sparse_df.loc[req_user_ids, :]

# function for test and train item matrix and target labels

def train_test_i():
    num_cols = df_movies.shape[0]
    # train
    num_row_train = len(req_user_ids)*ITEM_BOUND
    final_out_train_X = np.empty((num_row_train, num_cols))
    final_out_train_Y = np.empty((num_row_train, 1))
    # test
    num_rows_test = len(req_user_ids)*TEST_BOUND
    final_out_test_X = np.empty((num_rows_test, num_cols))
    final_out_test_Y = np.empty((num_rows_test, 1))
    start_train = 0
    start_test = 0
    for user, interaction in zip(req_user_ids, req_num_interaction):
        item_range = range(IDX_OFFSET, df_movies.shape[0]+IDX_OFFSET)
        interacted_items = sparse_df.loc[sparse_df.USER_ID==user, 'MOVIE_ID'].to_numpy()
        y_interacted = req_test_sparse.loc[req_test_sparse.index==user, 'MOVIE_ID'].to_numpy()
        actual_interacted = np.concatenate((interacted_items, y_interacted), axis=None)
        non_interacted_items = set(item_range).difference(set(actual_interacted))
        # train
        end = start_train + ITEM_BOUND
        y_step = start_train + interaction
        req_num_non_interacted = ITEM_BOUND - interaction 
        non_interacted_items_train = rng.choice(list(non_interacted_items), 
                                                 size=req_num_non_interacted, replace=False)
        items_idx = np.concatenate((interacted_items, non_interacted_items_train), axis=None)
        final_out_train_X[start_train:end] = identity_item[items_idx-IDX_OFFSET]
        final_out_train_Y[start_train:y_step] = 1
        final_out_train_Y[y_step:end] = 0
        start_train = end
        # test
        end_t = start_test + TEST_BOUND
        y_step_t = start_test + INTERACTION_SIZE_TEST
        non_selected_items = non_interacted_items.difference(set(non_interacted_items_train))
        req_num_non_selected = TEST_BOUND - INTERACTION_SIZE_TEST
        non_interacted_items_test = rng.choice(list(non_selected_items), 
                                               size=req_num_non_selected, replace=False)
        item_idx_test = np.concatenate((y_interacted, non_interacted_items_test), axis=None)
        final_out_test_X[start_test:end_t] = identity_item[item_idx_test-IDX_OFFSET]
        final_out_test_Y[start_test:y_step_t] = 1
        final_out_test_Y[y_step_t:end_t] = 0
        start_test = end_t
    final_out_test_X = pd.DataFrame(final_out_test_X)
    final_out_test_Y = pd.DataFrame(final_out_test_Y)
    final_out_train_X = pd.DataFrame(final_out_train_X)
    final_out_train_Y = pd.DataFrame(final_out_train_Y)
    
    return final_out_train_X, final_out_train_Y, final_out_test_X, final_out_test_Y

# function for train and test user matrix

def train_test_u():
    num_cols = df_users.shape[0]
    # train
    num_row_train = len(req_user_ids)*ITEM_BOUND
    final_out_train_X = np.empty((num_row_train, num_cols))
    # test
    num_rows_test = len(req_user_ids)*TEST_BOUND
    final_out_test_X = np.empty((num_rows_test, num_cols))
    start_train = 0
    start_test = 0
    for user in req_user_ids:
        # train 
        end = start_train + ITEM_BOUND
        user_train = [user]*ITEM_BOUND
        final_out_train_X[start_train:end] = identity_user[user_train]
        start_train = end
        #test
        end_t = start_test + TEST_BOUND
        user_test = [user]*TEST_BOUND
        final_out_test_X[start_test:end_t] = identity_user[user_test]
        start_test = end_t
    final_out_train_X = pd.DataFrame(final_out_train_X)
    final_out_test_X = pd.DataFrame(final_out_test_X)
    
    return final_out_train_X, final_out_test_X

# data retrieval

train_x_i, train_y, test_x_i, test_y = train_test_i()
train_x_u, test_x_u = train_test_u()

# Model Architecture

# so, the original paper discuss 3 architectures, GMF short for Generalised Matrix Factorisation
# MLP - NCF framework for Multilayer Perceptron
# & NeuMF short for Neural Matrix Factorisation, basically the fusion of the two above
# now, NeuMF also has two forms, 1. the one which is trained from scratch
#                                2. the one which is trained from scratch, 
#                                   but the weights initialization is not random, like in the first case,
#                                   but uses pretrained weights of GMF and MLP trained seperately
# Now, if it's not clear, then you may refer to the original paper for more context and information
# we'll be implementing the 2nd one, which is supposed to be the main implementation in the paper

# GMF Architecture

# inputs
inputs_u = Input(shape=(U_INPUTS,))
inputs_i = Input(shape=(I_INPUTS,))
# embeddings
user_embedding = Dense(units=16)(inputs_u) # Dense Embedding for user
item_embedding = Dense(units=16)(inputs_i) # Dense Embedding for item
# dot
dot = Dot(axes=1)([user_embedding, item_embedding])

gmf_layer = Dense(units=1, activation='sigmoid')(dot)

gmf = keras.Model(inputs=[inputs_u, inputs_i], outputs=gmf_layer, name='GMF')

# MLP Architecture

# inputs
inputs_u = Input(shape=(U_INPUTS,))
inputs_i = Input(shape=(I_INPUTS,))
# embeddings
user_embedding = Dense(units=16)(inputs_u) 
item_embedding = Dense(units=16)(inputs_i)
# concatenation
concat_embeddings = Concatenate(axis=-1)([user_embedding, item_embedding])
# MLP - 64, 32, 16
layer1 = Dense(units=64, activation='relu')(concat_embeddings)
layer2 = Dense(units=32, activation='relu')(layer1)
layer3 = Dense(units=16, activation='relu')(layer2)
out = Dense(units=1, activation='sigmoid')(layer3)

mlp = keras.Model(inputs=[inputs_u, inputs_i], outputs=out, name='MLP')

# some tensorboard stuff for clean analysis

def get_tensorboard(model_name):
    
    folder_name = '{} at {}'.format(model_name, strftime("%H.%M"))
    dir_paths = os.path.join(LOG_DIR, folder_name)

    try:
        os.makedirs(dir_paths)
    except OSError as err: # built in exceptions in python like syntaxerror, typeerror, nameerror etc
    # these are called try catch blocks, they are used to mitigate errors in python code during runtime
    # they can be used for all built in exceptions in python in this given pattern.
        print(err.strerror)
    else:
        print('Successfully created directory')

    return TensorBoard(log_dir=dir_paths)

# compile GMF

gmf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# compile MLP

mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# training data

train_xu = train_x_u.to_numpy()
train_xi = train_x_i.to_numpy()
y_train = train_y.to_numpy()

# GMF training

gmf.fit([train_xu, train_xi], y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, 
        callbacks=[get_tensorboard('GMF')], verbose=0)

# MLP training

mlp.fit([train_xu, train_xi], y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, 
        callbacks=[get_tensorboard('MLP')], verbose=0)

# now, weights retrieval and initialization

# now there could be an easy way to do this, but I'm already very exhausted mentally, so I'll be very straight

# gmf weights and bias
gmf_ue, gmf_ue_b = tuple(gmf.layers[2].get_weights())
gmf_ie, gmf_ie_b = tuple(gmf.layers[3].get_weights())
gmf_weights, gmf_weights_b = tuple(gmf.layers[5].get_weights())

# initializer
gmf_ue, gmf_ue_b = tf.constant_initializer(gmf_ue), tf.constant_initializer(gmf_ue_b)
gmf_ie, gmf_ie_b = tf.constant_initializer(gmf_ie), tf.constant_initializer(gmf_ie_b)
gmf_weights, gmf_weights_b = tf.constant_initializer(gmf_weights), tf.constant_initializer(gmf_weights_b)

# mlp weights and bias
mlp_ue, mlp_ue_b = tuple(mlp.layers[2].get_weights())
mlp_ie, mlp_ie_b = tuple(mlp.layers[3].get_weights())
mlp_d1, mlp_d1_b = tuple(mlp.layers[5].get_weights())
mlp_d2, mlp_d2_b = tuple(mlp.layers[6].get_weights())
mlp_d3, mlp_d3_b = tuple(mlp.layers[7].get_weights())
mlp_d4, mlp_d4_b = tuple(mlp.layers[8].get_weights())

# initializer
mlp_ue, mlp_ue_b = tf.constant_initializer(mlp_ue), tf.constant_initializer(mlp_ue_b)
mlp_ie, mlp_ie_b = tf.constant_initializer(mlp_ie), tf.constant_initializer(mlp_ie_b)
mlp_d1, mlp_d1_b = tf.constant_initializer(mlp_d1), tf.constant_initializer(mlp_d1_b)
mlp_d2, mlp_d2_b = tf.constant_initializer(mlp_d2), tf.constant_initializer(mlp_d2_b)
mlp_d3, mlp_d3_b = tf.constant_initializer(mlp_d3), tf.constant_initializer(mlp_d3_b)
mlp_d4, mlp_d4_b = tf.constant_initializer(mlp_d4), tf.constant_initializer(mlp_d4_b)

# NeuMF Architecture

# inputs
inputs_u = Input(shape=(U_INPUTS,))
inputs_i = Input(shape=(I_INPUTS,))

# GMF section

# embeddings
user_embedding_gmf = Dense(units=16, 
                           kernel_initializer=gmf_ue, 
                           bias_initializer=gmf_ue_b)(inputs_u) # Dense Embedding for user
item_embedding_gmf = Dense(units=16, 
                           kernel_initializer=gmf_ie, 
                           bias_initializer=gmf_ie_b)(inputs_i) # Dense Embedding for item
# dot
dot = Dot(axes=1)([user_embedding_gmf, item_embedding_gmf])
# gmf layer
gmf_layer = Dense(units=1, kernel_initializer=gmf_weights, 
                  bias_initializer=gmf_weights_b)(dot)

# MLP section

user_embedding_mlp = Dense(units=16, kernel_initializer=mlp_ue, 
                           bias_initializer=mlp_ue_b)(inputs_u) 
item_embedding_mlp = Dense(units=16, kernel_initializer=mlp_ie, 
                           bias_initializer=mlp_ie_b)(inputs_i)
# concatenation
concat_embeddings = Concatenate(axis=-1)([user_embedding_mlp, item_embedding_mlp])
# MLP - 64, 32, 16
layer1 = Dense(units=64, activation='relu', kernel_initializer=mlp_d1, 
               bias_initializer=mlp_d1_b)(concat_embeddings)
layer2 = Dense(units=32, activation='relu', kernel_initializer=mlp_d2, 
               bias_initializer=mlp_d2_b)(layer1)
layer3 = Dense(units=16, activation='relu', kernel_initializer=mlp_d3, 
               bias_initializer=mlp_d3_b)(layer2)
layer4 = Dense(units=1, kernel_initializer=mlp_d4, 
               bias_initializer=mlp_d4_b)(layer3)

# concatenation of the two sections
concat = Concatenate(axis=-1)([gmf_layer*ALPHA, layer4*ALPHA]) # alpha represents tradeoff

# NeuMF
neumf_layer = Dense(units=1, activation='sigmoid')(concat)

neumf = keras.Model(inputs=[inputs_u, inputs_i], outputs=neumf_layer, name='NeuMF')

# compiling NeuMF

neumf.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

# fitting NeuMF

neumf.fit([train_xu, train_xi], y_train, batch_size=batch_size, epochs=nr_epochs, 
        callbacks=[get_tensorboard('NeuMF')], verbose=0)

# testing NeuMF

test_xu = test_x_u.to_numpy()
test_xi = test_x_i.to_numpy()
y_test = test_y.to_numpy()

test_loss, test_acc = neumf.evaluate([test_xu, test_xi], y_test)

print('Test loss : {:.3f} and Test Accuracy : {:.1%}'.format(test_loss, test_acc))

