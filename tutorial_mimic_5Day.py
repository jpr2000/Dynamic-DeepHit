# In[1]:

print("Top of the morning to you, good sir.")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


_EPSILON = 1e-08

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os

from sklearn.model_selection import train_test_split

import import_mimic as impt


from class_DeepLongitudinal import Model_Longitudinal_Attention

from utils_eval             import c_index, brier_score
from utils_log              import save_logging, save_string, load_logging
from utils_helper           import f_get_minibatch, f_get_boosted_trainset


# In[154]:


#int(1.2*6.1) = 7 


# In[157]:


def _f_get_pred(sess, model, data, data_mi, pred_horizon):
    '''
        predictions based on the prediction time.
        create new_data and new_mask2 that are available previous or equal to the prediction time (no future measurements are used)
    '''
    new_data    = np.zeros(np.shape(data))
    new_data_mi = np.zeros(np.shape(data_mi))

    meas_time = np.concatenate([np.zeros([np.shape(data)[0], 1]), np.cumsum(data[:, :, 0], axis=1)[:, :-1]], axis=1)

    for i in range(np.shape(data)[0]):
        last_meas = np.sum(meas_time[i, :] <= pred_horizon)

        new_data[i, :last_meas, :]    = data[i, :last_meas, :]
        new_data_mi[i, :last_meas, :] = data_mi[i, :last_meas, :]

    return model.predict(new_data, new_data_mi)


def f_get_risk_predictions(sess, model, data_, data_mi_, pred_time, eval_time):
    
    pred = _f_get_pred(sess, model, data_[[0]], data_mi_[[0]], 0)
    _, num_Event, num_Category = np.shape(pred)
       
    risk_all = {}
    for k in range(num_Event):
        risk_all[k] = np.zeros([np.shape(data_)[0], len(pred_time), len(eval_time)])
            
    for p, p_time in enumerate(pred_time):
        ### PREDICTION
        pred_horizon = int(p_time)
        pred = _f_get_pred(sess, model, data_, data_mi_, pred_horizon)
        #print(pred.shape)
        # print("Pred ==============================================")
        # print(pred)


        for t, t_time in enumerate(eval_time):
            eval_horizon = int(t_time) + pred_horizon #if eval_horizon >= num_Category, output the maximum...

            # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
            risk = np.sum(pred[:,:,pred_horizon:(eval_horizon+1)], axis=2) #risk score until eval_time
            risk = risk / (np.sum(np.sum(pred[:,:,pred_horizon:], axis=2), axis=1, keepdims=True) +_EPSILON) #conditioniong on t > t_pred
            
            for k in range(num_Event):
                risk_all[k][:, p, t] = risk[:, k]
                
    return pred, risk_all


# ### 1. Import Dataset
# #####      - Users must prepare dataset in csv format and modify 'import_data.py' following our examplar 'PBC2'

# In[158]:


data_mode                   = 'PBC2' 
seed                        = 1234

##### IMPORT DATASET
'''
    num_Category            = max event/censoring time * 1.2
    num_Event               = number of evetns i.e. len(np.unique(label))-1
    max_length              = maximum number of measurements
    x_dim                   = data dimension including delta (1 + num_features)
    x_dim_cont              = dim of continuous features
    x_dim_bin               = dim of binary features
    mask1, mask2, mask3     = used for cause-specific network (FCNet structure)
'''

if data_mode == 'PBC2':
    (x_dim, x_dim_cont, x_dim_bin), (data, time, label), (mask1, mask2, mask3), (data_mi) = impt.import_dataset(norm_mode = 'standard')
    
    # This must be changed depending on the datasets, prediction/evaliation times of interest
    #pred_time = [52, 3*52, 5*52] # prediction time (in months)
    #pred_time = [300]
    #eval_time = [12, 36, 60, 120] # months evaluation time (for C-index and Brier-Score)
    #eval_time = [6]
    
    # pred_time = [5 * 24]
    # eval_time = [1, 2, 3, 4, 5, 6]
    pred_time = [30, 60, 90]
    eval_time = [30, 60, 90]
else:
    print ('ERROR:  DATA_MODE NOT FOUND !!!')

_, num_Event, num_Category  = np.shape(mask1)  # dim of mask3: [subj, Num_Event, Num_Category]
max_length                  = np.shape(data)[1]

from datetime import datetime
file_path = '{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))

if not os.path.exists(file_path):
    os.makedirs(file_path)


# In[159]:


file_path


# In[96]:


label.shape


# In[68]:


data.shape


# In[15]:


data.shape


# In[149]:


data.shape[0] * .8


# ### 2. Set Hyper-Parameters
# ##### - Play with your own hyper-parameters!

# In[51]:


# burn_in_mode                = 'ON' #{'ON', 'OFF'}
# boost_mode                  = 'OFF' #'ON' #{'ON', 'OFF'}

# ##### HYPER-PARAMETERS
# # TODO: Change to 32
# new_parser = {'mb_size': 32,

#              'iteration_burn_in': 3000,
#              #'iteration': 25000,
#               'iteration': 10000,

#              'keep_prob': 0.6,
#              'lr_train': 3e-4, # 1e-4

#              'h_dim_RNN': 100,
#              'h_dim_FC' : 100,
#              'num_layers_RNN':2,
#              'num_layers_ATT':2,
#              'num_layers_CS' :2,

#              'RNN_type':'GRU', #{'LSTM', 'GRU'}

#              'FC_active_fn' : tf.nn.relu,
#              'RNN_active_fn': tf.nn.relu,

#             'reg_W'         : 1e-5,
#             'reg_W_out'     : 0.,

#              'alpha' :1.0,
#              'beta'  :0.1,
#              'gamma' :1.0
# }


# # INPUT DIMENSIONS
# input_dims                  = { 'x_dim'         : x_dim,
#                                 'x_dim_cont'    : x_dim_cont,
#                                 'x_dim_bin'     : x_dim_bin,
#                                 'num_Event'     : num_Event,
#                                 'num_Category'  : num_Category,
#                                 'max_length'    : max_length }
# print(input_dims)

# # NETWORK HYPER-PARMETERS
# network_settings            = { 'h_dim_RNN'         : new_parser['h_dim_RNN'],
#                                 'h_dim_FC'          : new_parser['h_dim_FC'],
#                                 'num_layers_RNN'    : new_parser['num_layers_RNN'],
#                                 'num_layers_ATT'    : new_parser['num_layers_ATT'],
#                                 'num_layers_CS'     : new_parser['num_layers_CS'],
#                                 'RNN_type'          : new_parser['RNN_type'],
#                                 'FC_active_fn'      : new_parser['FC_active_fn'],
#                                 'RNN_active_fn'     : new_parser['RNN_active_fn'],
#                                 'initial_W'         : tf.contrib.layers.xavier_initializer(),

#                                 'reg_W'             : new_parser['reg_W'],
#                                 'reg_W_out'         : new_parser['reg_W_out']
#                                  }


# mb_size           = new_parser['mb_size']
# iteration         = new_parser['iteration']
# iteration_burn_in = new_parser['iteration_burn_in']

# keep_prob         = new_parser['keep_prob']
# lr_train          = new_parser['lr_train']

# alpha             = new_parser['alpha']
# beta              = new_parser['beta']
# gamma             = new_parser['gamma']

# # SAVE HYPERPARAMETERS
# log_name = file_path + '/hyperparameters_log.txt'
# save_logging(new_parser, log_name)

# print_log = file_path + '/print_log.txt'

burn_in_mode                = 'OFF' #{'ON', 'OFF'}
boost_mode                  = 'OFF' #'ON' #{'ON', 'OFF'}


param_dict = {
    "n_node_shared": [50, 100, 200, 300],
    "n_node_specific": [50, 100, 200, 300],
    "n_node_attention": [50, 100, 200, 300],
    "alpha": [0.1, 1, 3, 5],
    "beta": [0.1, 1, 3, 5],
    "gamma": [0.1, 1, 3, 5],
    "n_layers_shared": [1, 2, 3],
    "n_layers_specific": [1, 2, 3, 5],
    "rnn_type": ['LSTM', 'GRU'],
    "mini_batch": [4, 32, 64, 128],
}
    
import random    
def hyper_retriever(param):
    value_list = param_dict[param]
    return random.choice(value_list)
    


##### HYPER-PARAMETERS
new_parser = {'mb_size': 32,

             # 'iteration_burn_in': 3000,
             #'iteration': 25000,
              'iteration_burn_in': 500,
              'iteration': 10000,

             'keep_prob': 1.0,
             'lr_train': 3e-4,

             'h_dim_RNN': 1000,
             'h_dim_FC' : 1000,
             'num_layers_RNN':5,
             'num_layers_ATT':5,
             'num_layers_CS' :5,

             'RNN_type':'LSTM', #{'LSTM', 'GRU'}

             'FC_active_fn' : tf.nn.relu,
             'RNN_active_fn': tf.nn.tanh,

            'reg_W'         : 0., # 1e-5,
            'reg_W_out'     : 0.,

             'alpha' :5.0,
             'beta'  :0.1,
             'gamma' :0.1, # 1.0
}


# INPUT DIMENSIONS
input_dims                  = { 'x_dim'         : x_dim,
                                'x_dim_cont'    : x_dim_cont,
                                'x_dim_bin'     : x_dim_bin,
                                'num_Event'     : num_Event,
                                'num_Category'  : num_Category,
                                'max_length'    : max_length }

# NETWORK HYPER-PARMETERS
network_settings            = { 'h_dim_RNN'         : new_parser['h_dim_RNN'],
                                'h_dim_FC'          : new_parser['h_dim_FC'],
                                'num_layers_RNN'    : new_parser['num_layers_RNN'],
                                'num_layers_ATT'    : new_parser['num_layers_ATT'],
                                'num_layers_CS'     : new_parser['num_layers_CS'],
                                'RNN_type'          : new_parser['RNN_type'],
                                'FC_active_fn'      : new_parser['FC_active_fn'],
                                'RNN_active_fn'     : new_parser['RNN_active_fn'],
                                'initial_W'         : tf.contrib.layers.xavier_initializer(),

                                'reg_W'             : new_parser['reg_W'],
                                'reg_W_out'         : new_parser['reg_W_out']
                                 }


mb_size           = new_parser['mb_size']
iteration         = new_parser['iteration']
iteration_burn_in = new_parser['iteration_burn_in']

keep_prob         = new_parser['keep_prob']
lr_train          = new_parser['lr_train']

alpha             = new_parser['alpha']
beta              = new_parser['beta']
gamma             = new_parser['gamma']

# SAVE HYPERPARAMETERS
log_name = file_path + '/hyperparameters_log.txt'
save_logging(new_parser, log_name)
print_log = file_path + '/old_new_print_log.txt'

import pickle
import copy
total_dict = {
    "input_dims": copy.deepcopy(input_dims),
    "network_settings": copy.deepcopy(network_settings)
}
total_dict['network_settings']['initial_W'] = 'xavier'

with open(file_path + '/saved_dictionary.pkl', 'wb') as f:
    pickle.dump(total_dict, f)


### TRAINING-TESTING SPLIT
# TODO: could do stratified k-fold
(tr_data,te_data, tr_data_mi, te_data_mi, tr_time,te_time, tr_label,te_label, 
 tr_mask1,te_mask1, tr_mask2,te_mask2, tr_mask3,te_mask3) = train_test_split(data, data_mi, time, label, mask1, mask2, mask3, test_size=0.2, random_state=seed) 

(tr_data,va_data, tr_data_mi, va_data_mi, tr_time,va_time, tr_label,va_label, 
 tr_mask1,va_mask1, tr_mask2,va_mask2, tr_mask3,va_mask3) = train_test_split(tr_data, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3, test_size=0.125, random_state=seed) 

# if boost_mode == 'ON':
#     tr_data, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3 = f_get_boosted_trainset(tr_data, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3)

print("shapes")
print(tr_data.shape, va_data.shape, te_data.shape)
print(tr_data)
# raise Exception()

# In[162]:


tr_data.shape, va_data.shape, te_data.shape


# In[168]:


va_label.sum() / 631, te_label.sum()/te_data.shape[0],  tr_label.sum()/tr_data.shape[0]


# In[13]:


va_data.shape


# In[14]:


te_data.shape


# In[15]:


tr_data.shape


# In[17]:


import time


# ### 4. Train the Networ

# In[19]:


##### CREATE DYNAMIC-DEEPFHT NETWORK
tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = Model_Longitudinal_Attention(sess, "Dyanmic-DeepHit", input_dims, network_settings)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())


# In[151]:


tr_data.shape


# In[169]:


print("Entering print mode")


# In[ ]:

save_string('REGULAR -\nThis message will be written to a file.', print_log)

PRINT_ITER = 10
IS_SAVING_BEST = True

start = time.time()
### TRAINING - BURN-IN
if burn_in_mode == 'ON':
    save_string( "BURN-IN TRAINING ...", print_log)
    for itr in range(iteration_burn_in):
        x_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb = f_get_minibatch(mb_size, tr_data, tr_data_mi, tr_label, tr_time, tr_mask1, tr_mask2, tr_mask3)
        DATA = (x_mb, k_mb, t_mb)
        MISSING = (x_mi_mb)

        _, loss_curr = model.train_burn_in(DATA, MISSING, keep_prob, lr_train)

        if (itr+1)%PRINT_ITER == 0:
            save_string('itr: {:04d} | loss: {:.4f}'.format(itr+1, loss_curr), print_log)


### TRAINING - MAIN
save_string( "MAIN TRAINING ...", print_log)
min_valid = 0.5


for itr in range(iteration):
    x_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb = f_get_minibatch(mb_size, tr_data, tr_data_mi, tr_label, tr_time, tr_mask1, tr_mask2, tr_mask3)
    DATA = (x_mb, k_mb, t_mb)
    MASK = (m1_mb, m2_mb, m3_mb)
    MISSING = (x_mi_mb)
    PARAMETERS = (alpha, beta, gamma)

    _, loss_curr = model.train(DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train)

    if (itr+1)%PRINT_ITER == 0:
        save_string('itr: {:04d} | loss: {:.4f}'.format(itr+1, loss_curr), print_log)

    ### VALIDATION  (based on average C-index of our interest)
    if (itr+1)%PRINT_ITER == 0:        
        pred, risk_all = f_get_risk_predictions(sess, model, va_data, va_data_mi, pred_time, eval_time)

        for p, p_time in enumerate(pred_time):
            pred_horizon = int(p_time)
            val_result1 = np.zeros([num_Event, len(eval_time)])

            for t, t_time in enumerate(eval_time):                
                eval_horizon = int(t_time) + pred_horizon
                for k in range(num_Event):
                    val_result1[k, t] = c_index(risk_all[k][:, p, t], va_time, (va_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)

            if p == 0:
                val_final1 = val_result1
            else:
                val_final1 = np.append(val_final1, val_result1, axis=0)

        tmp_valid = np.mean(val_final1)
        save_string('val c_index: {:.4f}'.format(tmp_valid), print_log)
        
        if IS_SAVING_BEST:
            if tmp_valid >  min_valid:
                min_valid = tmp_valid
                saver.save(sess, file_path + '/best_model')
                save_string( 'updated.... best average c-index = ' + str('%.4f' %(tmp_valid)), print_log)
        # else:
        # min_valid = tmp_valid
        saver.save(sess, file_path + '/model')
        save_string( 'average c-index = ' + str('%.4f' %(tmp_valid)), print_log)
            

end = time.time()
save_string('Elapsed Time= ' + str(end - start), print_log)

save_string("================================", print_log)
save_string("         END OF TRAIN           ", print_log)
save_string("================================", print_log)


# In[93]:


from datetime import date, datetime
datetime.now().strftime("%Y%m%d_%H%M%S")


# In[14]:


with open('DD_pred_val_5day_Tmax120.npy', 'wb') as f:
    np.save(f, pred)


# In[15]:


with open('DD_risk_all_val_5day_Tmax120.npy', 'wb') as f:
    np.save(f, risk_all)


# In[17]:


with open('DD_pred_val_5day_Tmax120.npy', 'rb') as f:
    temp1 = np.load(f)


# In[ ]:


# We have pred and risk_all
temp1.shape


# ### 5. Test the Trained Network

# In[99]:


file_path


# In[21]:



# pred_time = [5 * 24]
# eval_time = [1, 2, 3, 4, 5, 6]
saver.restore(sess, file_path + '/model')


pred_test, risk_all = f_get_risk_predictions(sess, model, te_data, te_data_mi, pred_time, eval_time)

for p, p_time in enumerate(pred_time):
    pred_horizon = int(p_time)
    result1, result2 = np.zeros([num_Event, len(eval_time)]), np.zeros([num_Event, len(eval_time)])

    for t, t_time in enumerate(eval_time):                
        eval_horizon = int(t_time) + pred_horizon
        for k in range(num_Event):
            result1[k, t] = c_index(risk_all[k][:, p, t], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
            result2[k, t] = brier_score(risk_all[k][:, p, t], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
    
    if p == 0:
        final1, final2 = result1, result2
    else:
        final1, final2 = np.append(final1, result1, axis=0), np.append(final2, result2, axis=0)
        
        
row_header = []
for p_time in pred_time:
    for t in range(num_Event):
        row_header.append('pred_time {}: event_{}'.format(p_time,k+1))
            
col_header = []
for t_time in eval_time:
    col_header.append('eval_time {}'.format(t_time))

# c-index result
df1 = pd.DataFrame(final1, index = row_header, columns=col_header)

# brier-score result
df2 = pd.DataFrame(final2, index = row_header, columns=col_header)

### PRINT RESULTS
print('========================================================')
print('--------------------------------------------------------')
print('- C-INDEX: ')
print(df1)
print('--------------------------------------------------------')

print('- BRIER-SCORE: ')
print(df2)
print('========================================================')


# In[25]:


tr_data.shape, tr_data_mi.shape


# In[61]:


tr_data[0,:,0]


# In[29]:


te_data.shape, te_data_mi.shape


# ## Prediction

# In[45]:


preds_train = model.predict(tr_data, tr_data_mi)


# In[62]:


preds_train.shape


# In[98]:


# preds_val = 
preds_val = model.predict(va_data, va_data_mi)
preds_test = model.predict(te_data, te_data_mi)

preds_val.shape, preds_test.shape


# In[119]:


arr = []
for_flagging = pd.DataFrame()


# In[115]:


preds_test = preds_test.reshape(1262, 120)
preds_test.shape


# In[124]:


np.repeat(2, 120)


# In[125]:


arr = np.array([])
for i in range(1262):
    new = np.repeat(i, 120)
    arr = np.concatenate((arr, new), axis=None)


# In[133]:


te_label.shape


# In[131]:


for_flagging['id'] = arr
for_flagging['hazard'] = preds_test.reshape(1262*120)
for_flagging


# In[136]:


# for i in te_label:
#     print(i)
#     return

arr = np.array([])
for i in range(1262):
    new = np.repeat(i, 120)
    arr = np.concatenate((arr, new), axis=None)


# In[137]:


l_arr = np.array([])
for i in te_label:
    new = np.repeat(i, 120)
    l_arr = np.concatenate((l_arr, new), axis=None)
l_arr.shape


# In[143]:


for_flagging['label'] = l_arr
for_flagging


# In[120]:


for i, row in enumerate(preds_test):
    for j, cell in enumerate(row):
        for_flagging.iloc[i*j + j, 0] = i


# In[ ]:





# ## Prediction - Other

# In[83]:


preds_total = model.predict(data, data_mi)


# In[85]:


preds_total.shape


# In[84]:


preds_total


# In[46]:


# Zhale's older stuff below


# In[23]:


pred_test.shape


# In[27]:


pred_test_reshaped = pred_test.reshape(6309, 120)


# In[29]:


pred_test_df = pd.DataFrame(pred_test_reshaped)


# In[30]:


pred_test_df.to_csv('DD_pred_test_5day_Tmax120.csv', index = False)


# In[26]:


with open('DD_pred_test_5day.npy', 'wb') as f:
    np.save(f, pred_test)


# In[31]:


with open('DD_risk_all_test_5day_Tmax120.npy', 'wb') as f:
    np.save(f, risk_all)


# In[28]:


risk_all[0].shape


# In[29]:


pred_test.shape


# In[18]:


pred_test[:10]


# In[23]:


data[21]


# In[17]:


pred_test_modified = pred_test.reshape(1000,6)
first = []
second = []
third = []
forth = []
fifth = []
sixth = []
for i in range(1000):
  first.append(pred_test_modified[i][0])
  second.append(pred_test_modified[i][1])
  third.append(pred_test_modified[i][2])
  forth.append(pred_test_modified[i][3])
  fifth.append(pred_test_modified[i][4])
  sixth.append(pred_test_modified[i][5])


# In[142]:


#pred_time = 5


# In[143]:


predictions_test5 = pd.DataFrame()
predictions_test5['o1'] = first
predictions_test5['o2'] = second
predictions_test5['o3'] = third
predictions_test5['o4'] = forth
predictions_test5['o5'] = fifth
predictions_test5['o6'] = sixth


# In[144]:


predictions_test5[:10]


# In[ ]:


#pred_time = 1


# In[18]:


predictions_test1 = pd.DataFrame()
predictions_test1['o1'] = first
predictions_test1['o2'] = second
predictions_test1['o3'] = third
predictions_test1['o4'] = forth
predictions_test1['o5'] = fifth
predictions_test1['o6'] = sixth


# In[21]:


predictions_test1[:10]


# In[ ]:


#pred_time = 4


# In[105]:


predictions_test4 = pd.DataFrame()
predictions_test4['o1'] = first
predictions_test4['o2'] = second
predictions_test4['o3'] = third
predictions_test4['o4'] = forth
predictions_test4['o5'] = fifth
predictions_test4['o6'] = sixth


# In[106]:


predictions_test4[:10]


# #pred_time = 3

# In[79]:


pred_test_modified = pred_test.reshape(1000,6)
first = []
second = []
third = []
forth = []
fifth = []
sixth = []
for i in range(1000):
  first.append(pred_test_modified[i][0])
  second.append(pred_test_modified[i][1])
  third.append(pred_test_modified[i][2])
  forth.append(pred_test_modified[i][3])
  fifth.append(pred_test_modified[i][4])
  sixth.append(pred_test_modified[i][5])


# In[80]:


predictions_test3 = pd.DataFrame()
predictions_test3['o1'] = first
predictions_test3['o2'] = second
predictions_test3['o3'] = third
predictions_test3['o4'] = forth
predictions_test3['o5'] = fifth
predictions_test3['o6'] = sixth


# In[81]:


predictions_test3[:10]


# In[87]:


risk_all[0][0:10]

#Pred_time = 2
# In[60]:


pred_test_modified = pred_test.reshape(1000,6)
first = []
second = []
third = []
forth = []
fifth = []
sixth = []
for i in range(1000):
  first.append(pred_test_modified[i][0])
  second.append(pred_test_modified[i][1])
  third.append(pred_test_modified[i][2])
  forth.append(pred_test_modified[i][3])
  fifth.append(pred_test_modified[i][4])
  sixth.append(pred_test_modified[i][5])


# In[61]:


predictions_test = pd.DataFrame()
predictions_test['o1'] = first
predictions_test['o2'] = second
predictions_test['o3'] = third
predictions_test['o4'] = forth
predictions_test['o5'] = fifth
predictions_test['o6'] = sixth


# In[63]:


predictions_test[:10]


# In[ ]:




