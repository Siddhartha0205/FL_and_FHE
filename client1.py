#Importing necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import flwr as fl
import sys
import matplotlib.pyplot as plt
import encryption as encr
import time
from datetime import datetime
import tenseal as ts
import filedata as fd

#Declaration of certain variables
eval_accuracy = []
eval_loss = []
time_per_round = []
memory_per_round = []
if encr.Enc_needed.encryption_needed.value:
    serializedMemory_per_round = []

#Loading the CSV data file for model training
df = pd.read_csv('dataset_normal_client1.csv')

#setting up the global seed to control the randomness
tf.random.set_seed(42)

# DATA PREPROCESSING

# Remove unused features
df_removed = df.drop(columns = ['time', 'yaw', 'heading', 'location_x', 'location_y', 'gnss_latitude', 'gnss_longitude', 'gyroscope_x', 'gyroscope_y', 'height', 'reverse', 'hand_brake', 'manual_gear_shift', 'gear'])

# Engineer new features
import feature_engineering as fteng
df_engineered = fteng.engineer_features(df_removed)

# Aggregate all features
import feature_aggregration as ftagg
df_aggregated = ftagg.aggregate_features(df_engineered)

# Define input features and target
X = df_aggregated[['mean_speed', 'max_speed', 'mean_acceleration', 'max_acceleration', 'mean_yaw_speed', 'max_yaw_speed', 'mean_throttle', 'max_throttle_count', 'mean_steer_change', 'max_steer_change', 'mean_positive_brake_change', 'max_positive_brake_change']]
Y = df_aggregated['abnormality']

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, train_size = 0.8)

# Scale input features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X1_row, X1_col = np.shape(X_train)

#Building the model
dbp1_model = tf.keras.models.Sequential([
    keras.layers.Flatten(input_shape = (X1_col,)),
    keras.layers.Dense(16, activation = 'relu'),
    keras.layers.Dense(16, activation = 'relu'),
    keras.layers.Dense(2, activation = 'softmax')
])														

dbp1_model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ['accuracy'])	

keras.backend.clear_session()

#Explicitly setting the initial weights to decrease randomness of output
initial_weights_filename = "initial_weights_client1_1.h5"
dbp1_model.load_weights(initial_weights_filename)

#Loading training dataset of client - 2 as test dataset of client - 1
df_test = pd.read_csv('dataset_abnormal_client2.csv')

df_test_removed = df_test.drop(columns=['time', 'yaw', 'heading', 'location_x', 'location_y', 'gnss_latitude', 'gnss_longitude', 'gyroscope_x', 'gyroscope_y', 'height', 'reverse', 'hand_brake', 'manual_gear_shift', 'gear'])

# Engineer new features
import feature_test_engineering as ftengtest
df_test_engineered = fteng.engineer_features(df_test_removed)

# Aggregate all features
import feature_aggregration as ftagg
df_test_aggregated = ftagg.aggregate_features(df_test_engineered)

X_test = df_test_aggregated[['mean_speed', 'max_speed', 'mean_acceleration', 'max_acceleration', 'mean_yaw_speed', 'max_yaw_speed', 'mean_throttle', 'max_throttle_count', 'mean_steer_change', 'max_steer_change', 'mean_positive_brake_change', 'max_positive_brake_change']]
y_test = df_test_aggregated['abnormality']

X_test = scaler.transform(X_test)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        #FL setup initialization timestamp
        init_start = time.strftime("%H:%M:%S", time.localtime())
        print("Federated Learning at client-1 is initiated at: ", init_start)
        return dbp1_model.get_weights()

    def fit(self, parameters, config):
        #FL round training timestamp
        curr_time1 = time.strftime("%H:%M:%S", time.localtime())
        global fit_start
        fit_start = datetime.strptime(curr_time1, "%H:%M:%S") 
        print("Fit Round started at: ", curr_time1)
        
        dbp1_model.set_weights(parameters)
        
        r1 = dbp1_model.fit(X_train, y_train, batch_size= len(X_train), epochs = 5, validation_data = (X_valid, y_valid), verbose=0)
        hist1 = r1.history
        print("Fit history : " , hist1)
        
        #Adding encryption layer for dbp1_model.get_weights()  and then send them for aggregation
        if encr.Enc_needed.encryption_needed.value == 1:                                        #Full encryption depth is selected

            #Declaration of array to process encrypted CKKS vector
            params_encrypted = []

            #calling encryption function from encryption.py file
            params_encrypted_list, serialized_dataspace_ = encr.param_encrypt(dbp1_model.get_weights(), 'Client1')
            serializedMemory_per_round.append(serialized_dataspace_)
            print("Exiting the encryption layer and Back to communication - client1")
            
            for i in range(0, encr.num_modweights):
                params_encrypted.append(params_encrypted_list[i])
            
            print("Size of Encrypted weights of client1 in Bytes: ", sys.getsizeof(params_encrypted))
            memory_per_round.append(sys.getsizeof(params_encrypted))
            
            print("---- Initializing the aggegration of Encrypted model updates ----")
            
        elif encr.Enc_needed.encryption_needed.value == 2:                                      #Partial encryption depth is selected
            
            #Declaration of array to process encrypted CKKS vector
            params_encrypted = []

            #calling encryption function from encryption.py file
            params_encrypted_list, serialized_dataspace_ = encr.param_encrypt(dbp1_model.get_weights()[3: ], 'Client1')
            serializedMemory_per_round.append(serialized_dataspace_)
            print("Exiting the encryption layer and Back to communication - client1")
            
            for i in range(0, encr.num_modweights):
                params_encrypted.append(params_encrypted_list[i])
            
            print("Size of Encrypted weights of client1 in Bytes: ", sys.getsizeof(params_encrypted))
            memory_per_round.append(sys.getsizeof(params_encrypted))
            
            print("---- Initializing the aggegration of Encrypted model updates ----")
            
            print("Finished Aggregating the model updates from the clients and Back to communicaton - client1")
            
        else:                                                                                     #No encryption is selected
            
            params_encrypted = dbp1_model.get_weights()
        
        return dbp1_model.get_weights(), len(X_train), {}                                         #Model weights are shared wth the server as CKKS encrypted object is not support with Flower framework

    def evaluate(self, parameters, config):
        
        if encr.Enc_needed.encryption_needed.value == 1:                                          #Full encryption depth is selected
            
            print("Entered decryption layer - client1")
            
            #Calling decryption function from encryption.py file
            params_decrypted1 = encr.param_decrypt()
            
            print("Exiting decryption layer - client 1")
            
            # List to store the reshaped arrays
            reshaped_params = []

            # Define the shapes of the original arrays
            shapes = [np.shape(arr) for arr in parameters]

            # Variable to keep track of the current index in the data
            current_index = 0

            # Reshape the data and split it into individual arrays
            for shape in shapes:
                data_result = []
                size = np.prod(shape)
                
                #As the shape of model weights is not uniform, CKKS encryption fails to encrypt model weights as a tensor
                #As a solution, it is converted to vector i.e., 1-D vector and encryption - decryption is performed
                #To adapt the decrypted model weights to the model -> vector needsd to be reshaped back to its original shape 
                reshaped_arr = np.reshape(params_decrypted1[current_index:current_index + size], shape)
                reshaped_params.append(reshaped_arr)
                current_index += size

            print("Assigning the decrypted aggregated results to the model")
            
            dbp1_model.set_weights(reshaped_params)
            
        elif encr.Enc_needed.encryption_needed.value == 2:                                      #Partial encryption depth is selected
            
            print("Entered decryption layer - client1")

            #Calling decryption function from encryption.py file
            params_decrypted1 = encr.param_decrypt()

            print("Exiting decryption layer - client 1")
            
             # List to store the reshaped arrays
            reshaped_params = []

            # Define the shapes of the original arrays
            shapes = [np.shape(arr) for arr in parameters[3: ]]

            # Variable to keep track of the current index in the data
            current_index = 0

            # Reshape the data and split it into individual arrays
            for shape in shapes:
                data_result = []
                size = np.prod(shape)

                #As the shape of model weights is not uniform, CKKS encryption fails to encrypt model weights as a tensor
                #As a solution, it is converted to vector i.e., 1-D vector and encryption - decryption is performed
                #To adapt the decrypted model weights to the model -> vector needsd to be reshaped back to its original shape 
                reshaped_arr = np.reshape(params_decrypted1[current_index:current_index + size], shape)
                reshaped_params.append(reshaped_arr)
                current_index += size
            
            #Since latter half of the model needs to be encrypted in this feature, the latter half from the decrypted vector will concatenated with unencrypted model weights
            parameters[3: ] = reshaped_params
            
            print("Assigning the decrypted aggregated results to the model")
            
            dbp1_model.set_weights(parameters)
        
        else:                                                                                   #No encryption is selected                                                              

            memory_per_round.append(sys.getsizeof(parameters))
            dbp1_model.set_weights(parameters)
            
        loss, accuracy = dbp1_model.evaluate(X_test, y_test, verbose=0)

        #Storing timestamp as evaluation is finished in FL round
        curr_time2 = time.strftime("%H:%M:%S", time.localtime())
        global eval_end
        eval_end = datetime.strptime(curr_time2, "%H:%M:%S") 
        print("Evaluation Round is finished at: ", curr_time2)

        #Saving the evaluation values for plotting
        eval_accuracy.append(accuracy)

        #Saving the loss values for plotting
        eval_loss.append(loss)

        #Calculating the time taken to complete a FL round
        time_per_round.append((eval_end - fit_start).seconds)
        print('Time taken per round list: ', time_per_round)
        
        return loss, len(X_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address = "localhost: 8080", 
        client = FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)

#code snippet for confusion matrix
y1_pred = dbp1_model.predict(X_test)
#print(y1_pred)

y1_pred_labels = (y1_pred[:, 1] >= 0.51).astype(int)
confusion_client1 = confusion_matrix(y_test, y1_pred_labels)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y1_pred_labels)

print('Confusion Matrix:')
print(confusion_client1)

#plotting the model performance

x_points = np.array(range(1, len(np.array(eval_accuracy))+1))
print('Client1 - eval accuracy list:', eval_accuracy)
y_points1 = np.array(eval_accuracy)
y_points2 = np.array(eval_loss)

#Initialization of new lists to calculate total time duration and total memory consumed by FL setup
cumulative_time_per_round = time_per_round[:]
cumulative_memory_per_round = memory_per_round[:]

if encr.Enc_needed.encryption_needed.value:
    cumulative_serializedMemory_per_round = serializedMemory_per_round[:]

for i in range(1, len(time_per_round)):
    cumulative_time_per_round[i] = cumulative_time_per_round[i-1] + cumulative_time_per_round[i]
    cumulative_memory_per_round[i] = cumulative_memory_per_round[i-1] + cumulative_memory_per_round[i]
    if encr.Enc_needed.encryption_needed.value:
        cumulative_serializedMemory_per_round[i] = cumulative_serializedMemory_per_round[i-1] + cumulative_serializedMemory_per_round[i]

print("Instanteous Time: ", time_per_round)
print("Cumulative Time: ", cumulative_time_per_round)
if encr.Enc_needed.encryption_needed.value:
    print("Instantaneous Memory consumption by encrypted parameters: ",memory_per_round)
    print("Cumulative Memory consumption by encrypted parameters: ", cumulative_memory_per_round)
    print("Cumulative Memory consumption by serialized data: ", cumulative_serializedMemory_per_round)
else:
    print("Instantaneous Memory consumption by model parameters: ", memory_per_round)
    print("Cumulative Memory consumption by model parameters: ", cumulative_memory_per_round)

#Generating Plots to visualize

#Evaluation Accuracy and Evaluation Loss Metrics
plt.figure(1)
plt.plot(x_points, y_points1, '-o', color = 'green', label = 'Eval Accuracy')
plt.plot(x_points, y_points2,'-o',  color = 'blue', label = 'Eval Loss')
plt.legend()
if encr.Enc_needed.encryption_needed.value == 1:
    plt.title("Model Evaluation Metrics with encryption - client 1")
elif encr.Enc_needed.encryption_needed.value == 2:
    plt.title("Model Evaluation Metrics with partial encryption - client 1")
else:
    plt.title("Model Evaluation Metrics without encryption - client 1")
plt.xlabel("Number of FL rounds ")
plt.ylabel("Accuracy and Loss metrics of the Model ")

#Instantaneous and Cumulative Time duration 
plt.figure(3)
plt.plot(x_points, time_per_round, '-o', color = 'green', label = 'Instantaneous Time taken for each FL round')
plt.plot(x_points, cumulative_time_per_round, '-o', color = 'blue', label = 'Cumulative Time taken for each FL round')
plt.legend()
if encr.Enc_needed.encryption_needed.value == 1:
    plt.title('Time duration with encryption - client 1')
elif encr.Enc_needed.encryption_needed.value == 2:
    plt.title('Time duration with partial encryption - client 1')
else:
    plt.title('Time duration without encryption - client 1')
plt.xlabel("Number of FL rounds ")
plt.ylabel("Time Duration in seconds ")

#Instantaneous and Cumulative Memory consumption
plt.figure(5)
plt.plot(x_points, memory_per_round, '-o', color = 'green', label = 'Instantaneous Memory consumed in each FL round')
plt.plot(x_points, cumulative_memory_per_round, '-o', color = 'blue', label = 'Cumulative Memory consumed in each FL round')
plt.legend()
if encr.Enc_needed.encryption_needed.value == 1:
    plt.title('Memory consumption with encryption - client 1')
elif encr.Enc_needed.encryption_needed.value == 2:
    plt.title('Memory consumption with partial encryption - client 1')
else:
    plt.title('Memory consumption without encryption - client 1')
plt.xlabel("Number of FL rounds ")
plt.ylabel("Memory consumed by parameters in Bytes ")

#Instantaneous and Cumulative Serialized Memory consumption
if encr.Enc_needed.encryption_needed.value:
    plt.figure(7)
    plt.plot(x_points, serializedMemory_per_round, '-o', color = 'green', label = 'Instantaneous Serialized Memory consumed in each FL round')
    plt.plot(x_points, cumulative_serializedMemory_per_round, '-o', color = 'blue', label = 'Cumulative Serialized Memory consumed in each FL round')
    plt.legend()
    if encr.Enc_needed.encryption_needed.value == 1:
        plt.title('Serialized Memory consumption with encryption - client 1')
    elif encr.Enc_needed.encryption_needed.value == 2:
        plt.title('Serialized Memory consumption with partial encryption - client 1')
    plt.xlabel("Number of FL rounds ")
    plt.ylabel("Memory consumed by parameters in Mega Bytes ")

#ROC curve
plt.figure(9)
plt.plot(fpr, tpr, '-o', label = 'ROC Curve Display from predictions')
if encr.Enc_needed.encryption_needed.value == 1:
    plt.title('ROC Curve with encryption - client 1')
elif encr.Enc_needed.encryption_needed.value == 2:
    plt.title('ROC Curve with partial encryption - client 1')
else:
    plt.title('ROC Curve without encryption - client 1')
plt.xlabel("False Positive Rate ")
plt.ylabel("True Positive Rate ")

plt.show()

del dbp1_model