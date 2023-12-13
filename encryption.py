#Importing necessary libraries
import tenseal as ts
import filedata as fd
import numpy as np
from enum import Enum
import sys

#This block decides if the Federated Learning setup undergoes encryption and decryption during communication
class Enc_needed(Enum):
    #encryption_needed = 0 : data encryption and decryption is not necessary and thus regular Federated Learning is carried out
    #encyrption_needed = 1 : data encryption and decryption is necessary -> Full encryption
    #encyrption_needed = 2 : data encryption and decryption is necessary for a fraction of model -> Partial encryption
    encryption_needed = 1
        
def create_context():                                       #Declaration of context to generate keys 
    global context
    context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree = 8192,
    coeff_mod_bit_sizes = [60, 40, 40, 60]
    )
    
    #generating public key and private key pair
    context.generate_galois_keys()
    context.global_scale = 2**40
        
    #generting secret key and saving it in a text file
    secret_key_context = context.serialize(save_secret_key = True)
    private_key_file = "secret.txt"
    fd.write_data(private_key_file, secret_key_context)
        
    #generating public key and saving it in a text file
    context.make_context_public()                           #drops the private key
    public_key_context = context.serialize()
    public_key_file = "public_key.txt"
    fd.write_data(public_key_file, public_key_context)

def param_encrypt(param_list, clientID: str):               #Function to implement encryption
    print("---- Entered encryption layer ----")
    
    #Loading public key for encryption
    public_key_context = ts.context_from(fd.read_data("public_key.txt"))    
    
    #print("Flattening the model updates")
    concatenated_data = np.concatenate([np.array(arr).flatten() for arr in param_list])
    global num_modweights
    num_modweights = len(concatenated_data)
          
    #print('concatenated data', concatenated_data)
    print("---- Initiating CKKS encryption for model updates ----")

    #Generating a plain-text tensor from message data
    plain_text = ts.plain_tensor(concatenated_data)

    #Generating a cipher-text tensor from plain-text tensor
    data_encrypted_list = ts.ckks_tensor(public_key_context, plain_text)

    #Creating a text file considering client ID and encryption depth selected
    if Enc_needed.encryption_needed.value == 1:
        encrypted_data_file_path = "data_encrypted_" + str(clientID) + ".txt"
    elif Enc_needed.encryption_needed.value == 2:
        encrypted_data_file_path = "data_encrypted_2_" + str(clientID) + ".txt"
    
    #Writing the encrypted data into the respective file
    fd.write_data(encrypted_data_file_path, data_encrypted_list.serialize())

    #Calculating the size of serialized memory of encrypted data
    serialized_dataspace = sys.getsizeof(data_encrypted_list.serialize())/(1024*1024)
    print('Memory space occupied by serialized data in Mega Bytes: ', serialized_dataspace)
        
    return data_encrypted_list, serialized_dataspace

def param_decrypt():                                        #Function to implement decryption
    
    #Loading secret key to decrypted the encrypted data
    secret_context = ts.context_from(fd.read_data('secret.txt'))
    
    #Selecting the text file that stores aggregation results for decryption   
    if Enc_needed.encryption_needed.value == 1:
        new_result_proto = fd.read_data('result_ex.txt')
    elif Enc_needed.encryption_needed.value == 2:
        new_result_proto = fd.read_data('result_ex_2.txt')

    #Reading the aggregated data from the result file
    new_result = ts.lazy_ckks_tensor_from(new_result_proto)
    new_result.link_context(secret_context)

    #Returning the decrypted data in the form of a list
    return new_result.decrypt().tolist()