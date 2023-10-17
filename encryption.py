import tenseal as ts
import filedata as fd
import numpy as np
from enum import Enum


#This block decides if the Federated Learning setup undergoes encryption and decryption during communication
class Enc_needed(Enum):

    #Initialising a boolean variable to control the requirement of encryption and decryption
    #encryption_needed = 0 indicates that data encryption and decryption is not necessary and thus regular Federated Learning is carried out
    #encyrption_needed = 1 indicates that data encryption and decryption is necessary
    encryption_needed = 1
        
def create_context():
    global context
    context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree = 8192,
    coeff_mod_bit_sizes = [60, 40, 40, 60]
    )
    
    #generating public key and private key pair
    context.generate_galois_keys()
    context.global_scale = 2**40
        
    #generting private key and saving it in a text file
    secret_key_context = context.serialize(save_secret_key = True)
    #private_key_file = "private_key_" + clientID + ".txt"
    private_key_file = "secret.txt"
    fd.write_data(private_key_file, secret_key_context)
        
    #generating public key and saving it in a text file
    context.make_context_public()                         #drops the private key
    public_key_context = context.serialize()
    #public_key_file = "public_key_" + clientID + ".txt"
    public_key_file = "public_key.txt"
    fd.write_data(public_key_file, public_key_context)

def param_encrypt(param_list, clientID: str):
     
    temp_decr_list = []
    
    #encrypt messages
    public_key_context = ts.context_from(fd.read_data("public_key.txt"))    
    secret_key_context = ts.context_from(fd.read_data("secret.txt"))
    
    #As model has different shapes of the parameter arrays, it is not possible to encrypt without uniform shape in tenseal.
    #Hence it is decided to Flatten the array to 1-D Tensor (Vector in fact) and then proceed with encryption 
    #print('Flattening line')
    concatenated_data = np.concatenate([np.array(arr).flatten() for arr in param_list])
    #print('concatenated data', concatenated_data)
    
    #The flattened tensor is converted to plain text (polynomial) tensor object
    plain_text = ts.plain_tensor(concatenated_data)
    #print('plain text at line 54 of encryption file: ', plain_text)
    
    #The plain text tensor object is now converted to encrypted CKKS tensor object
    data_encrypted_list_2 = ts.ckks_tensor(secret_key_context, plain_text)
    
    #Temporary check - decrypting the encrypted values to check if the values after decryption matches with the original data 
    #temp_decr_list.append(data_encrypted_list_2.decrypt().tolist())  
    #print('Temporary decryption list: ', temp_decr_list)
    
    #Creating a file to store the encrypted CKKS tensor object for each client
    encrypted_data_file_path = "data_encrypted_" + str(clientID) + ".txt"
    fd.write_data(encrypted_data_file_path, data_encrypted_list_2.serialize())
      
    return data_encrypted_list_2

def decrypt(enc, dec_context_):
    #print(type(enc))
    #print('length: ', len(enc.decrypt().tolist()))
    
    return enc.decrypt().tolist()

def param_decrypt(weights, clientID: str):
    
    decrypted_result = []
    
    #Loading context for decrypting the aggregated results of each client
    secret_context = ts.context_from(fd.read_data('secret.txt'))   
    
    #Each element in encrypted aggregated weight tensor is decrypted and added to the list named 'decrypted_result'
    for item in weights:
        #print('item: ', item)
        decrypted_result.append(decrypt(item, secret_context))
    
    return decrypted_result
    
    