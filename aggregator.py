import numpy as np
import filedata as fd
import tenseal as ts


def user_aggregator():
    
    #initialization of variables to store the data
    results_list = []   
    
    #Loading the context for encryption and decryption operations on data
    context = ts.context_from(fd.read_data("secret.txt"))
    
    #Loading the encrypted data of client1
    inp1_proto = fd.read_data('data_encrypted_Client1.txt')
    inp1 = ts.lazy_ckks_tensor_from(inp1_proto)
    inp1.link_context(context)
    
    #Loading the encrypted data of client2
    inp2_proto = fd.read_data('data_encrypted_Client2.txt')
    inp2 = ts.lazy_ckks_tensor_from(inp2_proto)
    inp2.link_context(context)
    
    #Adding the encrypted data of client1 and client2
    results1 = (inp1) + (inp2)
    
    #Dividing with the number of clients. Since there are 2 clients, float number 0.5 is multiplied to the Tensor as Fully Homomorphic Encryption supports multiplication but not division
    secret_key_context = ts.context_from(fd.read_data('secret.txt'))
    denominator_plain = ts.plain_tensor([0.5])
    denominator_ckks = ts.ckks_tensor(secret_key_context, denominator_plain)
    results = results1 * denominator_ckks    
    
    #Appending the aggregated result to a list
    for i in range(514):
        results_list.append(results[i])
        
    return results_list