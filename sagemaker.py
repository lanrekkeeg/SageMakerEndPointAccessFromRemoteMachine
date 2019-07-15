import boto3
import ast
import numpy as np


def sagemaker_endpoint():
    payload = []
    with open("floppy.jpg", 'rb') as datapoint:
        payload = datapoint.read()
        payload = bytearray(payload)

    endpoint_name = "#endpoint name"
    runtime = boto3.Session().client(service_name='sagemaker-runtime',region_name='AWS sagemaker', aws_access_key_id='#aws aws_access_key_id',
        aws_secret_access_key='#aws_secret_access_key')
    response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/x-image', Body=payload)
    probs = response['Body'].read().decode() # byte array
    probs = ast.literal_eval(probs) # array of floats
    probs = np.array(probs) # numpy array of floats

    topk_indexes = probs.argsort() # indexes in ascending order of probabilities
    topk_indexes = topk_indexes[::-1][:257] # indexes for top k probabilities in descending order
    topk_categories = []

    # here we take first three
    # off total 257 classes
    # output shape: {classname: prob}
    for i in range(3):
           topk_categories.append((i+1, probs[i]))
    print(str(topk_categories))

if __name__ =="__main__":
    sagemaker_endpoint()
