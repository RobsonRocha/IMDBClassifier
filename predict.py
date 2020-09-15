import boto3
import os
import json


def lambda_handler(event, context):

    runtime_client = boto3.Session().client('sagemaker-runtime')

    payload = event['body']

    # If the API is called by other program, like postman, the variable event has no some attributes.
    try:
        s = event['headers']['content-type']
        if s.find("text/plain") != -1:
            payload = '{"text":"'+payload+'"}'
    except Exception as e:
        print("Error: {}".format(e))


    response = runtime_client.invoke_endpoint(EndpointName=os.environ.get("ENDPOINT_NAME"),
                                       ContentType='application/json',
                                       Body=payload)

    result = response['Body'].read()

    jsonStr = result.decode('utf8').replace("'", '"')

    print(jsonStr)

    result = {
        "statusCode" : 200,
        "headers" : { "Content-Type" : "application/json", "Access-Control-Allow-Origin" : "*" },
        "body" : jsonStr
    }

    print(result)

    return result
