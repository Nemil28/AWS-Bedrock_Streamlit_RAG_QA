import boto3
import json
import base64
import os

prompt_data = """
Give me an image of a developer working on debugging a problem
with his headphones on very late at night. make it look realistic.
"""

prompt_template = [{'text': prompt_data, 'weight': 1}]

bedrock = boto3.client(service_name = 'bedrock-runtime')

payload = {
    'text_prompts': prompt_template,
    'cfg_scale': 10,
    'seed': 0,
    'steps': 50,
    'width': 512,
    'height': 512
}

body = json.dumps(payload)

model_id = 'stability.stable-diffusion-xl-v0'

response = bedrock.invoke_model(
    body = body,
    modelId = model_id,
    accept = 'application/json',
    contentType = 'application/json'
)

response_body = json.loads(response.get('body').read())
print(response_body)

artifact = response_body.get('artifacts')[0]
image_encoded = artifact.get('base64').encode('utf-8')
image_bytes = base64.b64decode(image_encoded)

# Save image to a directory

output_dir = 'generated_image'
os.makedirs(output_dir, exist_ok = True)

file_name = f"{output_dir}/generated-img.png"
with open(file_name, 'wb') as f:
    f.write(image_bytes)