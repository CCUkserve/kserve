import sys
import requests
import json
import math
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import time
from skimage.color import gray2rgb, label2rgb  # since the code wants color images

print('************************************************************')
print('************************************************************')
print('************************************************************')
print("starting query")

if len(sys.argv) < 3:
    raise Exception("No endpoint specified. ")
endpoint = sys.argv[1]
headers = {
    'Host': sys.argv[2]
}
parameters = {}
test_num = 1002
is_file = False
if len(sys.argv) > 3:
    try:
        test_num = int(sys.argv[3])
    except Exception:  # pylint: disable=broad-except
        is_file = True

    if len(sys.argv) > 4:
        try:
            parameters = json.loads(sys.argv[4])
        except Exception:  # pylint: disable=broad-except
            raise Exception("Failed to convert to json format. ")
if is_file:
    inputs = open(sys.argv[2])
    inputs = json.load(inputs)
    actual = "unk"
else:
    data = fetch_20newsgroups()
    inputs = data.data[test_num]
    labels = data.target[test_num]
 
    actual = data.target_names[labels]
    # to do list is below 
input_text = {"instances": [inputs]}
input_text.update(parameters)
print("Sending Explain Query")

x = time.time()

res = requests.post(endpoint, json=input_image, headers=headers)

print("TIME TAKEN: ", time.time() - x)

print(res)
if not res.ok:
    res.raise_for_status()
res_json = res.json()
temp = np.array(res_json["explanations"]["temp"])
masks = np.array(res_json["explanations"]["masks"])
top_labels = np.array(res_json["explanations"]["top_labels"])

fig, m_axs = plt.subplots(math.ceil(len(top_labels)/5), 5, figsize=(12, 6))
for i, c_ax in enumerate(m_axs.flatten()):
    if i >= len(top_labels):
        c_ax.axis('off')
        continue
    mask = masks[i]
    c_ax.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
    c_ax.set_title('Positive for {}\nActual {}'.format(top_labels[i], actual))
    c_ax.axis('off')
plt.show()
