from PIL import Image
import os
import warnings
#warnings.filterwarnings('always')

invalid = []
labels = os.listdir('train_full')
for label in labels:
    if label[0] == '.':
        continue
    files = os.listdir('train_full/' + label)
    counter = 0
    with warnings.catch_warnings(record=True) as w:
        current_len_w = 0
        for file in files:
            try:
                im = Image.open('train_full/' + label + '/' + file)
                im.verify()
                im.close()
                counter += 1
            except Warning:
                invalid.append('train_full/' + label + '/' + file)
            if len(w) != current_len_w:
                invalid.append('train_full/' + label + '/' + file)
                current_len_w = len(w)
    print(label + ' - ' + str(counter) + '/' + str(len(files)) + ' valid')

print(invalid)

for invalid_path in invalid:
    os.remove(invalid_path)

