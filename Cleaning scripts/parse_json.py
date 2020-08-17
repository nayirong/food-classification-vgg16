import json
import os
import matplotlib.pyplot as plt

# list of desired classes
desired_classes = []
with open('labels.txt','r') as infile:
    for line in infile:
        desired_classes.append(line.strip())

# create train, val, test directories
for directory in ['train', 'val', 'test']:
    if not os.path.exists(directory):
        os.makedirs(directory)

for directory in ['all_train', 'all_val', 'all_test']:

    new_dir = directory.replace('all_','')
    with open(directory + '/annotations.json','r') as infile:
        file_str = infile.read().strip()
        file_dict = json.loads(file_str)

    # create dictionary for mapping of class id to class name
    class_id_mapping = {}
    for i in file_dict['categories']:
        if i['name_readable'] in desired_classes:
            class_id_mapping[i['id']] = i['name_readable']
    assert len(class_id_mapping) == len(desired_classes)

    # ignore images with multiple labels
    single_label_images = set(i['image_id'] for i in file_dict['annotations'])
    assert len(file_dict['images']) == len(single_label_images)

    # create dictionary for mapping of image id to image name
    image_id_mapping = {}
    for i in file_dict['images']:
        if i['id'] in single_label_images:
            image_id_mapping[i['id']] = i['file_name']

    # create dictionary for mapping of image name to bounding box
    bbox_mapping = {}
    for i in file_dict['annotations']:
        if i['image_id'] in single_label_images:
            bbox_mapping[image_id_mapping[i['image_id']]] = i['bbox']

    label_images_dict = {}
    for label in class_id_mapping.values():
        label_images_dict[label] = []

    for annotation in file_dict['annotations']:
        if annotation['image_id'] in single_label_images and annotation['category_id'] in class_id_mapping:
            class_name = class_id_mapping[annotation['category_id']]
            file_name = image_id_mapping[annotation['image_id']]
            label_images_dict[class_name].append(file_name)

    # write to new folders
    for label in label_images_dict:
        if not os.path.exists(new_dir + '/' + label):
            os.makedirs(new_dir + '/' + label)
        for image_file_name in label_images_dict[label]:
            image = plt.imread(directory + '/images/' + image_file_name)
            dims = bbox_mapping[image_file_name]
            cropped_image = image[int(dims[0]):int(dims[0]+dims[2]),
                                  int(dims[1]):int(dims[1]+dims[3]),
                                  :]
            plt.imsave(new_dir + '/' + label + '/' + image_file_name.replace('jpg','png'), cropped_image)
            plt.close()
        print(directory + ': ' + str(len(label_images_dict[label])) + ' images written for class ' + label)
        
    print('')
