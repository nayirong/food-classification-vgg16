# import json

# f = open('/Users/YiRong/Downloads/train/annotations.json')

# data = json.loads(f.read())

# for i in data["categories"]:
# 	print(i)

# print(data["images"][0])

# f.close()

import json

f = ('train/annotations.json')
label_files = open('labels.txt', 'w')

with open(f,'r') as infile:
    file_str = infile.read().strip()
    file_dict = json.loads(file_str)

print(file_dict.keys())
classes = [i['name_readable'] for i in file_dict['categories']]
idNum = [i['id'] for i in file_dict['categories']]
#images = [i['']]
i=0

for item in classes:
 	print(item + " have id num: " + str(idNum[i]))
 	i+=1

# i = 0
# for items in classes:
# 	i+=1
# 	n = label_files.write(str(i) + "." + " " + items + "\n")


