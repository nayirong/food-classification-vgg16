import os, os.path, sys

food_class = []

#dir_path_train = "/Volumes/My Passport for Mac/YiRong/Machine learning/Data Sets/food-101/AIFood/Dataset/train_full"
dir_path_train = "/home/yirongyirongyirong/train_full"
#dl_test_dir = "/Users/YiRong/Desktop/Machine Learning/Food Classification/AIFood/Dataset/download test"

list_subfolders_with_paths = [f.path for f in os.scandir(dir_path_train) if f.is_dir()]
for food_class_path in list_subfolders_with_paths:
	#print(os.path.basename(food_class_path) + " needs " + str(total_image_required-len([1 for x in list(os.scandir(food_class_path)) if x.is_file()])) + " more images")
	#food_class.append(os.path.basename(food_class_path))
    for filename in os.listdir(food_class_path):
        infilename = os.path.join(folder,filename)
        if not os.path.isfile(infilename): continue
        oldbase = os.path.splitext(filename)
        newname = infilename.replace('.jpg', '.jpeg')
        output = os.rename(infilename, newname)
