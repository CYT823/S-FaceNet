import os
import cv2

if __name__ == "__main__":
    source_root = "AgeDB_2"
    dest_root = "AgeDB_2_new"

    '''
    依據不同的人，新增資料夾
    '''
    if not os.path.isdir(dest_root): 
        os.mkdir(dest_root)

    last_folder_name = ""
    current_folder_path = ""
    files = os.listdir(source_root)
    files.sort(key= lambda x:int(x[: x.find('_')]))
    for file in files:
        print("Processing... {}".format(file))
        
        folder_name = file.split("_")[1]
        if folder_name != last_folder_name:
            last_folder_name = folder_name
            current_folder_path = os.path.join(dest_root, folder_name)
            if not os.path.isdir(current_folder_path): 
                os.mkdir(current_folder_path)

        try:
            img = cv2.imread(os.path.join(source_root, file))
            cv2.imwrite(os.path.join(current_folder_path, file), img)
        except Exception:
            print("{} does not succeed!".format(file))
            continue