import os

if __name__ == "__main__":
    id_img_file = "./CFP-FP/Protocol/Pair_list_F.txt"
    file_root = "./CFP-FP/Protocol/Split/FF"
    pair_file = "./CFP-FP/Protocol/pairs.txt"

    # 取得編號對應的影像
    with open(id_img_file) as f:
        lines = f.read().splitlines()
    
    img_list = [] # index 0 == number 1
    for line in lines:
        img_list.append(line.split(" ")[1])

    count_same = 0
    count_diff = 0
    # 將每個protocol(same && diff)依據編號，找到對應影像後寫檔
    for subroot in os.listdir(file_root):
        path = os.path.join(file_root, subroot)
        print("Processing...{}".format(path))

        # 這邊是相同人的配對
        with open(os.path.join(path, "same.txt")) as f:
            lines = f.read().splitlines()
        label = 1
        for line in lines:
            img1 = img_list[int(line.split(",")[0])-1]
            img2 = img_list[int(line.split(",")[1])-1]
            count_same += 1
            # WF(append)
            with open(pair_file, 'a') as f:
                f.write(img1 + ' ' + img2 + ' ' + str(label)+'\n')
            f.close()
        
        # 這邊是不同人的配對
        with open(os.path.join(path, "diff.txt")) as f:
            lines = f.read().splitlines()
        label = -1
        for line in lines:
            img1 = img_list[int(line.split(",")[0])-1]
            img2 = img_list[int(line.split(",")[1])-1]
            count_diff += 1
            # WF(append)
            with open(pair_file, 'a') as f:
                f.write(img1 + ' ' + img2 + ' ' + str(label)+'\n')
            f.close()
        
    print("same pairs: {}, diff pairs: {}, total: {}".format(count_same, count_diff, count_same+count_diff))
    print("Process Finished! Save it at {}".format(pair_file))