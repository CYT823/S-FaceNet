import scipy.io

pair_file = "./AgeDB_2/pairs.txt"
mat_file = scipy.io.loadmat("./AgeDB_2/protocol.mat")
protocols = mat_file.get("splits")

for i, protocol in enumerate(protocols):
    print("Processing...{} folds.".format(i))

    data = protocol[0][0][0]
    namesL = data[0][0]
    namesR = data[0][1]
    labels = data[1][0]

    for i in range(0, len(namesL)):
        img1 = namesL[i][0][0][0][0] + ".jpg"
        img2 = namesR[i][0][0][0][0] + ".jpg"
        
        with open(pair_file, 'a') as f:
            f.write(img1.split("_")[1] + "/" + img1 + ' ' + \
                    img2.split("_")[1] + "/" + img2 + ' ' + \
                    str(labels[i]) + '\n')
        f.close()
