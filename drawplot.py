import matplotlib.pyplot as plt

CASIA_logging_file = "test_CASIA_logging_0528.txt" # test_CASIA_logging.txt、test_VGG_Face2_logging.txt
LFW = []
CFP_FP = []
AgeDB = []
with open(CASIA_logging_file) as f:
    lines = f.read().splitlines()

for line in lines:
    info = line.split("， ")[1]
    if info.split(" ")[0] == "LFW":
        LFW.append(float(info.split(":")[1][:6]))
    elif info.split(" ")[0] == "CFP_FP":
        CFP_FP.append(float(info.split(":")[1][:6]))
    elif info.split(" ")[0] == "AgeDB":
        AgeDB.append(float(info.split(":")[1][:6]))

# print max accuracy and their iteration
print("LFW\tCFP-FP\tAgeDB")

LFW_max_value = max(LFW)
LFW_max_index = LFW.index(LFW_max_value)
print(LFW_max_value, CFP_FP[LFW_max_index], AgeDB[LFW_max_index], LFW_max_index*3000+3000)

CFP_FP_max_value = max(CFP_FP)
CFP_FP_max_index = CFP_FP.index(CFP_FP_max_value)
print(LFW[CFP_FP_max_index], CFP_FP_max_value, AgeDB[CFP_FP_max_index], CFP_FP_max_index*3000+3000)

AgeDB_max_value = max(AgeDB)
AgeDB_max_index = AgeDB.index(AgeDB_max_value)
print(LFW[AgeDB_max_index], CFP_FP[AgeDB_max_index], AgeDB_max_value, AgeDB_max_index*3000+3000)

sum_list = [a + b + c for a, b, c in zip(LFW, CFP_FP, AgeDB)]
all_max_value = max(sum_list)
all_max_index = sum_list.index(all_max_value)
print("The best model for all three benchmark datasets: LFW: {}, CFP_FP: {}, AgeDB: {}, iter_{}".format(LFW[all_max_index], CFP_FP[all_max_index], AgeDB[all_max_index], all_max_index*3000+3000))


# x軸
x = list(range(int(len(lines)/3)))

# plotting the points
plt.plot(x, LFW, 'r-', label='LFW')
plt.plot(x, CFP_FP, 'g-', label='CFP_FP')
plt.plot(x, AgeDB, 'b-', label='AgeDB')

plt.xlabel('')
plt.ylabel('Acc')
plt.title('Evaluation and Benchmarking')
 
# function to show the plot
plt.show()