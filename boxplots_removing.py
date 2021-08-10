import numpy as np
import matplotlib.pyplot as plt
import csv


file_name = './results/performance_removing.csv'
case_array = []
accuracy_rate = []
miss_rate = []

with open(file_name, 'r') as csvFile:
    reader = csv.reader(csvFile)
    next(reader)
    for row in reader:
        case_array.append(int(row[0]))
        accuracy_rate.append(float(row[8]))
        miss_rate.append(float(row[9]))
csvFile.close()

case_array = np.array(case_array)
accuracy_rate = np.array(accuracy_rate)
miss_rate = np.array(miss_rate)

find_1 = np.where(case_array == 1)
find_2 = np.where(case_array == 2)
find_3 = np.where(case_array == 3)
find_4 = np.where(case_array == 4)
find_5 = np.where(case_array == 5)
find_6 = np.where(case_array == 6)
find_7 = np.where(case_array == 7)
find_8 = np.where(case_array == 8)
find_9 = np.where(case_array == 9)
find_10 = np.where(case_array == 10)
find_11 = np.where(case_array == 11)


box1 = np.column_stack((accuracy_rate[find_1], accuracy_rate[find_2], accuracy_rate[find_3], accuracy_rate[find_4], accuracy_rate[find_5], accuracy_rate[find_6], accuracy_rate[find_7], accuracy_rate[find_8], accuracy_rate[find_9],accuracy_rate[find_10],accuracy_rate[find_11]))
box2 = np.column_stack((miss_rate[find_1], miss_rate[find_2], miss_rate[find_3], miss_rate[find_4], miss_rate[find_5], miss_rate[find_6], miss_rate[find_7], miss_rate[find_8], miss_rate[find_9], miss_rate[find_10], miss_rate[find_11]))

labelList =['All', 'All-A','All-HD', 'All-AG', 'All-HT','All-M', 'All-W', 'All-SS', 'All-G', 'All-BMI', 'All-RT']

fig, ax = plt.subplots(1, figsize=(9, 5))
bp1 = ax.boxplot(box1, positions=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], widths=0.2)
print(bp1.keys())
plt.setp(bp1['boxes'], color='red')
plt.setp(bp1['fliers'], color='red')
plt.setp(bp1['caps'], color='red')
plt.setp(bp1['medians'], color='red')
plt.setp(bp1['whiskers'], color='red')


ax2 = ax.twinx()
bp2 = ax2.boxplot(box2, positions=[1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3, 10.3, 11.3], widths=0.2)
plt.setp(bp2['boxes'], color='blue')
plt.setp(bp2['fliers'], color='blue')
plt.setp(bp2['caps'], color='blue')
plt.setp(bp2['medians'], color='blue')
plt.setp(bp2['whiskers'], color='blue')


ax.set_xticklabels(labelList)
ax.set_xticks([1.15, 2.15, 3.15, 4.15, 5.15, 6.15, 7.15, 8.15, 9.15, 10.15, 11.15])

ax.set_ylabel('Accuracy rate', fontsize=13, color='r')
ax.set_xlabel('Features combination', fontsize=14)
ax2.set_ylabel('Miss rate', fontsize=13, color='b')

ax.grid(True)
plt.savefig('./results/feature-removal.pdf', format='pdf')
plt.show()