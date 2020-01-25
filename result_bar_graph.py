import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height-0.5),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

# Training data
log_file_path = './log_training/'
log_files = ['log_part1.txt', 'log_part2.txt', 'log_part3.txt', 'log_part4.txt', 'log_part5.txt']

line_count = 0
epochs = []
score = []
epoch_counter = 0
random_counter = 0
max_train_score = []
for log in log_files:
    random_counter = 0
    with open(log_file_path+log) as file:
        line = file.readline()
        while line:
            line = line.replace("\n","")
            fields = line.split(',')
            if line_count and random_counter == 319:
                epochs.append(epoch_counter)
                score.append(int(float(fields[1])))
                epoch_counter += 1
            else:
                line_count += 1
                random_counter += 1
            line = file.readline()
    max_train_score.append(max(score))    
    line_count = 0

print(max_train_score)

# Testing data
training_hours = [5, 7, 18, 22, 37]
folders_names = ['after_5h/', 'after_7h/', 'after_18h/', 'after_22h/', 'after_37h/']

test_score = []
max_test_score = []
for folder in folders_names:
    with open('./agent_progress/'+folder+'log_test.txt') as file:
        line = file.readline()
        while line:
            line = line.replace("\n","")
            fields = line.split(',')
            if line_count:
                test_score.append(int(float(fields[1])))
            else:
                line_count += 1
            line = file.readline()
    max_test_score.append(max(test_score))    
    line_count = 0
    
print(max_test_score)

# Graph
labels = ['5h', '7h', '18h', '22h', '37h']
x = np.arange(len(labels))
width = 0.35
fix, ax = plt.subplots()
rects1 = ax.bar(x-width/2, max_train_score, width, label='Train', color='k', alpha=0.8)
rects2 = ax.bar(x+width/2, max_test_score, width, label='Test', color='#fc7b03', alpha=0.9)

ax.set_ylabel('Scores')
ax.set_title('Max games scores after hours of training')
ax.set_xticks(x)
ax.set_xticklabels(labels)
autolabel(rects1)
autolabel(rects2)
ax.legend()
plt.savefig(log_file_path+'score_compare.png', dpi=400)
plt.show()

