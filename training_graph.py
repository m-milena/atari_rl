import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter

log_file_path = './log_training/'
log_files = ['log_part1.txt', 'log_part2.txt', 'log_part3.txt', 'log_part4.txt', 'log_part5.txt']

line_count = 0
epochs = []
score = []
epoch_counter = 0
random_counter = 0
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
            
    line_count = 0
        
# filter data
n = 40
b = [1.0 / n] * n
a = 1
score_filtered = lfilter(b,a,score)

plt.plot(epochs, score, linewidth=1, linestyle="-", c="#fc7b03", alpha=0.4)        
plt.plot(epochs, score_filtered, linewidth=2, linestyle="-", c="#fc7b03") 
plt.title('Score during training')
plt.ylabel('Score')
plt.xlabel('Epochs')
plt.savefig(log_file_path+'training_score.png', dpi=400)
plt.show()
