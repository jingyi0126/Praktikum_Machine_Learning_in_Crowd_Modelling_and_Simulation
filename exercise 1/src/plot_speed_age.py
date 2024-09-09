import matplotlib.pyplot as plt
import pandas as pd

'''extract the samples (age, speed, ID)'''

df1 = pd.read_csv('../outputs/samples.csv')
df1['speed'] = df1['speed']
'''extract the measured speed in test4 (ID, speed)'''
df2 = pd.read_csv('../outputs/travel_time_test4.csv')
df2.drop_duplicates(subset=['ID'], keep='first')
# print(df1)
# print(df2)
'''match the measured speed in df2 with the age recording in df1 based on identical ID'''
df3 = df1.filter(['age'], axis=1)
df3['travel_time'] = df2['travel_time']
df3['measured_speed'] = 50 / df3['travel_time']
# print(df3)


# Create a scatter plot
plt.scatter(df3['age'], df3['measured_speed'], color='b', marker='o', label='measured speeds (m/s)')
plt.scatter(df1['age'], df1['speed'], color='r', marker='+', label='assigned speeds (m/s) from rimea_7_speeds.csv')
plt.scatter(df1['age'], df1['speed'] - 0.33, color='g', marker='+', label='assigned speeds (m/s) with an offset of -0.33')
# Customize the plot (add labels and title)
plt.xlabel('Age')
plt.ylabel('Speed (m/s)')
plt.title('Scatter Plot: Speeds w.r.t Age in Test 4')
plt.grid(True)
plt.legend()
# plt.show()
plt.savefig('../outputs/measured_speed_age.png', dpi=300)