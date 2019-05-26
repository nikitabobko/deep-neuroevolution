import matplotlib.pyplot as plt
import numpy as np

############
### Time ###
############
y = [10.8, 23.6, 46.1, 71.5, 100.3, 114.0, 127.8, 157.5, 188.3, 218.3, 245.0, 268.3]
x = np.arange(len(y) + 1)[1:]
plt.bar(x, y, color='red')
for i in range(len(x)):
    plt.text(i + 1 - 0.21*len(str(y[i]))/2, y[i] + 0.5, str(y[i]))

y = [8.1, 14, 21.3, 30, 40.2, 52.2, 65.7, 80.5, 97.5, 116.3, 137.7, 165.7]
x = np.arange(len(y) + 1)[1:]
plt.bar(x, y, color='green')
for i in range(len(x)):
    if i == 0:
        plt.text(i + 1 - 0.21*len(str(y[i]))/2, y[i] - 6, str(y[i]))
        continue
    plt.text(i + 1 - 0.21*len(str(y[i]))/2, y[i] + 0.5, str(y[i]))

plt.legend(["Uber AI labs ES", "Мой DE"])
plt.xticks(x, x)
plt.ylabel("Время в минутах с начала, до момента,\n когда закончился расчет i-ого поколения")
plt.xlabel("Поколение")
plt.savefig('my_time_plot.png')

plt.savefig('time_plot.png')
plt.close()

##############
### Scores ###
##############
y = [270, 270, 980, 980, 980, 980, 980, 980, 980, 980, 980, 980]
x = np.arange(len(y) + 1)[1:]
plt.plot(x, y, color='green')

y = [40, 160, 220, 170, 220, 220, 120, 120, 150, 200, 200, 200]
plt.plot(x, y, color='red')

plt.legend(["Мой DE", "Uber AI labs ES"])

plt.xticks(x, x)
plt.ylabel("Число очков, которое набирает лучший в поколении")
plt.xlabel("Поколение")
plt.savefig('score_plot.png')
plt.close()


