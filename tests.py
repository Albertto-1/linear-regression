from extract import getDataXandY
import matplotlib.pyplot as plt

data = getDataXandY('carprice.csv')

for col in data.columns:
    # print(col)
    if col != 'price':
        plt.plot(data[col], data['price'], 'ro')
        plt.xlabel(col)
        plt.ylabel('Price')
        plt.show()
