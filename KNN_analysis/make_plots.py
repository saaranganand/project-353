import data_fetching
import numpy as np
import matplotlib.pyplot as plt
import sys

def main():

    data = data_fetching.main()

    plt.figure(figsize=(10, 5))

    plt.plot(data.index.values, data['daily_return'].values, label='Daily Returns')

    plt.legend()

    plt.savefig('plots/daily_returns.png')

    plt.figure(figsize=(10, 5))

    plt.plot(data.index.values, data['50_day_ma'].values, label='50 Day Moving Average')

    plt.legend()

    plt.savefig('plots/50_mv_avg.png')

    return

if __name__ == '__main__':
    sys.exit(main())