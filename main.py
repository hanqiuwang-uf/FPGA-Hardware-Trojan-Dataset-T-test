import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import h5py

if __name__ == '__main__':
    # Test to see if it is welch's t test
    # a = [17.2, 20.9, 22.6, 18.1, 21.7, 21.4, 23.5, 24.2, 14.7, 21.8]
    # b = [21.5, 22.8, 21.0, 23.0, 21.6, 23.6, 22.5, 20.7, 23.4, 21.8, 20.7, 21.7, 21.5, 22.5, 23.6, 21.5, 22.5, 23.5, 21.5, 21.8]
    # c = scipy.stats.ttest_ind(a, b, equal_var=False)

    # read file
    hf = h5py.File('SakuraG-A-LUT-10GSs-Trojan_active-new-11-10x10-250.trcs', 'r')

    # index
    xlab = list(hf.keys())
    # Since the subscripts under all secondary arrays are the same, just use the first one
    ylab = list(hf[xlab[0]].keys())
    k = 0
    l = []
    m = []
    f = [[] for x in range(10)]
    for i in xlab:
        for j in ylab:
            random = np.transpose(hf[i][j]['TVLA']['random'][:])
            semifixed = np.transpose(hf[i][j]['TVLA']['semifixed'][:])
            for k in range(len(random)):
                l.append(scipy.stats.ttest_ind(random[k], semifixed[k], equal_var=False)[0])
            m.append(round(max(l),4))
            l = []
    for i in range(10):
        f[i] = m[10*i:10*i+9]

    n = np.array(f)
    plt.imshow(n, cmap="YlGnBu", interpolation='nearest')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.colorbar()
    plt.show()
    wf = h5py.File('test.trcs', 'w')
    for i in range(10):
        tg1 = wf.create_group(xlab[i])
        for j in range(10):
            tg2 = tg1.create_group(ylab[j])
            tg2.create_dataset("t", data=m[i*10+j])

    wf.close()

