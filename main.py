import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import h5py


def Ttest250(inputfile, outputfile="Ttest250.trcs"):
    '''
    :param inputfile: Input dataset filename
    :param outputfile: Output dataset filename you want
    The original version of the t-test, which calculates
    the maximum value of the 10*10 sets of 250 data
    '''
    # Test to see if it is welch's t test
    # a = [17.2, 20.9, 22.6, 18.1, 21.7, 21.4, 23.5, 24.2, 14.7, 21.8]
    # b = [21.5, 22.8, 21.0, 23.0, 21.6, 23.6, 22.5, 20.7, 23.4, 21.8, 20.7, 21.7, 21.5, 22.5, 23.6, 21.5, 22.5, 23.5, 21.5, 21.8]
    # c = scipy.stats.ttest_ind(a, b, equal_var=False)

    # read file
    hf = h5py.File(inputfile, 'r')
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
            m.append(round(max(l), 4))
            l = []
    for i in range(10):
        f[i] = m[10 * i:10 * i + 9]

    n = np.array(f)
    plt.imshow(n, cmap="YlGnBu", interpolation='nearest')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.colorbar()
    plt.show()
    wf = h5py.File(outputfile, 'w')
    for i in range(10):
        tg1 = wf.create_group(xlab[i])
        for j in range(10):
            tg2 = tg1.create_group(ylab[j])
            tg2.create_dataset("t", data=m[i * 10 + j])

    wf.close()


def Ttest249(inputfile, outputfile="Ttest249.trcs"):
    '''
    :param inputfile: Input dataset filename
    :param outputfile: Output dataset filename you want
    The modified version of the T-test that calculates
    the maximum value of the 10*10 sets of 249 data
    (with the last set of startup data removed)
    '''
    hf = h5py.File(inputfile, 'r')
    xlab = list(hf.keys())
    ylab = list(hf[xlab[0]].keys())

    l = []
    m = []
    f = [[] for x in range(10)]
    for i in xlab:
        for j in ylab:
            # The last group is the start-up information
            random = np.transpose(hf[i][j]['TVLA']['random'][:-1])
            semifixed = np.transpose(hf[i][j]['TVLA']['semifixed'][:-1])
            for k in range(len(random)):
                l.append(scipy.stats.ttest_ind(random[k], semifixed[k], equal_var=False)[0])
            m.append(round(max(l), 4))
            l = []
    for i in range(10):
        f[i] = m[10 * i:10 * i + 9]

    n = np.array(f)
    plt.imshow(n, cmap="YlGnBu", interpolation='nearest')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.colorbar()
    plt.show()

    wf = h5py.File(outputfile, 'w')
    for i in range(10):
        tg1 = wf.create_group(xlab[i])
        for j in range(10):
            tg2 = tg1.create_group(ylab[j])
            tg2.create_dataset("t", data=m[i * 10 + j])

    wf.close()


def ED(inputfile):
    '''
    :param inputfile: Input dataset filename
    :return: dist_list_list: A 10*10*250 list
    Calculate the Euclidean distance between random and
    semifixed if there is only one file input, and
    finally return a data of 10*10*250
    '''
    hf = h5py.File(inputfile, 'r')
    xlab = list(hf.keys())
    ylab = list(hf[xlab[0]].keys())
    # TVLA = list(hf[xlab[0]][ylab[0]].keys())[0]
    # lab = list(hf[xlab[0]][ylab[0]][TVLA].keys())[1]
    x = len(hf[xlab[0]][ylab[0]]['TVLA']['random'])     # 250
    y = len(hf[xlab[0]][ylab[0]]['TVLA']['random'][0])  # 1000
    # print()
    dist = []
    dist_list = []
    dist_list_list = []
    for i in xlab:
        for j in ylab:
            random = np.array(hf[i][j]['TVLA']['random'][:])
            semifixed = np.array(hf[i][j]['TVLA']['semifixed'][:])
            for k in range(len(random)):
                dist.append(np.linalg.norm(random[k]-semifixed[k]))
            dist_list.append(dist)
            dist = []
        dist_list_list.append(dist_list)
        dist_list = []
    # Final dist_list_list is a 10*10*250 array
    return dist_list_list


def ED2(inputfile1, inputfile2):
    '''
    :param inputfile1: First dataset filename
    :param inputfile2: Second dataset filename
    :return random_list_list: A 10*10*250 data
    :return semifixed_list_list: A 10*10*250 data
    If there are two files input, calculate the Euclidean distance
    between random and semifixed of different files, and
    finally return two 10*10*250 data
    '''
    # First dataset
    hf1 = h5py.File(inputfile1, 'r')
    xlab1 = list(hf1.keys())
    ylab1 = list(hf1[xlab1[0]].keys())
    # TVLA = list(hf1[xlab1[0]][ylab1[0]].keys())[0]
    # lab = list(hf1[xlab1[0]][ylab1[0]][TVLA].keys())[1]
    x1 = len(hf1[xlab1[0]][ylab1[0]]['TVLA']['random'])     # 250
    y1 = len(hf1[xlab1[0]][ylab1[0]]['TVLA']['random'][0])  # 1000 or 2000

    # Seconf dataset
    hf2 = h5py.File(inputfile2, 'r')
    xlab2 = list(hf2.keys())
    ylab2 = list(hf2[xlab2[0]].keys())
    # TVLA = list(hf2[xlab2[0]][ylab2[0]].keys())[0]
    # lab = list(hf2[xlab2[0]][ylab2[0]][TVLA].keys())[1]
    x2 = len(hf2[xlab2[0]][ylab2[0]]['TVLA']['random'])     # 250
    y2 = len(hf2[xlab2[0]][ylab2[0]]['TVLA']['random'][0])  # 1000 or 2000

    random = []
    random_list = []
    random_list_list = []
    semifixed = []
    semifixed_list = []
    semifixed_list_list = []

    for i in range(len(xlab1)):
        for j in range(len(ylab1)):
            random1 = np.array(hf1[xlab1[i]][ylab1[j]]['TVLA']['random'][:])
            random2 = np.array(hf2[xlab2[i]][ylab2[j]]['TVLA']['random'][:])
            semifixed1 = np.array(hf1[xlab1[i]][ylab1[j]]['TVLA']['semifixed'][:])
            semifixed2 = np.array(hf2[xlab2[i]][ylab2[j]]['TVLA']['semifixed'][:])
            for k in range(len(random1)):
                random.append(np.linalg.norm(random1[k]-random2[k]))
                semifixed.append(np.linalg.norm(semifixed1[k]-semifixed2[k]))
            random_list.append(random)
            semifixed_list.append(semifixed)
            random = []
            semifixed = []
        random_list_list.append(random_list)
        semifixed_list_list.append(semifixed_list)
        random_list = []
        semifixed_list = []
    return random_list_list, semifixed_list_list


if __name__ == '__main__':
    # dist = ED(inputfile='SakuraG-A-LUT-10GSs-Trojan_active-new-11-10x10-250.trcs')
    random, semifixed = ED2(inputfile1='SakuraG-A-LUT-10GSs-Trojan_active-new-11-10x10-250.trcs', inputfile2='SakuraG-A-LUT-10GSs-Trojan_active-new-11-10x10-250.trcs')