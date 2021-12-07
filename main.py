import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import h5py


def Ttest250(inputfile, outputfile="Ttest250.trcs"):
    '''
    :param inputfile: Input dataset filename
    :param outputfile: Output dataset filename you want
    The original version of the t-test, which calculates
    the maximum value of the 10*10 sets of 250 data
    '''
    hf = h5py.File(inputfile, 'r')
    xlab = list(hf.keys())
    ylab = list(hf[xlab[0]].keys())

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


def Ttest259(inputfile, outputfile="Ttest259.trcs"):
    '''
    :param inputfile: Input dataset filename
    :param outputfile: Output dataset filename you want
    The original version of the t-test, which calculates
    the maximum value of the 10*10 sets of 250 data
    '''
    hf = h5py.File(inputfile, 'r')
    xlab = list(hf.keys())
    ylab = list(hf[xlab[0]].keys())

    l = []
    m = []
    f = [[] for x in range(10)]
    for i in xlab:
        for j in ylab:
            random_temp = np.zeros((2000,250))
            semifixed_temp = np.zeros((2000,250))
            random_temp[:,0:34] = np.transpose(hf[i][j]['TVLA']['random'][0:34])
            semifixed_temp[:,0:18] = np.transpose(hf[i][j]['TVLA']['random'][35:53])
            random_temp[:,34:58] = np.transpose(hf[i][j]['TVLA']['random'][54:78])
            semifixed_temp[:,18:46] = np.transpose(hf[i][j]['TVLA']['random'][79:107])
            random_temp[:,58:76] = np.transpose(hf[i][j]['TVLA']['random'][108:126])
            semifixed_temp[:,46:68] = np.transpose(hf[i][j]['TVLA']['random'][127:149])
            random_temp[:,76:105] = np.transpose(hf[i][j]['TVLA']['random'][150:179])
            semifixed_temp[:,68:87] = np.transpose(hf[i][j]['TVLA']['random'][180:199])
            random_temp[:,105:150] = np.transpose(hf[i][j]['TVLA']['random'][200:245])
            semifixed_temp[:,87:90] = np.transpose(hf[i][j]['TVLA']['random'][246:249])
            random_temp[:,150:159] = np.transpose(hf[i][j]['TVLA']['random'][250:259])
            random_temp[:,159:180] = np.transpose(hf[i][j]['TVLA']['semifixed'][0:21])
            semifixed_temp[:,90:143] = np.transpose(hf[i][j]['TVLA']['semifixed'][22:75])
            random_temp[:,180:199] = np.transpose(hf[i][j]['TVLA']['semifixed'][76:95])
            semifixed_temp[:,143:180] = np.transpose(hf[i][j]['TVLA']['semifixed'][96:133])
            random_temp[:,199:228] = np.transpose(hf[i][j]['TVLA']['semifixed'][134:163])
            semifixed_temp[:,180:229] = np.transpose(hf[i][j]['TVLA']['semifixed'][164:213])
            random_temp[:,228:250] = np.transpose(hf[i][j]['TVLA']['semifixed'][214:236])
            semifixed_temp[:,229:250] = np.transpose(hf[i][j]['TVLA']['semifixed'][237:258])
            random = random_temp
            semifixed = semifixed_temp
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
    Calculate the Euclidean distance between random and
    semifixed if there is only one file input, and
    finally plot the data
    '''
    hf = h5py.File(inputfile, 'r')
    xlab = list(hf.keys())
    ylab = list(hf[xlab[0]].keys())

    r, s = [], []
    r_list, s_list = [], []
    r_list_list, s_list_list = [], []
    for i in xlab:
        for j in ylab:
            random = np.array(hf[i][j]['TVLA']['random'][:-1])
            semifixed = np.array(hf[i][j]['TVLA']['semifixed'][:-1])

            tmp_r = [sum(x) / len(random[0]) for x in zip(*random)]
            tmp_s = [sum(x) / len(semifixed[0]) for x in zip(*semifixed)]

            for k in range(len(random)):
                r.append(np.linalg.norm(random[k] - tmp_r))
                s.append(np.linalg.norm(semifixed[k] - tmp_s))
            r_list.append(r)
            s_list.append(s)
            r, s = [], []
        r_list_list.append(r_list)
        s_list_list.append(s_list)
        r_list, s_list = [], []
    plt.figure(figsize=(40,40))
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
    for i in range(len(r_list_list)):
        for j in range(len(r_list_list[0])):
            plt.subplot(10, 10, i*10+j+1)
            plt.hist(r_list_list[i][j], **kwargs)
            plt.hist(s_list_list[i][j], **kwargs)
    plt.show()



def TtestMultiFile(inputfile1, inputfile2, flag='R_R'):
    '''
    :param inputfile1:
    :param inputfile2:
    :param flag: 'R_R' means t-test between different datasets random data with random data
                 'R_S' means t-test between different datasets of random data and semifixed data
    '''
    hf1 = h5py.File(inputfile1, 'r')
    xlab1 = list(hf1.keys())
    ylab1 = list(hf1[xlab1[0]].keys())

    hf2 = h5py.File(inputfile2, 'r')
    xlab2 = list(hf2.keys())
    ylab2 = list(hf2[xlab2[0]].keys())

    l1, l2 = [], []
    m1, m2 = [], []
    f1, f2 = [[] for x in range(10)], [[] for x in range(10)]
    for i in range(len(xlab1)):
        for j in range(len(ylab1)):
            random1 = np.array(hf1[xlab1[i]][ylab1[j]]['TVLA']['random'][:-1])
            random2 = np.array(hf2[xlab2[i]][ylab2[j]]['TVLA']['random'][:-1])
            semifixed1 = np.array(hf1[xlab1[i]][ylab1[j]]['TVLA']['semifixed'][:-1])
            semifixed2 = np.array(hf2[xlab2[i]][ylab2[j]]['TVLA']['semifixed'][:-1])
            for k in range(len(random1)):
                if flag == 'R_R':
                    l1.append(scipy.stats.ttest_ind(random1[k], random2[k], equal_var=False)[0])
                    l2.append(scipy.stats.ttest_ind(semifixed1[k], semifixed2[k], equal_var=False)[0])
                elif flag == 'R_S':
                    l1.append(scipy.stats.ttest_ind(random1[k], semifixed2[k], equal_var=False)[0])
                    l2.append(scipy.stats.ttest_ind(semifixed1[k], random2[k], equal_var=False)[0])
            m1.append(max(l1))
            m2.append(max(l2))
            l1 = []
            l2 = []

    for i in range(10):
        f1[i] = m1[10 * i:10 * i + 9]
        f2[i] = m2[10 * i:10 * i + 9]

    n1, n2 = np.array(f1), np.array(f2)
    print(n1, n2)
    plt.subplot(121)
    plt.imshow(n1, cmap="YlGnBu", interpolation='nearest')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(n2, cmap="YlGnBu", interpolation='nearest')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    print("test")
    Ttest259('withtrojan_trojan2_samesemi_manualshuffle_xinpian_sga_250_new1.trcs', 'Ttest259.trcs')
    Ttest259('withtrojan_INACTIVE_trojan2_samesemi_manualshuffle_xinpian_sga_250_new1.trcs', 'Ttest259.trcs')
    Ttest259('withtrojan_INACTIVE_trojan2_semifix1vssemifix2_notrigger_manualshuffle_xinpian_sga_250_new1.trcs', 'Ttest259.trcs')
    # Ttest249('xxx.trcs', 'Ttest249.trcs')
    # ED(inputfile='xxx.trcs')
    # TtestMultiFile(inputfile1='xxx.trcs',
    #                inputfile2='xxx.trcs',
    #                flag='R_S')
