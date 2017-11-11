import numpy as np
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
names = ['seismic', 'seismoacoustic', 'shift', 'genergy', 'gpuls', 'gdenergy', 'gdpuls', 'ghazard', 'nbumps', 'nbumps2',
         'nbumps3', 'nbumps4', 'nbumps5', 'nbumps6', 'nbumps7', 'nbumps89', 'energy', 'maxenergy', 'class']
df = pd.read_csv("dataset.csv", names=names)
#df = df.sample(frac=1)


x = df.as_matrix()
y = x[:, -1]
z = np.array(y)
cols = list(df.loc[:, 'seismic':'shift']) + ['ghazard']
dis = df[cols]


def preprocess(data, z):
    data1 = data.as_matrix()
    a = data1[:, 0]
    b = data1[:, 1]
    c = data1[:, 2]
    d = data1[:, 3]
    dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'N': 4, 'W': 5}
    for x in range(2584):
        a[x] = dict[a[x]]
        b[x] = dict[b[x]]
        c[x] = dict[c[x]]
        d[x] = dict[d[x]]
    e = np.column_stack((a, b, c, d, z))
    e = np.array(e)
    return (e)


def earthquake():
    discrete = preprocess(dis, z)
    continous = df.drop(['seismic', 'seismoacoustic', 'shift', 'ghazard', 'nbumps5', 'nbumps6', 'nbumps7', 'nbumps89'], axis=1)
    #continous = df.drop(['seismic', 'seismoacoustic', 'shift', 'ghazard'], axis=1)
    return continous, discrete


def gaussian(test, meansy, meansn, vary, varn):
    py = pn = 1.0
    for i in range(meansy.shape[0] - 1):
        x = float(math.pow((test[i] - meansy[i]), 2)) * (-1)
        y = float(math.pow((test[i] - meansn[i]), 2)) * (-1)
        g = 2 * vary[i]
        h = 2 * varn[i]
        try:
            u = float(x / (1.0 * g))
            v = float(y / (1.0 * h))
        except:
            print("error1")
        try:
            denom_yes = float(math.sqrt(2 * math.pi) * math.sqrt(vary[i]))
            denom_no = float(math.sqrt(2 * math.pi) * math.sqrt(varn[i]))
            exp_yes = math.exp(u)
            exp_no = math.exp(v)
            ans_yes = float(exp_yes / (1.0 * denom_yes))
            ans_no = float(exp_no / (1.0 * denom_no))
        except :
            print("error2")
        py *= ans_yes
        pn *= ans_no
    return (py, pn)

def normalnb(data, test):
    p = []
    classyes = data[data[:, -1] == 1]
    classno = data[data[:, -1] == 0]

    a=classyes.shape[0]
    b=classno.shape[0]
    a1_haz = classyes[classyes[:, 0] == 0]  # first column is zero and class hazard
    a1_nothaz = classno[classno[:, 0] == 0]  # first column is one and class not hazard
    b1_haz = classyes[classyes[:, 0] == 1]
    b1_nothaz = classno[classno[:, 0] == 1]
    c1_haz = classyes[classyes[:, 0] == 2]
    c1_nothaz = classno[classno[:, 0] == 2]
    d1_haz = classyes[classyes[:, 0] == 3]
    d1_nothaz = classno[classno[:, 0] == 3]

    a2_haz = classyes[classyes[:, 1] == 0]  # second column is zero and class hazard
    a2_nothaz = classno[classno[:, 1] == 0]  # second column is one and class not hazard
    b2_haz = classyes[classyes[:, 1] == 1]
    b2_nothaz = classno[classno[:, 1] == 1]
    c2_haz = classyes[classyes[:, 1] == 2]
    c2_nothaz = classno[classno[:, 1] == 2]
    d2_haz = classyes[classyes[:, 1] == 3]
    d2_nothaz = classno[classno[:, 1] == 3]

    n_haz = classyes[classyes[:, 2] == 4]
    n_nothaz = classno[classno[:, 2] == 4]
    w_haz = classyes[classyes[:, 2] == 5]
    w_nothaz = classno[classno[:, 2] == 5]

    a4_haz = classyes[classyes[:, 3] == 0]  # fourth column is zero and class hazard
    a4_nothaz = classno[classno[:, 3] == 0]  # fourth column is one and class not hazard
    b4_haz = classyes[classyes[:, 3] == 1]
    b4_nothaz = classno[classno[:, 3] == 1]
    c4_haz = classyes[classyes[:, 3] == 2]
    c4_nothaz = classno[classno[:, 3] == 2]
    d4_haz = classyes[classyes[:, 3] == 3]
    d4_nothaz = classno[classno[:, 3] == 3]

    col1 = [[a1_haz.shape[0] / a, a1_nothaz.shape[0] / b], [b1_haz.shape[0] / a, b1_nothaz.shape[0] / b],
            [c1_haz.shape[0] / a, c1_nothaz.shape[0] / b], [d1_haz.shape[0] / a, d1_nothaz.shape[0] / b]]
    col2 = [[a2_haz.shape[0] / a, a2_nothaz.shape[0] / b], [b2_haz.shape[0] / a, b2_nothaz.shape[0] / b],
            [c2_haz.shape[0] / a, c2_nothaz.shape[0] / b], [d2_haz.shape[0] / a, d2_nothaz.shape[0] / b]]
    col3 = [[n_haz.shape[0] / a, n_nothaz.shape[0] / b], [w_haz.shape[0] / a, w_nothaz.shape[0] / b]]
    col4 = [[a4_haz.shape[0] / a, a4_nothaz.shape[0] / b], [b4_haz.shape[0] / a, b4_nothaz.shape[0] / b],
            [c4_haz.shape[0] / a, c4_nothaz.shape[0] / b], [d4_haz.shape[0] / a, d4_nothaz.shape[0] / b]]

    for i in range(test.shape[0]):
        py = col1[test[i][0]][0] * col2[test[i][1]][0] * col3[test[i][2] - 4][0] * col4[test[i][3]][0]
        pn = col1[test[i][0]][1] * col2[test[i][1]][1] * col3[test[i][2] - 4][1] * col4[test[i][3]][1]
        p.append([py, pn])
    p = np.array(p)
    return p

def gaussnb(data, test):
    p = []
    rows, cols = data.shape
    classyes = data[data[:, -1] == 1]
    classno = data[data[:, -1] == 0]
    meansy = classyes.sum(axis=0)
    meansn = classno.sum(axis=0)
    meansy = meansy / classyes.shape[0]
    meansn = meansn / classno.shape[0]
    vary = [0.0] * (cols)
    varn = [0.0] * (cols)
    for i in range(cols):
        for j in range(classyes.shape[0]):
            vary[i] += math.pow((classyes[j][i] - meansy[i]), 2)
        for k in range(classno.shape[0]):
            varn[i] += math.pow((classno[j][i] - meansn[i]), 2)
        if (vary[i] == 0):
            vary[i] = 1.0
        if (varn[i] == 0):
            varn[i] = 1.0
    vary = np.array(vary) / (classyes.shape[0] - 1)
    varn = np.array(varn) / (classno.shape[0] - 1)
    for i in range(test.shape[0]):
        py, pn = gaussian(test[i], meansy, meansn, vary, varn)
        p.append([py, pn])
    p = np.array(p)
    return (p)


def get_kfold(dataset):
    target=dataset[:,-1]
    attr=dataset[:,:-1]
    X_train, X_test, y_train, y_test = train_test_split(attr, target, test_size = 0.40)
    train = np.column_stack((X_train,y_train))
    test = np.column_stack((X_test,y_test))
    return train,test

def neg_handle(p):
    meany = meann = 0
    for i in range(p.shape[0]):
        if ((not p[i][0]) and (i != 0)):
            for j in range(i):
                meany = meany + p[j][0]
            meany = float(meany / (i + 1))
            p[i][0] = meany
        if ((not p[i][1]) and (i != 0)):
            for j in range(i):
                meann = meann + p[j][1]
            meann = float(meann / (i + 1))
            p[i][1] = meann
        if ((not p[i][1]) or (not p[i][1]) and (i == 0)):
            print ("hello")
    return p

def check_zeroprob(p, traindis, testdis, traincont, testcont):
    if (p[0][0] == 0 or p[0][1]):
        for j in range(traindis.shape[1]):
            traindis[0][j] += 1
        for j in range(traincont.shape[1]):
            traincont[0][j] += 1
    p_dis = normalnb(traindis, testdis)
    p_cont = gaussnb(traincont, testcont)
    p = p_dis * p_cont
    return (p)


# Main outer code starts here
continuos, discrete = earthquake()
discretedf = pd.DataFrame(discrete)
discretedf = discretedf.as_matrix()
continuousdf = pd.DataFrame(continuos)
continuousdf = continuousdf.as_matrix()
len = (df.as_matrix()).shape[0]
correct = 0
sum = 0.0
false_positive = 0
false_negative = 0
true_positive = 0
true_negative = 0

for j in range(10):
    traindis, testdis = get_kfold(discretedf)
    traincont, testcont = get_kfold(continuousdf)
    p_dis = normalnb(np.array(traindis), np.array(testdis))
    p_cont = gaussnb(np.array(traincont), np.array(testcont))
    p = p_dis * p_cont
    x = np.array(testdis)
    length = np.array(testdis).shape[0]
    #p = check_zeroprob(p, np.array(traindis), np.array(testdis), np.array(traincont), np.array(testcont))
    p = neg_handle(p)
    for i in range(np.array(testdis).shape[0]):
        if (p[i][0] > p[i][1]):
            pred = 1
        else:
            pred = 0
        if (pred == x[i][-1]):
            correct += 1
    correct = correct / length
    print(correct*100)
    sum = sum + (correct*100)
    correct = 0

print ("average is ",sum/10)