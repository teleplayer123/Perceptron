from random import random, randrange, seed
import matplotlib.pyplot as plt 

seed(0)

class Perceptron(object):

    def __init__(self, x, y, learn_rate=0.1, epochs=6):
        self.x = x
        self.y = y
        self.learn_rate = learn_rate
        self.epochs = epochs
    
    def init_weights(self):
        self.weights = [[random()] for _ in range(len(self.x[0])+1)]

    def init_bias(self):
        for row in range(len(self.x)):
            b = int(self.x[row][0] & self.x[row][1]) & (self.x[row][0] * self.x[row][1])
            self.x[row].append(b)

    def train(self):
        for _ in range(self.epochs):
            a = self.feed_forward(self.x) 
            e = self.calc_error(a, self.y)
            self.update_weights(transpose(self.x), e)
        return self.weights

    def update_weights(self, x, e):
        res = [[0] for _ in range(len(self.x))]
        for d in range(len(x)):
            for n in range(len(e[0])):
                for m in range(len(x[0])):
                    res[m][n] += x[d][m] * e[m][n]
                    self.weights[d][n] -= self.learn_rate * res[m][n]

    def calc_error(self, a, y):
        e = [[None] for _ in range(len(self.x))]
        for n in range(len(a)):
            for m in range(len(a[0])):
                e[n][m] = a[n][m] - y[n][m]
        return e 
                    
    def feed_forward(self, x):
        a = [[0] for _ in range(len(self.x))] 
        for d in range(len(x)):
            for n in range(len(self.y[0])):
                for m in range(len(x[0])):
                    a[d][n] += self.weights[m][n] * x[d][m]
                if a[d][n] > 0:
                    a[d][n] = 1
                else:
                    a[d][n] = 0
        return a

    def predict(self, x):
        a = [[0] for _ in range(len(x))] 
        for d in range(len(x)):
            for n in range(len(self.y[0])):
                for m in range(len(x[0])):
                    a[d][n] += self.weights[m][n] * x[d][m]
                a[d][n] = int(a[d][n] > 0)
        return a

    def confusion_matrix(self, actual, predicted):
        matrix = {"fired": {}, "not_fired": {}}
        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(len(actual)):
            for j in range(len(actual[0])):
                if actual[i][j] == predicted[i][j] and actual[i][j] == 1:
                    tp += 1
                if actual[i][j] != predicted[i][j] and actual[i][j] == 0:
                    fp += 1
                if actual[i][j] != predicted[i][j] and actual[i][j] == 1:
                    fn += 1
                if actual[i][j] == predicted[i][j] and actual[i][j] == 0:
                    tn += 1
        matrix["fired"]["should_fire"] = tp
        matrix["fired"]["shouldnt_fire"] = fn
        matrix["not_fired"]["should_fire"] = fp
        matrix["not_fired"]["shouldnt_fire"] = tn
        return matrix

    def accuracy(self, matrix):
        tp = matrix["fired"]["should_fire"]
        fn = matrix["fired"]["shouldnt_fire"]
        fp = matrix["not_fired"]["should_fire"]
        tn = matrix["not_fired"]["shouldnt_fire"]
        return ((tp + tn) / (tp + fp + tn + fn)) * 100.0

def transpose(matrix):
    return [[matrix[y][x] for y in range(len(matrix))] for x in range(len(matrix[0]))]


#xor example
x = [[0,0],[0,1],[1,0],[1,1]]
y = [[0], [1], [1], [0]]
p = Perceptron(x, y, learn_rate=0.25, epochs=20000)
p.init_weights()
p.init_bias()
p.train()
res = p.predict(x)
print(res)
m = p.confusion_matrix(y, res)
print(p.accuracy(m)) 

"""
#plot boundary line
x = [[0,0], [0,1], [1,0], [1,1]]
y = [[0], [1], [1], [1]]
p = Perceptron(x, y, learn_rate=0.25, epochs=20000)
p.init_weights()
p.init_bias()
w = p.train()
w1, w2, b = w[0][0], w[1][0], w[2][0]
xint = b/w1
yint = b/w2
xp = [0, xint]
y0 = (-yint/xint) * xp[0] + yint
y1 = (-yint/xint) * xp[1] + yint
yp = [y0, y1]
print(xp, yp)
plt.plot(xp, yp)
plt.scatter([0,0,1,1], [0,1,0,1])
plt.show()
"""
