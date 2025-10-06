import numpy as np
import matplotlib.pyplot as plt


class linearna_vrstva:
    def __init__(self, x, y):           # x-pocet vstupov; y-pocet vystupov
        self.w = np.random.randn(x, y)
        self.b = np.random.randn(1, y)
        self.input = None
        if moment:
            self.moment = moment_rate
            self.vW = np.zeros_like(self.w)
            self.vB = np.zeros_like(self.b)


    def dopredu(self, i):
        self.input = i
        return np.dot(i,self.w) + self.b

    def dozadu(self, g, learning_rate=0.1):
        gw = np.dot(self.input.T, g)
        gb = np.sum(g, axis=0, keepdims=True)

        # aktualizacia
        if moment:
            self.vW = self.moment * self.vW + (1 - self.moment) * gw
            self.vB = self.moment * self.vB + (1 - self.moment) * gb
            self.w -= learning_rate * self.vW
            self.b -= learning_rate * self.vB
        else:
            self.w -= learning_rate * gw
            self.b -= learning_rate * gb
        return np.dot(g, self.w.T)

class sigmoid:
    def __init__(self):
        self.output = None

    def dopredu(self, v):
        self.output = 1/(1+np.exp(-v))
        return self.output

    def dozadu(self, g):
        return g * self.output * (1-self.output)

class relu:
    def __init__(self):
        self.input = None

    def dopredu(self, v):
        self.input = v
        return np.maximum(0, v)

    def dozadu(self, g):
        return g * self.input * (self.input > 0)      #derivacia

class tanh:
    def __init__(self):
        self.output = None

    def dopredu(self, v):
        self.output = np.tanh(v)
        return self.output

    def dozadu(self, g):
        return g * (1 - self.output ** 2)

class model:
    def __init__(self):
        self.prva_vrstva = linearna_vrstva(2, 4)
        self.prva_aktivacia = tanh()
        #self.druha_vrstva = linearna_vrstva(4, 4)
        #self.druha_aktivacia = relu()
        self.output_vrstva = linearna_vrstva(4, 1)
        self.output_aktivacia = sigmoid()

    def dopredu(self, input):
        n1 = self.prva_vrstva.dopredu(input)
        n2 = self.prva_aktivacia.dopredu(n1)
        #n3 = self.druha_vrstva.dopredu(n2)
        #n4 = self.druha_aktivacia.dopredu(n3)
        o1 = self.output_vrstva.dopredu(n2)     #n4
        o2 = self.output_aktivacia.dopredu(o1)
        return o2

    def dozadu(self, grad):
        o1 = self.output_aktivacia.dozadu(grad)
        o2 = self.output_vrstva.dozadu(o1)
        #s3 = self.druha_aktivacia.dozadu(o2)
        #s2 = self.druha_vrstva.dozadu(s3)
        s1 = self.prva_aktivacia.dozadu(o2)     #s2
        self.prva_vrstva.dozadu(s1)

def spustenie():
    training_errors = []
    p = 0
    chyba = 0
    for i in range(2000):
        if p == 4:
            p = 0

        a = model.dopredu(input[p:p + 1])

        # chyba
        if i % 4 == 0:
            chyba = np.mean((a - good[p]) ** 2)
            training_errors.append(chyba)
        if i % 100 == 0 and i != 0:
            chyba = np.mean((a - good[p]) ** 2)
            print(f"Chyba po {int(i / 4)} epochách:", chyba)

        grad = 2 * (a - good[p])

        model.dozadu(grad)
        p += 1
    c = 0

    print("")
    vysledok = model.dopredu([0, 0])
    print(f"Vstup: 0,0\tVýstup: {np.round(vysledok).astype(int)}")
    if np.round(vysledok).astype(int) == 0:
        c += 1

    vysledok = model.dopredu([0, 1])
    print(f"Vstup: 0,1\tVýstup: {np.round(vysledok).astype(int)}")
    if np.round(vysledok).astype(int) == 1:
        c += 1

    vysledok = model.dopredu([1, 0])
    print(f"Vstup: 1,0\tVýstup: {np.round(vysledok).astype(int)}")
    if np.round(vysledok).astype(int) == 1:
        c += 1

    vysledok = model.dopredu([1, 1])
    print(f"Vstup: 1,1\tVýstup: {np.round(vysledok).astype(int)}")
    if np.round(vysledok).astype(int) == 0:
        c += 1

    print(f"Správnosť výsledku: {c / 4 * 100}%")

    plt.plot(training_errors)
    plt.title('Priebeh tréningu')
    plt.xlabel('Iterácie')
    plt.ylabel('Chyba')
    plt.grid(True)
    plt.show()




input = np.array([[0,0],[0,1],[1,0],[1,1]])
good = np.array([[0],[1],[1],[0]])

print("------------------- S momentom -------------------")
moment = True
moment_rate = 0.9
model = model()
spustenie()
print("")

print("------------------ Bez momentu ------------------")
moment = False
spustenie()