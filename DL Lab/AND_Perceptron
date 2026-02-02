def perceptron_AND(x1, x2):
    w1, w2 = 1, 1
    b = -1.5
    z = w1*x1 + w2*x2 + b
    return 1 if z >= 0 else 0

inputs = [(0,0), (0,1), (1,0), (1,1)]

for x in inputs:
    print(x, "->", perceptron_AND(x[0], x[1]))