import numpy as np

def f_activation(x):
    return 1/(1 + np.exp(-x)) # гиперболический тангенс

def df_activation(x):
    return f_activation(x) * (1 - f_activation(x))

w1 = np.array([[0.2, 0.4, 0.22],
               [-0.1, 0.6, -0.31]])

w2 = np.array([[0.4, 0.5],
               [0.5, -0.3],
               [0.12, 0.4]]) 

w3 = np.array([0.5, -0.5])
#w4 = np.array([1])
w = np.array([w1, w2, w3])

def f_out(inp):

    sum = []
    f_sum = []
    for i in range(len(w)):
        sum.append(np.dot(inp, w[i]))
        if isinstance(sum[i], float):
            y = f_activation(sum[i])
        else:
            inp = np.array([f_activation(x) for x in sum[i]])
            f_sum.append(inp)
    return sum, y

def train(input, prediction):
    lmd = 0.01
    sum_out, out = f_out(input)

    sum_y = sum_out[-1]
    sum_out = np.delete(sum_out, -1)
    np.insert(sum_out, 0, input)

    y = out[-1]
    out = np.delete(out, -1) 
    
    
    # e = (y - prediction)** 2 - ошибка 
    e = y -prediction
    delta = e * df_activation(sum_y) * 1
    for i_layer in range(len(w), 0, -1): # у нас всего три слоя
        sum_ = sum_out[i_layer]
        
        for i in i_layer:
            shift = delta * df_activation(sum_[i][0])
            w[i_layer][i] = w[i_layer][i] - lmd * shift
        


    

input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
prediction = np.array([[0], [1], [1], [0]])
#train(train_input, train_out)
#train(input, prediction)
#def back_prop()
print(f_out(np.array([0, 0])))




