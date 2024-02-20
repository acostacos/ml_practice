def train_model(training_set, init_w, init_b, alpha=0.01, iterations=5000):
    w = init_w
    b = init_b

    print('Starting gradient descent')
    for i in range(iterations):
        # print(f'w: {w}')
        # print(f'b: {b}')
        print(f'J(w,b): {cost_func_for_debug(training_set, w, b)}')

        new_w = w - alpha * cost_func_w(training_set, w, b)
        new_b = b - alpha * cost_func_b(training_set, w, b)

        w = new_w
        b = new_b

        if (i != 0 and i % 500 == 0):
            print(f'Current iteration: {i}...')

    return w, b


# Cost Function - must converge during Gradient Descent
def cost_func_for_debug(training_set, w, b):
    m = len(training_set)
    total_err = 0
    for _, row in training_set.iterrows():
        x = row['x']
        y = row['y']
        total_err += ((w * x + b) - y) ** 2

    return total_err / m


# Partial Derivative with Respect to w of Square Error Cost Function
def cost_func_w(training_set, w, b):
    m = len(training_set)
    total_err = 0
    for _, row in training_set.iterrows():
        x = row['x']
        y = row['y']
        total_err += ((w * x) - b - y) * x

    return total_err / m


# Partial Derivative with Respect to b of Square Error Cost Function
def cost_func_b(training_set, w, b):
    m = len(training_set)
    total_err = 0
    for _, row in training_set.iterrows():
        x = row['x']
        y = row['y']
        total_err += (w * x) - b - y

    return total_err / m
