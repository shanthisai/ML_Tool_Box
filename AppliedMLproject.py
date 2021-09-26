import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def head(data_frame):
    return data_frame.head()


def describe(data_frame):
    return data_frame.describe()


def ploting(data_frame, model):
    if model > 1:
        fig, axes = plt.subplots(figsize=(12, 4), nrows=model // 2, ncols=2)
        colors = ["b", "r", "o", "g"]
        for i in range(model):
            axes[i].scatter(data_frame[i], data_frame[model], color=colors[i % 4])
            axes[i].set_xlabel("X-axis")
            axes[i].set_ylabel("Dependent value")
            axes[i].set_title("Prediction")
    else:
        plt.scatter(data[0], data[model])
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Prediction")
    return plt.show()

def computeCost(X, y, thetaC):
    """
    Take in a numpy array X,y, theta and generate the cost function of using theta as parameter
    in a linear regression model
    """
    m = len(y)
    predictions = X.dot(thetaC)
    square_err = (predictions - y) ** 2

    return 1 / (2 * m) * np.sum(square_err)


def featureNormalization(X):
    """
    Take in numpy array of X values and return normalize X values,
    the mean and standard deviation of each feature
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_norm = (X - mean) / std

    return X_norm, mean, std

def gradientDescent(X, y, thetaG, alpha, num_iters):
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
    with learning rate of alpha

    return theta and the list of the cost of theta during each iteration
    """

    m = len(y)
    J_history = []

    for i in range(num_iters):
        predictions = X.dot(thetaG)
        error = np.dot(X.transpose(), (predictions - y))
        descent = alpha * 1 / m * error
        thetaG -= descent
        J_history.append(computeCost(X, y, thetaG))

    return thetaG, J_history

def procedure(data_frame,learning_rate,epochs,model):
    data_frame = data_frame.sample(frac=1)
    if model == 1:
        dftrain = data_frame[(len(data_frame) // 5):]
        dftrain = dftrain.values
        len_of_traindata = dftrain[:,0].size
        x_training = np.append(np.ones((len_of_traindata, 1)), dftrain[:,0].reshape(len_of_traindata,1), axis=1)
        y_training = dftrain[:,1].reshape(len_of_traindata,1)
        theta = np.zeros((2, 1))
        theta_values, J_history = gradientDescent(x_training, y_training, theta, learning_rate, epochs)
        # print("h(x) ="+str(round(theta_values[0,0],2))+" + "+str(round(theta_values[1,0],2))+"x1")
        return theta_values, J_history, x_training, y_training
    elif model > 1:
        data_n2 = data_frame.values
        m2 = len(data_n2[:, -1])
        x_training = data_n2[:, 0:model].reshape(m2, 2)
        x_training, mean_x_training, std_x_training = featureNormalization(x_training)
        x_training = np.append(np.ones((m2, 1)), x_training, axis=1)
        y_training = data_n2[:, -1].reshape(m2, 1)
        theta = np.zeros((model+1, 1))
        theta_values, J_history = gradientDescent(x_training, y_training, theta, learning_rate, epochs)

        return theta_values, J_history, x_training, y_training

def printHypothesis(theta_values):
    string = "h(x) ="+str(round(theta_values[0,0],2))
    for i in range(1,model+1):
        string += (" + "+str(round(theta_values[i,0],2))+"x"+str(i))
    return string

def plot_gradientdescent(J_history):
    plt.plot(J_history)
    plt.xlabel("Iteration")
    plt.ylabel("$J(\Theta)$")
    plt.title("Cost function using Gradient Descent")
    return plt.show()

def costFunctionVisualisation(X,y):

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array([theta0_vals[i], theta1_vals[j]])
            J_vals[i, j] = computeCost(X, y, t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap="coolwarm")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("$\Theta_0$")
    ax.set_ylabel("$\Theta_1$")
    ax.set_zlabel("$J(\Theta)$")

    # rotate for better angle
    ax.view_init(30, 120)
    return plt.show()

def bestFitline(data_frame,theta_values):
    data_frame = data_frame.sample(frac = 1)
    dfval = data_frame[:(len(data_frame) // 5)]
    dfval_x = dfval.values
    len_of_valdata = dfval_x[:,0].size
    x_val = np.append(np.ones((len_of_valdata, 1)), dfval_x[:,0].reshape(len_of_valdata,1), axis=1)
    y_val = np.dot(x_val, theta_values)
    # print(y_val)
    x_val = x_val[:, 1]
    plt.scatter(dfval[0], dfval[1])
    plt.plot(x_val, y_val, color="r")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Profit Prediction with Validation Set")
    return plt.show()

def predict(x, thetaP):
    """
    Takes in numpy array of x and theta and return the predicted value of y based on theta
    """
    predictionvalues = np.dot(x,thetaP)

    return predictionvalues

if __name__ == '__main__':
    input_file = input("Enter the text file with .txt extension only : >> ")
    data = pd.read_csv(input_file, header=None)
    # Header = None checks whether the columns have headers or not.#
    typeofLearning = input("Choose the type of Learning: 1. Supervised, 2. Unsupervised  : >> ")
    if typeofLearning == "2":
        print("Not implemented.")

    else:
        type = input("Choose type 1. Classification, 2. Regression : >> ")
        if type == "1":
            pass
        else:
            data_frame_len = len(data.columns) # if length is > 2 then model is Multivariate else Univariate #
            model = data_frame_len-1
            if model == 1:
                print("Model : Univariate")
            else:
                print("Model : Multi Variate")

            while(True):
                user_input = input("Choose methods 1.head(), 2.describe(), 3.Plotting of data, 4.Prediction Procedure, 5.Exit : >> ")
                if user_input == "1":
                    print(head(data))
                    continue
                elif user_input == "2":
                    print(describe(data))
                    continue
                elif user_input == "3":
                    ploting(data,model)

                elif user_input == "4":
                    learning_rate = 0.02
                    epochs = 1000
                    parameters_selection = input("Choose Parameters : 1. Yes 2. Use default : >> ")
                    if parameters_selection == "1":
                        print("Default Learning rate = ", str(learning_rate), " , Default epochs(iterations) = ",
                              str(epochs))
                        learning_rate = float(input("Enter Learning rate (preferable b/w 0.001 to 0.02) : >> "))
                        epochs = int(input("Enter number of (epochs)iterations (preferable b/w 1000 to 10000) : >> "))
                        print("Split ratio : 80:20 - 80% is for training and 20% is for validation")
                        theta_values, J_history, X, y = procedure(data, learning_rate, epochs, model)
                    else:
                        print("Default Learning rate = ",str(learning_rate)," , Default epochs(iterations) = ", str(epochs))
                        print("Split ratio : 80:20 - 80% is for training and 20% is for validation")
                        theta_values, J_history, X, y = procedure(data, learning_rate, epochs,model)
                    while(True):
                        if model == 1:
                            get_data = input(
                                "To Print choose 1.Hypothesis, 2.Plot J(Θ) against the number of iteration, 3.Cost function visualisation, 4. Graph with Line of Best Fit, 5. Predict with test data, 6.Exit : >> ")
                        else:
                            get_data = input(
                                "To Print choose A.Hypothesis, B.Plot J(Θ) against the number of iteration, C. Predict with test data, E.Exit : >> ")
                        if get_data == "1" or get_data == "A":
                            print("Hypothesis Function")
                            print(printHypothesis(theta_values))
                        elif get_data == "2" or get_data == "B":
                            plot_gradientdescent(J_history)
                        elif get_data == "3":
                            costFunctionVisualisation(X,y)
                        elif get_data == "4":
                            bestFitline(data,theta_values)
                        elif get_data == "5" or get_data == "C":
                            testfile = input("Enter the test text file with .txt extension only : >> ")
                            test_data = pd.read_csv(testfile, header=None)
                            dftest = test_data.values
                            len_of_testdata = dftest[:, 0].size
                            # print(dftest[:,0].reshape(len_of_testdata,1))
                            if model == 1:
                                test = np.append(np.ones((len_of_testdata, 1)), dftest[:, 0].reshape(len_of_testdata, 1),
                                                 axis=1)
                            else:
                                test = dftest[:, 0:model].reshape(len_of_testdata, 2)
                                test, mean_test, std_test = featureNormalization(test)
                                test = np.append(np.ones((len_of_testdata, 1)), test, axis=1)
                            predict1 = predict(test, theta_values)
                            print(predict1)
                        elif get_data == "6" or get_data == "E":
                            break
                else:
                    break
