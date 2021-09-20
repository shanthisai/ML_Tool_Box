import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def head(data_frame):
    return data_frame.head()


def describe(data_frame):
    return data_frame.describe()


def ploting(data_frame):
    plt.scatter(data_frame[0], data_frame[1])
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

def procedure(data_frame,learning_rate,epochs):
    data_frame = data_frame.sample()
    dftrain = data_frame[(len(data_frame) // 5):]
    dftrain = dftrain.values

    len_of_traindata = dftrain[:,0].size
    x_training = np.append(np.ones((len_of_traindata, 1)), dftrain[:,0].reshape(len_of_traindata,1), axis=1)
    y_training = dftrain[:,1].reshape(len_of_traindata,1)
    theta = np.zeros((2, 1))

    theta_values, J_history = gradientDescent(x_training, y_training, theta, learning_rate, epochs)
    # print("h(x) ="+str(round(theta_values[0,0],2))+" + "+str(round(theta_values[1,0],2))+"x1")
    return theta_values, J_history, x_training, y_training

def printHypothesis(theta_values):
    string = ("h(x) ="+str(round(theta_values[0,0],2))+" + "+str(round(theta_values[1,0],2))+"x1")
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
    data_frame = data_frame.sample()
    dfval = data_frame[:(len(data_frame) // 5)]
    x_val = dfval[0]
    print(x_val.values)

    # plt.scatter(dfval[0], dfval[1])
    # plt.plot(x_val, y_val, color="r")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.title("Profit Prediction")
    # return plt.show()

def predict(x, thetaP):
    """
    Takes in numpy array of x and theta and return the predicted value of y based on theta
    """
    predictionvalues = np.dot(thetaP, x)

    return predictionvalues

if __name__ == '__main__':
    input_file = input("Enter the text file with .txt extension only : >> ")
    data=pd.read_csv(input_file, header=None)
    # Header = None checks whether the columns have headers or not.#
    typeofLearning = input("Choose the type of Learning: 1. Supervised, 2. Unsupervised  : >> ")
    if typeofLearning == "2":
        print("Not implemented.")

    else:
        type = input("Choose type 1. Classification, 2. Regression : >> ")
        if type == "1":
            pass
        else:
            model = input("Choose Model 1.Univariate, 2.MultiVariate : >> ")

            while(True):
                user_input = input("Choose methods 1.head(), 2.describe(), 3.Plotting of data, 4.Predict, 5.Exit : >> ")
                if user_input == "1":
                    print(head(data))
                    continue
                elif user_input == "2":
                    print(describe(data))
                    continue
                elif user_input == "3":
                    ploting(data)

                elif user_input == "4":
                    learning_rate = 0.02
                    epochs = 3000
                    parameters_selection = input("Choose Parameters : 1. Yes 2. Use default : >> ")
                    if parameters_selection == "1":
                        learning_rate = float(input("Enter Learning rate (preferable b/w 0.001 to 0.05) : >> "))
                        epochs = int(input("Enter number of iterations (preferable b/w 1000 to 10000) : >> "))
                    flag = True
                    while(True):
                        get_data = input(
                            "To Print choose 1.Hypothesis, 2.Plot J(Θ) against the number of iteration, 3.Cost function visualisation, 4. Graph with Line of Best Fit, 5. Predict with test data, 6.Exit : >> ")
                        if flag:
                            theta_values, J_history, X,y = procedure(data, learning_rate, epochs)
                            flag = False
                        if get_data == "1":
                            print("Hypothesis Function")
                            print(printHypothesis(theta_values))
                        elif get_data == "2":
                            plot_gradientdescent(J_history)
                        elif get_data == "3":
                            costFunctionVisualisation(X,y)
                        elif get_data == "4":
                            bestFitline(data,theta_values)
                        elif get_data == "5":
                            testfile = input("Enter the test text file with .txt extension only : >> ")
                            testdata = pd.read_csv(testfile, header=None)
                            x = testdata.values
                            len_of_testdata = x.size
                            x = np.append(np.ones((len_of_testdata,1)),x[:, 0].reshape(len_of_testdata, 1),axis=1)
                            print(predict(x,theta_values))
                        elif get_data == "6":
                            break
                else:
                    break
