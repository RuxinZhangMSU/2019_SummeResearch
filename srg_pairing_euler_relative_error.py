# Quantifying the relative error as a function of the step size in s when using 1st-order Euler step method
# In the output figure, 6 colored line represent relative error of 6 eigenvalues of H(s=10) under different step length
# Ruxin Zhang, June 2019
# ------------------------------------------------------------------------------


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import SymLogNorm, Normalize
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable

import numpy as np
from numpy import array, dot, diag, reshape
from scipy.linalg import eigvalsh
from scipy.integrate import odeint

# Hamiltonian for the pairing model
def Hamiltonian(delta, g):
# delta is the spacing of single-particle levels and g is the strength of the paring interaction
    H = array(
        [[2 * delta - g, -0.5 * g, -0.5 * g, -0.5 * g, -0.5 * g, 0.],
         [-0.5 * g, 4 * delta - g, -0.5 * g, -0.5 * g, 0., -0.5 * g],
         [-0.5 * g, -0.5 * g, 6 * delta - g, 0., -0.5 * g, -0.5 * g],
         [-0.5 * g, -0.5 * g, 0., 6 * delta - g, -0.5 * g, -0.5 * g],
         [-0.5 * g, 0., -0.5 * g, -0.5 * g, 8 * delta - g, -0.5 * g],
         [0., -0.5 * g, -0.5 * g, -0.5 * g, -0.5 * g, 10 * delta - g]]
    )
    return H

# commutator of matrices
def commutator(a, b):
    return dot(a, b) - dot(b, a)

# derivative / right-hand side of the flow equation
def derivative(y, t, dim):
    # reshape the solution vector into a dim x dim matrix
    H = reshape(y, (dim, dim))

    # extract diagonal Hamiltonian...
    Hd = diag(diag(H))

    # ... and construct off-diagonal the Hamiltonian
    Hod = H - Hd

    # calculate the generator
    eta = commutator(Hd, Hod)

    # dH is the derivative in matrix form
    dH = commutator(eta, H)

    # convert dH into a linear array for the ODE solver
    dydt = reshape(dH, -1)

    return dydt

#Replace the python odeint with 1st-order forward Euler step method
#derivative: The derivative function
#y0: Initial value
#flowparams: Horizontal ordinates of output points
#step: Step length, accuracy
#args: Extra arguments to pass to model function.
def euler(derivative,y0,flowparams,step,args=()):
    y = np.zeros((2,args[0]*args[0])) #y_n & y_n+1
    a = np.zeros((len(flowparams),args[0] * args[0])) #output
    y[0] = y0 #initial value
    length = flowparams[len(flowparams)-1]+step-flowparams[0] #range of output
    t = np.linspace(flowparams[0],flowparams[len(flowparams)-1]+step,length/step+1) #steps
    j = 1  # counter

    for i in range(0,len(t)-1):
        y[1] = y[0] + dot(derivative(y[0],t[i],args[0]),(t[i+1]-t[i]))
        if flowparams[j] == t[i]: #
            a[j-1] = y[1]
            print flowparams[j]
        y[0] = y[1]

    return a[0]

#--------------------------------------------------------------
# Main program
#--------------------------------------------------------------
def main():
    g = 0.5
    delta = 1

    H0 = Hamiltonian(delta, g)
    dim = H0.shape[0]

    # calculate exact eigenvalues
    eigenvalues = np.zeros((10,dim)) #different steps * number of eigenvalues
    eigenvalues[0] = eigvalsh(H0) #initial eigenvalues before change

    # turn initial Hamiltonian into a linear array
    y0 = reshape(H0, -1)

    # flow parameters for snapshot images
    flowparams = array([0., 10.])

    # integrate flow equations - odeint returns an array of solutions,
    # which are 1d arrays themselves
    i = 1 # counter
    steps = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    for step in steps:
        print step
        ys = euler(derivative, y0, flowparams, step, args=(dim,))
        eigenvalues[i] = eigvalsh(reshape(ys,(dim,dim)))
        print reshape(ys,(dim,dim))
        i = i + 1

    plot_error(eigenvalues, steps, delta, g)


# ------------------------------------------------------------------------------
# plot helpers
# ------------------------------------------------------------------------------
def myLabels(x, pos):
    '''format tick labels using LaTeX-like math fonts'''
    return '$%s$' % x


def myPlotSettings(ax, formatter):
    '''save these settings for use in other plots'''
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis='both', which='major', width=1.5, length=8)
    ax.tick_params(axis='both', which='minor', width=1.5, length=5)
    ax.tick_params(axis='both', width=2, length=10, labelsize=20)
    for s in ['left', 'right', 'top', 'bottom']:
        ax.spines[s].set_linewidth(2)
    ax.set_xlim([0.00005, 0.02])
    return


# ------------------------------------------------------------------------------
# plot routines
# ------------------------------------------------------------------------------
def plot_error(eigenvalues, steps, delta, g):
    '''plot eigenvalues and diagonals'''
    dim = len(eigenvalues[0])
    formatter = FuncFormatter(myLabels)
    markers = ['o' for i in range(dim)]
    cols = ['blue', 'red', 'purple', 'green', 'orange', 'deepskyblue']

    # diagonals vs. eigenvalues on absolute scale
    fig, ax = plt.subplots()
    for i in range(dim):
        y = []
        for j in range(len(steps)):
            y.append((eigenvalues[j+1][i]-eigenvalues[0][i])/eigenvalues[0][i])
            print y[j]
        plt.semilogx(steps, y, color=cols[i], linestyle='solid')
    myPlotSettings(ax, formatter)

    plt.savefig("srg_pairing_euler_errordelta%2.1f_g%2.1f.pdf" % (delta, g), bbox_inches="tight", pad_inches=0.05)
    plt.show()

    return

if __name__ == "__main__":
    main()