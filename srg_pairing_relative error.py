# Quantifying the relative error as a function of parameter s when using odeint as ODE solver
# In the output figure, 6 colored line represent relative error of 6 eigenvalues of H(s) to H(0) under s-transformation
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

#--------------------------------------------------------------
# Main program
#--------------------------------------------------------------
def main():
    g = 0.5
    delta = 1

    H0 = Hamiltonian(delta, g)
    dim = H0.shape[0]

    # turn initial Hamiltonian into a linear array
    y0 = reshape(H0, -1)

    # flow parameters for snapshot images
    flowparams = array([0., 0.001, 0.01, 0.05, 0.1, 1., 5., 10.])

    # calculate exact eigenvalues
    eigenvalues = np.zeros((len(flowparams),dim)) #different steps * number of eigenvalues

    # integrate flow equations - odeint returns an array of solutions,
    # which are 1d arrays themselves
    ys = odeint(derivative, y0, flowparams, args=(dim,))


    # reshape individual solution vectors into dim x dim Hamiltonian
    # matrices
    Hs = reshape(ys, (-1, dim, dim))

    for i in range(0, len(flowparams)):
        eigenvalues[i] = eigvalsh(Hs[i])

    plot_error(eigenvalues, flowparams, delta, g)



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
    ax.tick_params(axis='both', width=2, length=10, labelsize=6)
    for s in ['left', 'right', 'top', 'bottom']:
        ax.spines[s].set_linewidth(2)
    ax.set_xlim([0.0007, 13])
    return

# ------------------------------------------------------------------------------
# plot routines
# ------------------------------------------------------------------------------
def plot_error(eigenvalues, flowparams, delta, g):
    dim = len(eigenvalues[0])
    formatter = FuncFormatter(myLabels)
    cols = ['blue', 'red', 'purple', 'green', 'orange', 'deepskyblue']

    fig, ax = plt.subplots()
    for i in range(dim):
        y = []
        for j in range(len(flowparams)):
            y.append((eigenvalues[j][i]-eigenvalues[0][i])/eigenvalues[0][i])
            print y[j]
        plt.semilogx(flowparams, y, color=cols[i], linestyle='solid')
    myPlotSettings(ax, formatter)

    plt.savefig("srg_pairing_error_delta%2.1f_g%2.1f.pdf" % (delta, g), bbox_inches="tight", pad_inches=0.05)
    plt.show()

    return

if __name__ == "__main__":
    main()