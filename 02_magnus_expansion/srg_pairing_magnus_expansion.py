# This version replaces the python odeint in srg_pairing.py to Magnus expansion method
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
import math

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


#Replace the python odeint with Magnus expantion method
#y0: Initial value, H0
#flowparams: Horizontal ordinates of output points
#kmax: Truncation accuracy
#step: Step length accuracy
#args: Extra arguments to pass to model function.
def magnus(y0,flowparams,kmax,step,args=()):
    H0 = reshape(y0, (args[0], args[0])) #H(0)
    omega = np.zeros((2, args[0], args[0]))
    B = (1, -1.0/2, 1.0/6, 0, -1.0/30, 0, 1.0/42, 0, -1.0/30, 0, 5.0/66)
    a = np.zeros((len(flowparams), args[0] * args[0]))  # output
    a[0] = y0
    length = flowparams[len(flowparams) - 1] - flowparams[0]  # range of output
    t = np.linspace(flowparams[0], flowparams[len(flowparams) - 1], length / step + 1)  # steps
    j = 1  # counter
    Hs = H0  # initial value
    for i in range(1, len(t)):
        Hd = diag(diag(Hs))
        eta = commutator(Hd, Hs - Hd)
        derivative = np.zeros((args[0], args[0])) # the derivative function of omega
        ad = eta # ad0_eta
        for k in range(0, kmax+1):
            derivative = derivative + ad * B[k] / math.factorial(k)
            ad = commutator(omega[0], ad)
        omega[1] = omega[0] + derivative * step
        ad = H0  # ad0_omega
        Hs = 0 # reset to zero
        for k in range(0, kmax + 1):
            Hs = Hs + ad / math.factorial(k)
            ad = commutator(omega[1], ad)
        if flowparams[j] == t[i]:  #
            a[j] = reshape(Hs, -1)
            j = j + 1
        omega[0] = omega[1]

    return a[len(a)-1]

#--------------------------------------------------------------
# Main program
#--------------------------------------------------------------
def main():
    g = 0.5
    delta = 1

    H0 = Hamiltonian(delta, g)
    dim = H0.shape[0]

    # calculate exact eigenvalues
    eigenvalues = eigvalsh(H0)

    # turn initial Hamiltonian into a linear array
    y0 = reshape(H0, -1)

    # flow parameters for snapshot images
    flowparams = array([0., 0.001, 0.01, 0.05, 0.1, 1., 5., 10.])

    # truncation
    kmax = 10

    # integrate flow equations - odeint returns an array of solutions,
    # which are 1d arrays themselves
    ys = magnus(y0, flowparams, kmax, 0.001, args=(dim,))

    # reshape individual solution vectors into dim x dim Hamiltonian
    # matrices
    Hs = reshape(ys, (-1, dim, dim))

    # print Hs[-1]
    # print eigvalsh(Hs[-1])

    data = []
    for h in Hs:
        data.append(diag(h))
    data = zip(*data)

#   plot_snapshots(Hs, flowparams, delta, g, 1)
#   plot_diagonals(data, eigenvalues, flowparams, delta, g)

 # TURN ON WHEN Evaluating the effect of kmax on H(10)
 #   ys = np.zeros((5, dim*dim))
 #   i = 0 # counter
 #   for k in (2,4,6,8):
 #       ys[i] = magnus(y0, flowparams, k, 0.001, args=(dim,))
 #       i = i + 1
#    Hs = reshape(ys, (-1, dim, dim))
#    plot_snapshots(Hs, flowparams, delta, g, 2)

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
    ax.set_xlim([0.0007, 13])
    return


# ------------------------------------------------------------------------------
# plot routines
# ------------------------------------------------------------------------------
def plot_diagonals(data, eigenvalues, flowparams, delta, g):
    '''plot eigenvalues and diagonals'''
    dim = len(data)
    formatter = FuncFormatter(myLabels)
    markers = ['o' for i in range(dim)]
    cols = ['blue', 'red', 'purple', 'green', 'orange', 'deepskyblue']

    # diagonals vs. eigenvalues on absolute scale
    fig, ax = plt.subplots()
    for i in range(dim):
        plt.semilogx(flowparams, [eigenvalues[i] for e in range(flowparams.shape[0])], color=cols[i], linestyle='solid')
        plt.semilogx(flowparams, data[i], color=cols[i], linestyle='dashed', marker=markers[i], markersize=10)

    myPlotSettings(ax, formatter)

    plt.savefig("srg_pairing_diag_delta%2.1f_g%2.1f.pdf" % (delta, g), bbox_inches="tight", pad_inches=0.05)
    plt.show()

    # difference between diagonals and eigenvalues
    fig, ax = plt.subplots()
    for i in range(dim):
        plot_diff = plt.semilogx(flowparams, data[i] - eigenvalues[i], color=cols[i], linestyle='solid',
                                 marker=markers[i], markersize=10)

    myPlotSettings(ax, formatter)

    plt.savefig("srg_pairing_diag-eval_delta%2.1f_g%2.1f.pdf" % (delta, g), bbox_inches="tight", pad_inches=0.050)
    plt.show()
    return


# ------------------------------------------------------------------------------
# plot matrix snapshots
# ------------------------------------------------------------------------------
def plot_snapshots(Hs, flowparams, delta, g, number):
    fig = plt.figure(1, (10., 5.))
    grid = AxesGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(2, Hs.shape[0] / 2),  # creates grid of axes
                    axes_pad=0.25,  # pad between axes in inch.
                    label_mode='L',  # put labels on left, bottom
                    cbar_mode='single',  # one color bar (default: right of last image in grid)
                    cbar_pad=0.20,  # insert space between plots and color bar
                    cbar_size='10%'  # size of colorbar relative to last image
                    )

    # create individual snapshots - figures are still addressed by single index,
    # despite multi-row grid
    for s in range(Hs.shape[0]):
        img = grid[s].imshow(Hs[s],
                             cmap=plt.get_cmap('RdBu_r'),  # choose color map
                             interpolation='nearest',
                             norm=SymLogNorm(linthresh=1e-10, vmin=-0.5 * g, vmax=10 * delta),  # normalize
                             vmin=-0.5 * g,  # min/max values for data
                             vmax=10 * delta
                             )

        # tune plots: switch off tick marks, ensure that plots retain aspect ratio
        grid[s].set_title('$s=%s$' % flowparams[s])

        # TURN ON WHEN CALCULATION RELATED TO KMAX
        #k = (2,4,6,8)
        #grid[s].set_title('$k=%s$' % k[s])

        grid[s].tick_params(

            bottom='off',
            top='off',
            left='off',
            right='off'
        )

        grid[s].set_xticks([0, 1, 2, 3, 4, 5])
        grid[s].set_yticks([0, 1, 2, 3, 4, 5])
        grid[s].set_xticklabels(['$0$', '$1$', '$2$', '$3$', '$4$', '$5$'])
        grid[s].set_yticklabels(['$0$', '$1$', '$2$', '$3$', '$4$', '$5$'])

        cbar = grid.cbar_axes[0]
        plt.colorbar(img, cax=cbar,
                     ticks=[-1.0e-1, -1.0e-3, -1.0e-5, -1.0e-7, -1.09e-9, 0.,
                            1.0e-9, 1.0e-7, 1.0e-5, 1.0e-3, 0.1, 10.0]
                     )

        cbar.axes.set_yticklabels(['$-10^{-1}$', '$-10^{-3}$', '$-10^{-5}$', '$-10^{-7}$',
                                   '$-10^{-9}$', '$0.0$', '$10^{-9}$', '$10^{-7}$', '$10^{-5}$',
                                   '$10^{-3}$', '$10^{-1}$', '$10$'])
        cbar.set_ylabel('$\mathrm{[a. u.]}$')

        plt.savefig("srg_pairing_delta%2.1f_g%2.1f_%d.pdf" % (delta, g, number), bbox_inches="tight", pad_inches=0.05)
        plt.show()
    return


if __name__ == "__main__":
    main()