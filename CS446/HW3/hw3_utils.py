#import numpy as np
import torch
import torch.utils.data
import scipy
import scipy.spatial
import matplotlib.pyplot as plt



def load_one_nearest_neighbor_data():
    torch.manual_seed(0)
    X1,Y1 = linear_problem(torch.tensor([-1,0.3]), margin=1.0, size=100)
    X2,Y2 = linear_problem(torch.tensor([1,-0.1]), margin=6.0, size=100, bounds=[-7,7],trans=3)
    X = torch.cat([X1,X2], dim=0)
    Y = torch.cat([Y1,Y2],dim=0)
    return X,Y


def voronoi_plot(X,Y):
    # takes as input data set and saves a voronoi plot on disk as pdf
    voronoi = scipy.spatial.Voronoi(X)
    plt.clf()
    #render it once transparent so that the two figures have the points in the same places.
    #(to avoid "popping" when showing the two figures.)
    scipy.spatial.voronoi_plot_2d(voronoi, show_points = False, show_vertices = False,
                                  line_alpha = 0.0, line_width = 0.5,)
    plt.scatter(X[:, 0], X[:, 1], c = [ 'red' if yy >= 0 else 'blue' for yy in Y ], marker = 'X')
    plt.tight_layout()
    plt.savefig('1nn_data.pdf')
    scipy.spatial.voronoi_plot_2d(voronoi, show_points = False, show_vertices = False,
                                  line_alpha = 1.0, line_width = 0.5,)
    plt.scatter(X[:, 0], X[:, 1], c = [ 'red' if yy >= 0 else 'blue' for yy in Y ], marker = 'X',
                zorder = 4)
    plt.tight_layout()
    plt.savefig('1nn_voronoi.pdf')

def linear_problem(w, margin, size, bounds=[-5., 5.], trans=0.0):
    in_margin = lambda x: torch.abs(w.flatten().dot(x.flatten())) / torch.norm(w) \
                          < margin
    X = []
    Y = []
    for i in range(size):
        x = torch.zeros(2).uniform_(bounds[0], bounds[1]) + trans
        while in_margin(x):
            x.uniform_(bounds[0], bounds[1]) + trans
        if w.flatten().dot(x.flatten()) + trans > 0:
            Y.append(torch.tensor(1.))
        else:
            Y.append(torch.tensor(-1.))
        X.append(x)
    X = torch.stack(X)
    Y = torch.stack(Y).reshape(-1, 1)

    return X, Y
