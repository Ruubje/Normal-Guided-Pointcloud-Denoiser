from matplotlib.patches import Circle as patch_Circle
from matplotlib.pyplot import (
    figure as plt_figure,
    scatter as plt_scatter,
    show as plt_show,
    title as plt_title,
    xlabel as plt_xlabel,
    ylabel as plt_ylabel,
)
from meshplot import plot as mp_plot
from mpl_toolkits.mplot3d import Axes3D as plt_tk_Axes3D
from numpy import (
    array as np_array,
    ndarray as np_ndarray,
    vstack as np_vstack, 
    zeros as np_zeros
)
from torch import (
    einsum as torch_einsum,
    linspace as torch_linspace,
    meshgrid as torch_meshgrid,
    tensor as torch_tensor,
    Tensor as torch_Tensor,
    zeros as torch_zeros,
    zeros_like as torch_zeros_like
)
from torch.linalg import (
    eigh as torch_linalg_eigh
)
from torch.nn.functional import normalize as torch_nn_functional_normalize
from typing import (
    Callable as typing_Callable
)

def visualize(vs, es):
    vsizes = np_array(list(map(lambda x: x.shape[0], vs)))
    csvsizes = vsizes.cumsum()
    c = np_zeros(csvsizes[-1])
    c[csvsizes[:-1]] = 1
    c.cumsum()
    v = np_vstack(vs)

    plot = mp_plot(v, c=c, shading={"point_size": 0.005})

    if not es:
        return
    
    esizes = np_array(list(map(lambda x: x.shape[1], es)))
    csesizes = esizes.cumsum()
    a = np_zeros(csesizes[-1], dtype=int)
    a[csesizes[:-1]] = csvsizes[:-1]
    a.cumsum()
    e = np_vstack(es)
    e += a

    starts = v[e[0]]
    ends = v[e[1]]

    plot.add_lines(starts, ends)

def visualize_coloring(v, c, f=None):
    mode = f is None
    a = np_zeros(len(v) if mode else len(f), dtype=int)
    for i, ci in enumerate(c):
        a[ci] = i + 1
    return mp_plot(v, f=f, c=a)

def plot_2d(x, y):
    plt_scatter(convert_to_numpy(x), convert_to_numpy(y))
    plt_xlabel('x')
    plt_ylabel('y')
    plt_title('Visualization of x and y data')
    plt_show()

def plot_3d(x, y, z):
    # Create a 3D scatter plot
    fig = plt_figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(convert_to_numpy(x), convert_to_numpy(y), convert_to_numpy(z), c='blue', marker='o')

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    return fig, ax
    
def plot_vectors(ax: plt_tk_Axes3D, start, direction, *args, **kwargs):
    ax.quiver(*start.t(), *direction.t(), *args, **kwargs)

def plot_plane(ax: plt_tk_Axes3D, center, normal, radius, alpha=0.5):
    xlim = center[0] + torch_tensor([-1, 1]) * radius
    ylim = center[1] + torch_tensor([-1, 1]) * radius
    resolution = 100
    x = torch_linspace(xlim[0], xlim[1], resolution)
    y = torch_linspace(ylim[0], ylim[1], resolution)
    x, y = torch_meshgrid(x, y, indexing='xy')
    z = ((normal*center).sum(dim=0) - normal[0] * x - normal[1] * y) / normal[2]
    ax.plot_surface(x.numpy(), y.numpy(), z, alpha=alpha, color="yellow")

"""
    Util functions below
"""

def convert_to_numpy(input_data):
    if isinstance(input_data, list):
        return np_array(input_data)
    elif isinstance(input_data, np_ndarray):
        return input_data
    elif isinstance(input_data, torch_Tensor):
        return input_data.numpy()
    else:
        raise ValueError("Unsupported data type. Please provide a list, numpy array, or torch tensor.")

"""
    Examples below
"""

def visTensorVoting(v: torch_Tensor, weight: torch_Tensor):
    center = v.mean(dim=0)
    input = v #- center.unsqueeze(0)
    outer = torch_einsum("bi,bj->bij", input, input)
    T = (weight.unsqueeze(1).unsqueeze(1)*outer).sum(dim=0)
    _, eigvec = torch_linalg_eigh(T)
    fig, ax = plot_3d(*v.t())
    r = (v.max(dim=0).values - v.min(dim=0).values)[0:2].max(dim=0).values * 0.5
    plot_vectors(ax, center, eigvec[:, 0] * r, color="red")
    plot_vectors(ax, center, eigvec[:, 1] * r, color="green")
    plot_vectors(ax, center, eigvec[:, 2] * r, color="blue")
    plot_plane(ax, center, eigvec[:, 0], r, 0.5)
    return fig, ax

def visNormalTensorVoting(v: torch_Tensor, n: torch_Tensor, weight: torch_Tensor):
    d = v - v.mean(dim=0, keepdim=True)
    w = torch_nn_functional_normalize(d.cross(n).cross(d))
    input = 2*torch_einsum("bi,bi->b", n, w).unsqueeze(1)*w - n
    outer = torch_einsum("bi,bj->bij", input, input)
    T = (weight.unsqueeze(1).unsqueeze(1)*outer).sum(dim=0)
    _, eigvec = torch_linalg_eigh(T)
    r = (n.max(dim=0).values - n.min(dim=0).values)[0:2].max(dim=0).values * 0.5
    points = input
    center = torch_zeros(points.size(1))
    fig, ax = plot_3d(*points.t())
    plot_vectors(ax, center, eigvec[:, 0] * r, color="red")
    plot_vectors(ax, center, eigvec[:, 1] * r, color="green")
    plot_vectors(ax, center, eigvec[:, 2] * r, color="blue")
    plot_plane(ax, center, eigvec[:, 0], r, 0.5)
    plot_vectors(ax, torch_zeros_like(points), points, color="purple", arrow_length_ratio=0.05, linewidth=0.3)
    return fig, ax