import numpy as np
import matplotlib.pyplot as plt
from palettable.cmocean.sequential import Ice_15
from matplotlib import cm
import pybbtd.stokes as stokes


def plot_Stokes_terms(outA, outB, outC, R, Lr):
    width_ratio = np.ones(R)
    width_ratio = np.append(width_ratio, 0.3)
    fig, axis = plt.subplots(
        2,
        R + 1,
        figsize=(3 * (2 + R * 2), 3 * 3),
        gridspec_kw={"width_ratios": width_ratio, "height_ratios": [1, 0.25]},
    )
    result = np.zeros((R, outA.shape[0], outB.shape[0]))

    for i in range(0, Lr * R, Lr):
        result[int(i / Lr)] = outA[:, i : (i + Lr)] @ outB[:, i : (i + Lr)].T
        normalized = (result[int(i / Lr)] - result[int(i / Lr)].min()) / result[
            int(i / Lr)
        ].max()
        im = axis[0, int(i / Lr)].imshow(normalized, cmap=Ice_15.mpl_colormap)
        axis[0, int(i / Lr)].set_aspect(aspect=1, adjustable="datalim")
        axis[0, int(i / Lr)].axis("off")

    plt.colorbar(im, ax=axis[0, R], aspect=8, fraction=1, location="left")

    cmap = cm.get_cmap("Spectral")

    for i in range(R):
        color = cmap(i * 50)
        elip = compute_ellipse(outC[:, i])
        im = axis[1, i].plot(elip[0, :], elip[1, :], color=color)[0]
        add_arrow(im, size=10, position=0)
        axis[1, i].axis("off")
        axis[1, i].set_xticks([])
        axis[1, i].set_yticks([])
        axis[1, i].set_aspect(aspect=1, adjustable="datalim", anchor="C")

    axis[1, R].axis("off")
    axis[0, R].axis("off")

    fig.tight_layout()


def compute_ellipse(S):
    psi, _, _, _ = stokes.stokes_2_elip(S)

    a = np.sqrt(
        1 / 2 * ((S[1] ** 2 + S[2] ** 2 + S[3] ** 2) + (np.sqrt(S[1] ** 2 + S[2] ** 2)))
    )
    b = np.sqrt(
        1 / 2 * ((S[1] ** 2 + S[2] ** 2 + S[3] ** 2) - (np.sqrt(S[1] ** 2 + S[2] ** 2)))
    )

    b = b / a
    # Check why this is one when updating the function
    a = 1

    t = np.linspace(0, 2 * np.pi, 100)
    Ell = np.array([a * np.cos(t), b * np.sin(t)])

    # 2-D rotation matrix
    R_rot = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])

    Ell_rot = np.zeros((2, Ell.shape[1]))

    for i in range(Ell.shape[1]):
        Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])

    return Ell_rot


def add_arrow(line, position=None, direction="right", size=15, color=None):
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # print(xdata.shape)
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    end_ind = start_ind + 0
    # print(end_ind)
    # if direction == 'right':
    #     end_ind = start_ind + 1
    # else:
    #     end_ind = start_ind - 1

    line.axes.annotate(
        "",
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="<|-", color=color),
        size=size,
    )
