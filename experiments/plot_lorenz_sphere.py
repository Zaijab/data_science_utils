# system = LorenzSphere()

# initial_states = jnp.array(
#     [
#         [0.0, 0.1, 0.6, 0.3],
#         [0.0, 0.2, 0.5, 0.3],
#         [0.0, 0.3, 0.4, 0.3],
#         [0.0, 0.5, 0.5, 0.0],
#         [0.0, 0.5, 0.2, 0.3],
#         [0.0, 0.6, 0.1, 0.3],
#         [0.0, 0.7, 0.2, 0.1],
#         [0.0, 0.8, 0.1, 0.1],
#         [0.0, 0.9, 0.0, 0.1],
#         [0.0, 1.0, 0.0, 0.0],
#         [0.0, 0.0, 0.5, 0.5],
#     ]
# )


# @eqx.filter_jit
# def solve_over_state(y):
#     ts = jnp.linspace(0, 50, 10_000)
#     sol = diffeqsolve(
#         ODETerm(system.vector_field),
#         Tsit5(),
#         t0=ts[0],
#         t1=ts[-1],
#         dt0=0.01,
#         y0=y,
#         stepsize_controller=ConstantStepSize(),
#         saveat=SaveAt(ts=ts),
#         max_steps=1_000_000,
#     )
#     return sol.ys


# thing = eqx.filter_vmap(solve_over_state)(initial_states)
# print(thing.shape)

####
import numpy as np
from matplotlib.collections import LineCollection

fig = plt.figure(figsize=(10, 10))
ax = fig.subplots()
theta = np.linspace(0, 2 * np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), "k-", alpha=0.3)


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


for i in range(thing.shape[0]):
    trajectory = thing[i, ...]
    projected_trajectory = trajectory[jnp.abs(trajectory[:, 0]) < 0.01]
    red_vs_blue = np.where(
        eqx.filter_vmap(system.vector_field)(None, projected_trajectory, None)[:, 0]
        > 0,
        "red",
        "blue",
    )
    bright_vs_dark = np.where(projected_trajectory[:, 3] > 0, "", "dark")

    colors = np.array(
        [f"{prefix}{color}" for prefix, color in zip(bright_vs_dark, red_vs_blue)]
    )
    # color_values = np.zeros(len(bright_vs_dark))
    # for i in range(len(color_values)):
    #     if bright_vs_dark[i] == "" and red_vs_blue[i] == "red":
    #         color_values[i] = 0  # bright red
    #     elif bright_vs_dark[i] == "dark" and red_vs_blue[i] == "red":
    #         color_values[i] = 1  # dark red
    #     elif bright_vs_dark[i] == "" and red_vs_blue[i] == "blue":
    #         color_values[i] = 2  # bright blue
    #     else:  # dark blue
    #         color_values[i] = 3

    # # Then use color_values with your existing colored_line function
    # cmap = plt.cm.colors.ListedColormap(["red", "darkred", "blue", "darkblue"])
    # colored_line(
    #     projected_trajectory[:, 1],
    #     projected_trajectory[:, 2],
    #     color_values,
    #     ax,
    #     cmap=cmap,
    # )
    # colors = [f"r-" for prefix, color in zip(bright_vs_dark, red_vs_blue)]

    # colored_line(projected_trajectory[:, 1], projected_trajectory[:, 2], colors, ax)

    # plt.plot(projected_trajectory[:, 1], projected_trajectory[:, 2], colors)

    plt.scatter(
        projected_trajectory[:, 1],
        projected_trajectory[:, 2],
        c=colors,
    )
    break

plt.savefig("plots/gle4_single_trajectory.png")
