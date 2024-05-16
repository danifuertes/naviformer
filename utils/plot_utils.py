import random
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import splprep, splev


def path_smoothing(path: np.ndarray) -> np.ndarray:
    """
    Smooth the given path.

    Args:
        path (numpy.ndarray): The path to be smoothed.

    Returns:
        numpy.ndarray: The smoothed path.
    """
    tck, *rest = splprep(path, s=0.005)
    return np.array(splev(np.linspace(0, 1, num=path.shape[1]), tck))


def assign_colors(n: int) -> list:
    """
    Assign colors to elements.

    Args:
        n (int): The number of elements to assign colors to.

    Returns:
        list: A list of colors assigned to the elements.
    """
    color = {k: [] for k in 'rgb'}
    for i in range(n):
        temp = {k: random.randint(0, 230) for k in 'rgb'}
        for k in temp:
            while 1:
                c = temp[k]
                t = set(j for j in range(c - 25, c + 25) if 0 <= j <= 230)
                if t.intersection(color[k]):
                    temp[k] = random.randint(0, 230)
                else:
                    break
            color[k].append(temp[k])
    return [(color['r'][i] / 256, color['g'][i] / 256, color['b'][i] / 256) for i in range(n)]


def get_title(
        model_name: str,
        problem: str,
        data_dist: str,
        distance: float = 0.,
        reward: float = 0.,
        total_prize: float = 0.,
        max_length: float = 0.,
        is_path: bool = False) -> str:
    """
    Generate the plot title.

    Args:
        model_name (str): The name of the model.
        problem (str): The problem type.
        data_dist (str): The data distribution.
        distance (float): The distance.
        reward (float): The reward.
        total_prize (float): The total prize.
        max_length (float): The maximum travel length allowed.
        is_path (bool): Whether the title is for a path or a route.

    Returns:
        str: The generated title.
    """
    prize_str = 'Nodes' if is_path else 'Prize'
    title = problem.upper()
    title += ' (' + data_dist.lower() + ')' if len(data_dist) > 0 else ''
    title += ' - {:s}'.format(model_name)
    if distance >= 0:
        title += ': Length = {:.4g}'.format(distance)
    if max_length >= 0:
        title += ' /' if distance >= 0 else ': Max Length = '
        title += ' {:.4g}'.format(max_length)
    if reward >= 0:
        title += ' |' if distance >= 0 or max_length >= 0 else ':'
        title += ' {} = {:.4g}'.format(prize_str, reward)
    if total_prize >= 0:
        title += ' | Max ' + prize_str.lower() + ' = ' if (distance >= 0 or max_length >= 0) and reward < 0 else ''
        title += ' /' if reward >= 0 else ''
        title += '' if distance >= 0 or max_length >= 0 or reward >= 0 else ':'
        title += ' {:.4g}'.format(total_prize)
    return title


def plot(
        tours: list,
        batch: dict,
        problem: str,
        model_name: str,
        data_dist: str = '',
        success: bool = True,
        num_dirs: int = 4) -> None:
    """
    Plot a general tour and then individual tours for each agent.

    Args:
        tours (list): The list of tours (one per agent).
        batch (dict): The batch data.
        problem (str): The problem type.
        model_name (str): The name of the model.
        data_dist (str): The data distribution.
        success (bool): Whether the tours were successful.
        num_dirs(int): number of directions available for predictors.
    """

    # General plot
    if len(tours) > 1:
        fig, ax = None, None
        colors = assign_colors(len(tours))
        nodes = np.unique(np.concatenate([np.unique(tour[..., 0]).tolist() for tour in tours]))
        nodes = len(nodes[np.logical_and(nodes != 0, nodes != len(batch['loc']) + 1)])
        tp = len(batch['loc'][batch['loc'][..., 0] > 0])
        title = get_title(
            model_name, problem, data_dist, distance=-1, reward=nodes, total_prize=tp, max_length=-1, is_path=True
        )
        for i, tour in enumerate(tours):
            scenario = True if i == 0 else False
            fig, ax = plot_path(
                tour, batch, problem, model_name,
                data_dist=data_dist,
                iteration=i,
                save_image='',
                show=False,
                fig=fig,
                ax=ax,
                scenario=scenario,
                colors=colors,
                title=title,
                num_dirs=num_dirs,
            )
        plt.show()
    else:
        colors = None

    # Individual plots
    for i, tour in enumerate(tours):
        nodes = np.unique(tour[..., 0])
        nodes = len(nodes[np.logical_and(nodes != 0, nodes != len(batch['loc']) + 1)])
        nodes -= 0 if success else 1
        plot_path(
            tour, batch, problem, model_name, data_dist=data_dist, iteration=i, nodes=nodes, colors=colors, num_dirs=num_dirs
        )


def plot_path(
        tour: np.ndarray,
        batch: dict,
        problem: str,
        model_name: str,
        data_dist: str = '',
        nodes: float = 0.,
        num_dirs: int = 4,
        iteration: int = 0,
        save_image: str = '',
        show: bool = True,
        fig: Figure | None = None,
        ax: Axes | None = None,
        scenario: bool = True,
        colors: list | None = None,
        title: str = '') -> Tuple[Figure, Axes] | None:
    """
        Plot a single path.

        Args:
            tour (numpy.ndarray): The tour data.
            batch (dict): The batch data.
            problem (str): The problem type.
            model_name (str): The name of the model.
            data_dist (str): The data distribution.
            nodes (float): The number of nodes.
            num_dirs (int): Number of directions available for predictors.
            iteration (int): The iteration number.
            save_image (str): Path to save the image.
            show (bool): Whether to show the plot.
            fig (matplotlib.figure.Figure, optional): The figure object.
            ax (matplotlib.axes.Axes, optional): The axes object.
            scenario (bool): Whether it's a scenario plot.
            colors (list, optional): List of colors for plotting.
            title (str): The title of the plot.

        Returns:
            matplotlib.figure.Figure, matplotlib.axes.Axes or None: The figure and axes if `show` is False, else None.
    """
    time_step = 2e-2

    # Initialize plot
    # fig = plt.figure(iteration) if fig is None else fig
    not_ax = True if ax is None else False
    ax = plt.subplots()[1] if not_ax else ax
    if not_ax:
        ax.set_aspect('equal')
        # fig.axes.append(ax)
        plt.xlim([-.05, 1.05])
        plt.ylim([-.05, 1.05])

    # Data
    depot_ini = batch['depot_ini']
    depot_end = batch['depot_end']
    loc = batch['loc']
    end_ids = len(loc) + 1
    max_length = batch['max_length']
    total_prize = batch['prize'].sum()

    # Plot nodes (purple circles), initial depot (blue circle), end depot (red circle) and obstacles (black circles)
    if scenario:
        plt.scatter(loc[..., 0], loc[..., 1], c='mediumpurple', s=90)
        plt.scatter(*depot_ini, c='tab:cyan', s=90)
        plt.scatter(*depot_end, c='tab:red', s=90)
        if 'obs' in batch:
            for obs in batch['obs']:
                ax.add_patch(plt.Circle(obs[:2], obs[2], color='k'))

    # If no tour is provided
    if len(tour.shape) == 0:
        title = get_title(
            model_name, problem, data_dist,
            distance=0, reward=0, total_prize=total_prize, max_length=max_length, is_path=False
        )
        plt.title(title)
        plt.show()
        return

    # Add depots to loc
    loc = np.concatenate(([depot_ini], loc, [depot_end]), axis=0)

    # Plot regions numbers (indexes)
    if scenario:
        for i in range(loc.shape[0]):
            plt.text(loc[i, 0], loc[i, 1], str(i))

    # Draw arrows
    c = 'g' if colors is None else colors[iteration]
    cur_coord = depot_ini
    d = 0
    for i in range(0, tour.shape[0]):
        if i == tour.shape[0] - 2:
            print()
        if tour.shape[1] == 2:
            angle = tour[i, 1] * 2 * np.pi / num_dirs
            new_coord = cur_coord + np.array([time_step * np.cos(angle), time_step * np.sin(angle)])
        else:
            new_coord = tour[i, 1:]
        d += np.linalg.norm(cur_coord - new_coord)
        plt.plot([cur_coord[0], new_coord[0]], [cur_coord[1], new_coord[1]], c=c)
        dist2end = np.linalg.norm(new_coord - depot_end, axis=-1)
        dist2obs = np.linalg.norm(new_coord - batch['obs'][:, :2], axis=-1)
        not_success = np.any(dist2obs < batch['obs'][:, 2]) or d > max_length
        if (dist2end <= time_step and tour[i, 0] == end_ids) or not_success:
            if not_success:
                plt.scatter(*new_coord, marker='x', c='r', s=90)
            break
        cur_coord = new_coord

    # Set title
    if title == '':
        title = get_title(
            model_name, problem, data_dist,
            distance=d, reward=nodes, total_prize=total_prize, max_length=max_length, is_path=False
        )
    plt.title(title)

    # Show/save plot
    if save_image != '':
        # plt.title('')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_image, dpi=300, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    else:
        return fig, ax
