import os.path
import argparse
import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch


def get_args_parser():
    """
    Create and return an argument parser for the radar plot script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description='Draw a radar plot.')
    parser.add_argument('--input_file', default='/data/private/Experiments_filled.xlsx', type=str,
                        help='Path to the input csv file containing experiment results.')
    parser.add_argument('--labels_away_axis', nargs='+', type=str,
                        help='Optional list of axis labels that should be moved slightly to avoid overlap.')
    parser.add_argument('--datasets', nargs='+', type=str,
                        help='Optional list of dataset names to include in the plot. '
                             'If not provided, all datasets in the file will be used.')
    parser.add_argument('--tick_length', default=5, type=int,
                        help='Number of ticks to display on each radar axis.')
    parser.add_argument('--save_filename', default='radar.png', type=str,
                        help='Name of the output radar plot file.')

    return parser


class Radar:
    """
    A class for creating a radar (spider) plot with multiple polar axes.

    Attributes:
        min_value (float): Minimum radius value for the main axis.
        tolerance (float): Range of data mapped to the main axis scale.
        mins (list[float]): List of minimum values for each axis.
        tols (list[float]): List of intervals (axis tick spacing) for each axis.
        angles (np.ndarray): Angles (in degrees) for each axis.
        axes (list): List of polar axes created for the radar plot.
        ax (matplotlib.axes._subplots.PolarAxesSubplot): Primary polar axis.
    """
    def __init__(
        self, 
        figure, 
        title, 
        labels, 
        ax0_min, 
        ax0_tol, 
        mins, 
        tols, 
        reg_tasks_list, 
        labels_away_axis=None, 
        rect=None
    ):
        """
        Initialize the Radar object.

        Args:
            figure (matplotlib.figure.Figure): The figure to draw on.
            title (list[str]): List of axis labels (text around the polar plot).
            labels (list[list]): List of lists, each containing tick labels for an axis.
            ax0_min (float): Minimum radius value for the 0th axis.
            ax0_tol (float): Tolerance (range) for the 0th axis.
            mins (list[float]): List of minimum values for each task axis.
            tols (list[float]): List of intervals between ticks for each task axis.
            reg_tasks_list (list[str]): List of tasks that are regressions (vs. classification).
            labels_away_axis (list[str]): Task labels that need to be shifted to avoid overlap.
            rect (list[float]): Figure coordinates: [left, bottom, width, height] for the axis.
        """
        self.min_value = ax0_min
        self.tolerance = ax0_tol
        self.mins = mins
        self.tols = tols

        # Default rectangular position if none is provided
        if rect is None:
            rect = [0.2, 0.2, 0.6, 0.6]

        # Number of tasks / axes
        self.n = len(title)
        # Divide full circle (360°) by the number of tasks/axes
        self.angles = np.arange(0, 360, 360.0 / self.n)

        # Create one polar axis per task
        self.axes = [
            figure.add_axes(rect, projection='polar', label='axes%d' % i)
            for i in range(self.n)
        ]

        # Store reference to the first (main) axis
        self.ax = self.axes[0]

        # Set polar coordinate angle grid lines and axis titles
        self.ax.set_thetagrids(self.angles, title)

        # Increase line width for radial (x) and circular (y) grid lines
        for x_gird_line, y_gird_line in zip(self.ax.xaxis.get_gridlines(), 
                                            self.ax.yaxis.get_gridlines()):
            x_gird_line.set_linewidth(2)
            y_gird_line.set_linewidth(2)

        # Adjust the line width of the polar radius grid lines
        for grid_line in self.ax.yaxis.get_gridlines():
            grid_line.set_linewidth(2)

        # Adjust the vertical offset and style of axis labels
        for label, tick in zip(self.ax.get_xticklabels(), self.ax.get_xticks()):
            label_name = label.get_text()

            # Shift labels for specified tasks to reduce overlap
            if labels_away_axis and any(element in label_name for element in labels_away_axis):
                label.set_y(label.get_position()[1] - 0.25)
            else:
                label.set_y(label.get_position()[1] - 0.15)

            # Set color scheme for regression vs. classification tasks
            if any(element in label_name for element in reg_tasks_list):
                # Regression tasks are deep pink
                color = '#991a5f'
            else:
                # Classification tasks are dark blue
                color = '#004d7f'

            # Add a box around each axis label
            label.set_bbox(dict(
                facecolor="white", 
                edgecolor=color, 
                boxstyle="round", 
                linewidth=2
            ))

        # Create a legend for task type: regression or classification
        task_legend_elements = [
            FancyBboxPatch(
                (0, 0), 2, 2,
                boxstyle="round,pad=0.5",
                linewidth=2,
                facecolor='white',
                edgecolor='#991a5f',
                label='Slide regression'
            ),
            FancyBboxPatch(
                (0, 0), 2, 2,
                boxstyle="round,pad=0.5",
                linewidth=2,
                facecolor='white',
                edgecolor='#004d7f',
                label='Slide classification'
            ),
        ]
        # Add the legend for task types at the top-left corner of the figure
        figure.legend(handles=task_legend_elements, loc='upper left',
                      bbox_to_anchor=(0., 0.98), frameon=True)

        # Hide repetitive backgrounds and grid lines from axes other than the main one
        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid(False)
            ax.xaxis.set_visible(False)

        # Set up radial grids, ticks, and limits for each axis
        for ax, angle, label in zip(self.axes, self.angles, labels):
            # Configure the radial ticks
            ax.set_rgrids(label[1:], angle=angle, labels=label[1:])
            ax.spines['polar'].set_visible(False)
            # The first element of `label` is the min, the last is the max
            ax.set_ylim(label[0], label[-1])

    def plot(self, values, *args, **kw):
        """
        Plot data values on the radar chart.

        Args:
            values (list[float]): Data values (one for each axis).
            *args: Additional positional arguments for matplotlib.
            **kw: Additional keyword arguments for matplotlib.
        """
        # Convert angles to radians and close the loop by repeating the first angle
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])

        # Map each data value to the scale of the main (0th) axis
        index = 1
        for item in values[1:]:
            temp = ((item - self.mins[index]) / self.tols[index]) * self.tolerance + self.min_value
            values[index] = temp
            index += 1

        # Close the data loop
        values = np.r_[values, values[0]]

        # Draw the polygon and fill its area
        self.ax.plot(angle, values, *args, **kw)
        self.ax.fill(angle, values, alpha=0.3, *args, **kw)


def main(args):
    """
    Main function to load data, build the radar plot, and save the figure.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    # Read the csv file
    source = pd.read_csv(args.input_file)
    # sort values
    source = source.sort_values(by=["task_type", "dataset", "model"])

    # Filter by dataset list if provided; otherwise, use all
    datasets = args.datasets if args.datasets else source['dataset'].unique()
    source = source.loc[source['dataset'].isin(datasets)]
    models = source['model'].unique()
    num_models = len(models)

    print(f'Find {len(datasets)} datasets and {num_models} models.')

    # Combine `task` and `dataset` in a single label
    source['task_name'] = source['task'] + '\n(' + source['dataset'] + ')'
    tasks_list = source['task_name'].unique()
    num_tasks = len(tasks_list)

    print(f'Find {num_tasks} tasks.')

    # Identify regression tasks for labeling
    reg_tasks_list = source.loc[source['task_type'] == 'reg', 'task'].unique()
    print("Regression tasks:", reg_tasks_list)

    # Calculate tick ranges for each task
    ticks = []
    tick_length = args.tick_length
    mins = []
    tols = []

    for task_name in tasks_list:
        task_df = source[source['task_name'] == task_name]
        task_result = task_df['result'].tolist()

        if any(element in task_name for element in reg_tasks_list):
            # For regression tasks, compute float-based ticks
            _min = min(task_result)
            _max = max(task_result)
            step = (_max - _min) / tick_length
            tick = np.linspace(_min - step, _max + step, tick_length).tolist()
            tick = [round(item, 2) for item in tick]
            mins.append(min(tick))
            tols.append(tick[1] - tick[0])
        else:
            # For classification tasks, use integer-based ticks
            _min = math.floor(min(task_result))
            _max = math.ceil(max(task_result))
            interval = (_max - _min) // (tick_length - 2) \
                       + (1 if (_max - _min) % (tick_length - 2) > 0 else 0)
            tols.append(interval)
            tick = [(_min - interval) + i * interval for i in range(tick_length)]
            mins.append(min(tick))

        ticks.append(tick)

    print('Ticks for each task:')
    for task, tick in zip(tasks_list, ticks):
        print(f'{task}: {tick}\n')

    # Initialize radar plot parameters using the first task's ticks
    min_value = min(ticks[0])
    tolerance = ticks[0][1] - ticks[0][0]

    # Create a new figure
    fig = plt.figure(figsize=(10, 10))

    # Add the first task name to labels that need offset, if none is provided
    labels_away_axis = args.labels_away_axis if args.labels_away_axis else []
    labels_away_axis.append(tasks_list[0])

    # Instantiate the Radar object
    radar = Radar(
        figure=fig,
        title=tasks_list,
        labels=ticks,
        ax0_min=min_value,
        ax0_tol=tolerance,
        mins=mins,
        tols=tols,
        reg_tasks_list=reg_tasks_list,
        labels_away_axis=labels_away_axis
    )

    # Get a color palette for different models
    palette = sns.color_palette("husl", n_colors=num_models)

    # Plot each model on the radar
    for i, model in enumerate(models):
        model_df = source[source['model'] == model]
        model_acc = model_df['result'].tolist()
        radar.plot(model_acc, '-', lw=1, color=palette[i], label=model)

    # Create legend handles for the models
    model_legend_elements = []
    for label, color in zip(models, palette):
        model_legend_elements.append(Patch(facecolor=color, label=label))

    # Add the legend for the models
    radar.ax.legend(handles=model_legend_elements, loc='upper right', 
                    bbox_to_anchor=(1.3, 1.3), ncol=1)

    # Save the plot
    plt.savefig(args.save_filename, dpi=300)
    print(f'Saved radar plot at {args.save_filename}.')


if __name__ == '__main__':
    # Parse command-line arguments
    parser = get_args_parser()
    args = parser.parse_args()

    # Run main script
    main(args)
