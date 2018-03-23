import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np


def plot_log(
    log_file_path, x_field, y_field, x_label, y_label, title,
    smooth_factor=None, to_accuracy=None,
        color=None, transparent=None, dpi=None, min_x=None, max_x=None):
    def _running_average_smooth(y, window_size):
        kernel = np.ones(window_size) / window_size
        y_pad = np.lib.pad(y, (window_size, ), 'edge')
        y_smooth = np.convolve(y_pad, kernel, mode='same')
        return y_smooth[window_size:-window_size]

    x = list()
    y = list()

    print('[plot_log]\tReading CSV log file ...')

    with open(log_file_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if min_x >= 0 and int(row[x_field].strip()) < min_x:
                continue
            if max_x >= 0 and int(row[x_field].strip()) > max_x:
                break
            if row[y_field].strip().lower() != 'nan':
                try:
                    x.append(int(row[x_field].strip()))
                except ValueError:
                    raise ValueError('x-axis value error at line {}: {}'.format(
                        i + 1, row[x_field].strip()
                    ))
                try:
                    y.append(
                        1.0 - float(row[y_field].strip()) if to_accuracy else
                        float(row[y_field].strip())
                    )
                except ValueError:
                    raise ValueError('y-axis value error at line {}: {}'.format(
                        i + 1, row[y_field].strip()
                    ))

    x = np.asarray(x)
    y = np.asarray(y)

    print('[plot_log]\tPlotting data ...')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.plot(x, y, alpha=0.2, color=color)
    plt.plot(
        x, _running_average_smooth(y, smooth_factor), color=color, linewidth=2
    )

    plt.grid()

    print('[plot_log]\tSaving figure ...')

    plt.savefig(
        '{}.png'.format(title.replace(' ', '_')),
        bbox_inches='tight', transparent=transparent, dpi=dpi
    )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'plot':
        parser = argparse.ArgumentParser(
            description='Plot CSV log file.'
        )
        POSITIONALS = [
            ("log_file_path", "path to log file"),
            ("x_field", "field name of x-axis in CSV"),
            ("y_field", "field name of y-axis in CSV"),
            ("x_label", "label for x-axis in figure"),
            ("y_label", "label for y-axis in figure"),
            ("title", "title for figure")
        ]
        OPTIONALS = [
            ("color", "color of line", "yellowgreen", str),
            ("transparent", "transparent background", False, bool),
            ("dpi", "DPI of saved figure file", 500, int),
            ("smooth_factor", "smooth factor", 9, int),
            ("to_accuracy", "convert to accuracy and plot", False, bool),
            ("min_x", "min x-value to plot", -1, int),
            ("max_x", "max x-value to plot", -1, int)
        ]
        for arg, h in POSITIONALS:
            parser.add_argument(arg, help=h)
        for arg, h, de, t in OPTIONALS:
            parser.add_argument(arg, help=h, type=t, default=de)

        args = vars(parser.parse_args(sys.argv[2:]))

        positionals = []
        for a, _ in POSITIONALS:
            positionals.append(args[a])
            del args[a]

        plot_log(*positionals, **args)
