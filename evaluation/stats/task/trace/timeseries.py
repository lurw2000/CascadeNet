"""
This script plots the packet rate of different datasets with dataset-specific visualization settings.
"""
import matplotlib.pyplot as plt
import os
import numpy as np
import nta.utils.vis as vis
from evaluation.stats.util.argparser import *

"""
Dataset-specific Configuration including color mappings
"""

COLORS_DICT = {**vis.COLORS_DICT, "raw": "tab:gray", "real": "tab:gray"}
DATASET_CONFIGS = {
    'caida': {
        'max_y_value': 1e6,
        'y_limit_min': -1e4,
        'y_unit_scaling': 1e6,
        'y_label': 'Packet Rate (Mpps)',
    },
    'ca': {
        'max_y_value': 5e3,
        'y_limit_min': -10,
        'y_unit_scaling': 1e3,
        'y_label': 'Packet Rate (Kpps)',
    },
    'dc': {
        'max_y_value': 2e4,
        'y_limit_min': -10,
        'y_unit_scaling': 1e3,
        'y_label': 'Packet Rate (Kpps)',
    },
    'ton_iot': {
        'max_y_value': 3e2,
        'y_limit_min': -10,
        'y_unit_scaling': 1e3,
        'y_label': 'Packet Rate (Kpps)',
    }
}

"""
General Configuration
"""
time_point_count = 100

flow_filter = {
    "flowsize_range": None,
    "flowDurationRatio_range": None,
    "nonzeroIntervalCount_range": None,
    "maxPacketrate_range": None,
}

# Style configuration
STYLE_CONFIG = {
    'default': {
        'label_size': 28,
        'title_size': 20,
        'tick_size': 28,
        'legend_size': 18,
        'figsize': (12, 6)
    },
    'many_subplots': {
        'label_size': 2,
        'title_size': 2,
        'tick_size': 2,
        'legend_size': 4,
        'figsize': (12, 6)
    }
}

def get_style_config(label_count):
    """Return appropriate style configuration based on number of labels."""
    return STYLE_CONFIG['many_subplots'] if label_count > 9 else STYLE_CONFIG['default']

def setup_save_config(project_root, folder, dataset):
    """Create save configuration dictionary."""
    if test:
        return {
            "folder": os.path.join(project_root, "test_result", "evaluation"),
            "filename": "trace_packet_rate_test",
            "format": "pdf",
        }
    else:
        return {
            "folder": os.path.join(project_root, "result", "evaluation", "stats", folder, dataset),
            "filename": f"trace_packet_rate_{dataset}",
            "format": "pdf",
        }

def get_line_style(label, index):
    """Get line style with correct color based on label."""
    # Start with the default line style
    style = vis.generate_line_style(index)
    
    # Override the color if the label is in COLORS_DICT
    if label in COLORS_DICT:
        style['color'] = COLORS_DICT[label]
    
    return style

def format_axis(ax, dataset_config, style_config):
    """Apply formatting to a single axis."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits and labels
    ax.set_ylim(dataset_config['y_limit_min'], dataset_config['max_y_value'])
    ax.set_xlabel("Time (s)", fontsize=style_config['label_size'])
    ax.set_ylabel(dataset_config['y_label'], fontsize=style_config['label_size'])
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=style_config['tick_size'])
    
    # Update y-axis formatter
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x/dataset_config["y_unit_scaling"]:g}')
    )
    
    # Set legend
    ax.legend(prop={'size': style_config['legend_size']})

def main():
    # Get project root
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(script_path)))))
    
    # Get style configuration
    style_config = get_style_config(len(label2path))
    
    # Get dataset-specific configuration
    dataset_config = DATASET_CONFIGS.get(dataset, DATASET_CONFIGS['caida'])  # Default to CAIDA config
    
    # Setup save configuration
    save_config = setup_save_config(project_root, folder, dataset)
    
    # Create figure and axes
    fig, axes = vis.subplots(1, 1, bp=3, sharex=True, sharey=True)
    fig.set_figwidth(style_config['figsize'][0])
    fig.set_figheight(style_config['figsize'][1])
    
    # Plot raw trace
    vis.plot_trace_level_packet_rate(
        label2path["raw"],
        axes=axes,
        mode=("time_point_count", time_point_count),
        label="real",
        **get_line_style("raw", 0),
        set_xlim=True,
        **flow_filter,
        read_csv_kwargs={"verbose": False, "need_divide_1e6": "auto"},
    )
    
    # Plot other traces
    for i, label in enumerate(label2path):
        if label == "raw" or label == "CTGAN":
            continue
            
        ax = axes[0]
        vis.plot_trace_level_packet_rate(
            label2path[label],
            ax=ax,
            mode=("time_point_count", time_point_count),
            label=label,
            alpha=0.7,
            **get_line_style(label, i),
            set_xlim=False,
            **flow_filter,
            read_csv_kwargs={"verbose": False, "need_divide_1e6": "auto"},
        )
    
    # Format all axes
    for ax in axes:
        format_axis(ax, dataset_config, style_config)
    
    plt.tight_layout()
    
    if save_figure:
        vis.savefig(fig, save_config)
    
    if show_figure:
        fig.canvas.manager.set_window_title(
            f"{dataset} - {os.path.basename(__file__).split('.')[0]}"
        )
        plt.show()

if __name__ == "__main__":
    main()