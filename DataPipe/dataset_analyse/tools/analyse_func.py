"""
Version: Dec 18 2024 11:30
"""
# Import required libraries
import pandas as pd  # Data processing and analysis
import matplotlib.pyplot as plt  # Plotting tools
import seaborn as sns  # Advanced statistical plotting tools
import os  # File path operations
from sklearn.preprocessing import MinMaxScaler  # Data normalization tools
import textwrap  # Text auto-wrapping tools

# ---------------------------
# Function: Plot correlation matrix heatmap
# ---------------------------
def plot_corr_matrix(df, selected_cols, dataset_name, save_path='./', figsize=(14.4, 12), dpi=300):
    """
    Plot the correlation matrix heatmap for specified columns.

    Args:
        df (DataFrame): The dataset.
        selected_cols (list): Columns to calculate correlations for.
        dataset_name (str): Dataset name for labeling the plot.
        save_path (str): Path to save the plot.
        figsize (tuple): Size of the plot.
        dpi (int): Plot resolution.
    """
    data = df  # Assign dataset
    filtered_data = data[selected_cols].copy()  # Retain only specified columns, create a copy

    # Check for text columns and map them to numerical values
    columns_with_text_options = [col for col in filtered_data.columns if filtered_data[col].dtype == 'object']
    for col in columns_with_text_options:
        unique_values = filtered_data[col].dropna().unique()  # Get unique values
        value_mapping = {val: idx for idx, val in enumerate(unique_values)}  # Create mapping dictionary
        filtered_data[col] = filtered_data[col].map(value_mapping)  # Replace text values with numerical values

    # Data normalization (scale values between 0 and 1)
    scaler = MinMaxScaler()  
    normalized_data = pd.DataFrame(scaler.fit_transform(filtered_data), columns=filtered_data.columns)

    # Calculate the correlation matrix
    correlation_matrix = normalized_data.corr()

    # Plot the heatmap
    plt.figure(figsize=figsize)  # Set figure size
    ax = sns.heatmap(
        correlation_matrix,  # Data
        cmap='coolwarm',  # Color scheme
        annot=True,  # Display correlation coefficients
        fmt=".2f",  # Format to two decimal places
        cbar=True,  # Display color bar
        xticklabels=correlation_matrix.columns,  # X-axis labels
        yticklabels=correlation_matrix.columns,  # Y-axis labels
        annot_kws={"size": 12},  # Annotation font size
        vmin=-1, vmax=1  # Value range
    )

    # Remove gridlines
    # ax.set_xticks([])  # Remove X-axis gridlines
    # ax.set_yticks([])  # Remove Y-axis gridlines

    plt.xticks(rotation=45, ha='right', fontsize=12)  # Adjust X-axis label rotation and font size
    plt.yticks(fontsize=12)  # Adjust Y-axis label font size
    plt.title(f'Correlation Matrix of {dataset_name}', fontsize=20)  # Add title
    plt.tight_layout()  # Adjust layout

    # Save the plot
    save_file_name = f'CorrelationPlot_{dataset_name}.png'  # File name
    plt.savefig(os.path.join(save_path, save_file_name), bbox_inches='tight', pad_inches=0.8, dpi=dpi)
    plt.close()  # Display the plot

# ---------------------------
# Function: Plot proportion of non-missing values
# ---------------------------
def draw_exist_value_fig(df, dataset_name, selected_cols=None, save_path='./', figsize=(10, 6), dpi=300):
    """
    Plot a bar chart showing the proportion of non-missing values in the dataset.

    Args:
        df (DataFrame): The dataset.
        dataset_name (str): Dataset name for labeling.
        selected_cols (list): Columns to plot.
        save_path (str): Path to save the plot.
        figsize (tuple): Size of the plot.
        dpi (int): Plot resolution.
    """
    data = df  # Assign dataset
    filtered_data = data[selected_cols] if selected_cols is not None else data  # Filter specified columns
    non_missing_proportions = filtered_data.notna().mean().sort_values(ascending=False)  # Calculate and sort proportions

    # Plot bar chart
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        y=non_missing_proportions.index,  # Y-axis column names
        width=non_missing_proportions.values,  # X-axis proportions
        color='steelblue',  # Bar color
        edgecolor='black'  # Bar edge color
    )

    # Add percentage labels
    for bar, value in zip(bars, non_missing_proportions.values):
        count = int(value * len(df))  # Calculate count of non-missing values
        ax.text(
            value + 0.02,  # X position for labels
            bar.get_y() + bar.get_height() / 2,  # Y position for labels
            f'{value * 100:.1f}%({count})',  # Display percentage
            va='center', fontsize=10  # Vertically center align, set font size
        )

    ax.set_xlabel('Proportion of Non-Missing Values (%)', fontsize=12)  # Set X-axis label
    ax.set_ylabel('Column Names', fontsize=12)  # Set Y-axis label
    ax.set_title(f'Proportion of Existing Values for {dataset_name}', fontsize=14)  # Set title
    ax.invert_yaxis()  # Invert Y-axis

    ax.grid(False)  # Disable gridlines

    plt.tight_layout()  # Adjust layout

    # Save the plot
    save_file_name = f'exist_values_{dataset_name}.png'  # File name
    plt.savefig(os.path.join(save_path, save_file_name), bbox_inches='tight', dpi=dpi)
    plt.close()

    
# ---------------------------
# Function: Plot count and boxplots
# ---------------------------
def plot_count_and_boxplot(df, x_var, y_var, plot_name, save_path='./', sel_palette='Blues', dpi=300):
    """
    Plot count and boxplots for specified columns.

    Args:
        df (DataFrame): The dataset.
        x_var (str): X-axis variable.
        y_var (str): Y-axis variable.
        plot_name (str): Name of the plot.
        save_path (str): Path to save the plot.
        sel_palette (str): Color palette.
        dpi (int): Plot resolution.
    """
    data = df  # Assign dataset
    filtered_data = data[[x_var, y_var]].dropna()  # Filter specified columns and drop missing values

    sns.set(style='whitegrid', font_scale=1.2)  # Set style
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1, 4]})

    # Count plot
    sns.countplot(x=x_var, data=filtered_data, ax=ax1, hue=x_var, palette=sel_palette, legend=False)
    ax1.set_title(f'Count of {x_var}', fontsize=14)  # Set title
    ax1.set_ylim(0, max([p.get_height() for p in ax1.patches]) * 1.2)  # Set Y-axis range
    for p in ax1.patches:  # Add count labels
        ax1.annotate(
            format(p.get_height(), '.0f'),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 10), textcoords='offset points',
            fontweight='bold'
        )

    # Boxplot
    sns.boxplot(x=x_var, y=y_var, data=filtered_data, ax=ax2, hue=x_var, palette=sel_palette, dodge=False)
    sns.stripplot(x=x_var, y=y_var, data=filtered_data, ax=ax2, color='#FFA07A', size=4, alpha=0.8, jitter=True)
    ax2.set_title(f'{y_var} by {x_var}', fontsize=14)  # Set title

    # Save the plot
    save_file_name = f'combi_countBoxplot_{x_var}_{y_var}_{plot_name}.png'  # File name
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, save_file_name), dpi=dpi)
    plt.close()

# ---------------------------
# Function: Plot pairplot
# ---------------------------
def plot_pairplot(df, selected_columns, hue_var, dataset_name, save_path='./', sel_palette=['#66c2a5', '#fc8d62'], dpi=300):
    """
    Plot pairwise relationships in the dataset.

    Args:
        df (DataFrame): The dataset.
        selected_columns (list): Columns to include in the pairplot.
        hue_var (str): Column for coloring points by category.
        dataset_name (str): Name of the dataset.
        save_path (str): Path to save the plot.
        sel_palette (list): Color palette.
        dpi (int): Plot resolution.
    """
    data = df  # Assign dataset
    filtered_data = data[selected_columns].dropna()  # Filter selected columns and drop missing values

    # Wrap column names for better display in the plot
    wrapped_labels = {column: textwrap.fill(column, width=15) for column in selected_columns[1:]}

    # Rename columns to wrapped names
    filtered_data.rename(columns=wrapped_labels, inplace=True)

    # Set Seaborn style and font scale
    sns.set(style='ticks', font_scale=0.8)

    # Create pairplot
    pairplot = sns.pairplot(
        filtered_data,                          # Dataset
        vars=list(wrapped_labels.values()),    # Columns to plot
        hue=hue_var,                           # Color by category
        palette=sel_palette,                   # Color palette
        diag_kind='kde',                       # Use kernel density estimation for diagonal plots
        plot_kws={'alpha': 0.7, 's': 50},      # Scatter plot transparency and size
        diag_kws={'fill': True}                # Fill diagonal plots
    )
    """
    this part no longer in use
    # Adjust layout
    pairplot.fig.subplots_adjust(top=0.85, wspace=0.25, hspace=0.25)

    # Adjust axes range and ticks
    for ax_row in pairplot.axes:
        for ax in ax_row:
            if ax is not None:
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                x_buffer = (x_max - x_min) * 0.05
                y_buffer = (y_max - y_min) * 0.2
                ax.set_xlim(0, x_max + x_buffer)
                ax.set_ylim(0, y_max + y_buffer)
                ax.tick_params(axis='both', which='major', labelsize=10, direction='in', length=5)

    sns.despine(trim=True)  # Remove top and right borders
    """
    # title
    title = f"Pairplot of {dataset_name}"
    pairplot.fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)  

    # Save the plot
    save_file_name = f'pairplot_{hue_var}_{len(selected_columns)}.png'  # File name
    pairplot.savefig(os.path.join(save_path, save_file_name), dpi=dpi, bbox_inches='tight', pad_inches=0.8)
    plt.close()
# ---------------------------
# Function: Plot classification task analysis
# ---------------------------
def plot_task_analyse_cls(df, selected_cols, dataset_name, save_path='./', sel_palette='Blues', dpi=300):
    """
    Plot single or multi-column analysis for classification tasks:
    - Add a 20% buffer on top of bar charts.
    - Narrow the width of all bar charts.
    - Plot count charts for categorical columns or boxplots for continuous columns.
    - Add scatter points to boxplots to show data distribution.

    Args:
        df (DataFrame): Processed dataset.
        selected_cols (str or list): Column(s) to analyze.
        dataset_name (str): Dataset name for labeling and saving files.
        save_path (str): Path to save the plot.
        sel_palette (str): Color palette for plots.
        dpi (int): Plot resolution.
    """
    for col in selected_cols:
        # Set Seaborn style
        sns.set(style='whitegrid', font_scale=1.2)
        
        # If column is categorical, plot count chart
        fig, ax = plt.subplots(figsize=(8, 8.2))
        sns.countplot(
            x=col,  # Column to plot
            data=df,  # Dataset
            hue=col,  # Color by category
            palette=sel_palette,  # Color palette
            legend=False,  # Disable legend
            ax=ax  # Use single axis object
        )

        ax.set_title(f'{col} count', fontsize=14)  # Set title
        ax.set_xlabel(col, fontsize=12)  # Set X-axis label
        ax.set_ylabel('Count', fontsize=12)  # Set Y-axis label

        # Add annotations for count
        for p in ax.patches:
            ax.annotate(
                format(p.get_height(), '.0f'),  # Format as integer
                (p.get_x() + p.get_width() / 2., p.get_height()),  # Position for annotation
                ha='center', va='center',
                xytext=(0, 10), textcoords='offset points',  # Adjust position
                fontsize=10, fontweight='bold'
            )

        plt.xticks(rotation=45, ha='right', fontsize=9.5)  # Adjust X-axis labels

        # Save plot
        save_file_name = f'countplot_{col}_{dataset_name}.png'
        os.makedirs(save_path, exist_ok=True)  # Ensure save path exists
        plt.tight_layout()  # Adjust layout
        plt.savefig(os.path.join(save_path, save_file_name), dpi=dpi, bbox_inches='tight')
        plt.close()  # Close figure to free memory

# ---------------------------
# Function: Plot regression task analysis
# ---------------------------
def plot_task_analyse_reg(df, selected_cols, dataset_name, save_path='./', sel_palette='Blues', dpi=300):
    """
    Plot analysis for continuous variables in regression tasks.

    Args:
        df (DataFrame): The dataset.
        selected_cols (list): List of columns to analyze.
        dataset_name (str): Dataset name for labeling and saving files.
        save_path (str): Path to save the plot.
        sel_palette (str): Color palette for plots.
        dpi (int): Plot resolution.
    """
    for col in selected_cols:
        # Plot boxplot for continuous variables
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(
            y=col,  # Column to plot
            data=df,  # Dataset
            orient='v',  # Vertical orientation
            color='#4884AF',
            ax=ax,  # Use single axis object
            dodge=False  # Consistent layout
        )

        sns.stripplot(
            y=col,  # Add scatter points
            data=df,
            color='#FFA07A',  # Scatter point color
            size=4,  # Point size
            alpha=0.8,  # Transparency
            jitter=True,  # Add jitter to avoid overlap
            ax=ax  # Use single axis object
        )

        ax.set_title(f'{col} distribution', fontsize=14)  # Set title
        ax.set_xlabel('', fontsize=12)  # No X-axis label
        ax.set_ylabel(col, fontsize=12)  # Set Y-axis label

        # Add mean and median lines
        mean_val = df[col].mean()  # Calculate mean
        median_val = df[col].median()  # Calculate median
        ax.axhline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')  # Add mean line
        ax.axhline(median_val, color='blue', linestyle='--', label=f'Median: {median_val:.2f}')  # Add median line
        ax.legend(loc='upper right', fontsize=10)  # Add legend

        # Save plot
        save_file_name = f'boxplot_{col}_{dataset_name}.png'
        os.makedirs(save_path, exist_ok=True)  # Ensure save path exists
        plt.tight_layout()  # Adjust layout
        plt.savefig(os.path.join(save_path, save_file_name), dpi=dpi, bbox_inches='tight')
        plt.close()  # Close figure to free memory
