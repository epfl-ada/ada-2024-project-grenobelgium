import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import networkx as nx
import community as community_louvain
import pandas as pd
import plotly.graph_objects as go

# Method that plots the distribution of subscribers between categories
def plot_category_subscribers_distribution(channels, category_colors):
    total_number_of_subscribers = channels['subscribers_cc'].sum()
    df_dist = channels.groupby('category_cc')['subscribers_cc'].sum().reset_index().sort_values(ascending=False, by='subscribers_cc')
    df_dist.columns = ['category', 'subscribers']
    df_dist['subscribers'] = 100 * df_dist['subscribers'] / total_number_of_subscribers
    df_dist['color'] = df_dist['category'].map(category_colors)

    plt.figure(figsize=(7, 5))
    sns.barplot(y='category', x='subscribers', data=df_dist, hue='category', palette=category_colors, dodge=False)
    plt.grid(alpha=0.2)
    plt.xlabel('Percentage of Total Subscribers (%)')
    plt.ylabel('')
    plt.title('Distribution of Subscribers Across YouTube Channel Categories')

    plt.show()


#****************************************************************************************************************
# Method that plots the distribution of channels across categories
def plot_category_distribution(data, category_colors):
    distribution = data['category_cc'].value_counts(normalize=True).sort_values(ascending=False)*100
    df_dist = distribution.reset_index()
    df_dist.columns = ['category', 'percentage']
    df_dist['color'] = df_dist['category'].map(category_colors)

    plt.figure(figsize=(7, 5))
    sns.barplot(y='category', x='percentage', data=df_dist, hue='category', palette=category_colors, dodge=False)    
    plt.grid(alpha=0.2)
    plt.xlabel('Percentage of Total Channels (%)')
    plt.ylabel('')
    plt.title('Distribution of YouTube Channel Categories')

    plt.show()


#****************************************************************************************************************
# Function that plot the mean or total number of the feature for each category.
def plot_feature_by_category(data, feature, category_colors, agg_mode='mean'):

    aggregation_methods = {'mean': 'mean', 'total': 'sum'}
    
    df_dist = data.groupby('category_cc')[feature].agg(aggregation_methods[agg_mode]).reset_index()
    df_dist.columns = ['category', feature]
    df_dist = df_dist.sort_values(ascending=False, by=feature)
    df_dist['color'] = df_dist['category'].map(category_colors)

    plt.figure(figsize=(7, 5))
    sns.barplot(y='category', x=feature, data=df_dist, hue='category', palette=category_colors, dodge=False)

    if feature == 'videos_cc':
        feature = 'Videos'
    if feature == 'subscribers_cc':
        feature = 'Subscribers'

    if agg_mode == 'mean':
        agg_mode='Average'

    plt.grid(alpha=0.2)
    plt.xlabel(f'{agg_mode.capitalize()} number of {feature}')
    plt.ylabel('')
    plt.title(f'{agg_mode.capitalize()} number of {feature} in each category')

    plt.show()


#****************************************************************************************************************
    # Plot of the average join date by category
def plot_mean_join_date_by_category(data, category_colors):

    data['float_join_date'] = data['year'] + (data['month'].apply(lambda x:x-1)) * (1/12) + (data['day'].apply(lambda x:x-1)) *  (1/30)
    df_dist = data.groupby('category_cc')['float_join_date'].mean().reset_index()
    df_dist.columns = ['category', 'join_date']
    df_dist = df_dist.sort_values(ascending=False, by='join_date')
    df_dist['color'] = df_dist['category'].map(category_colors)

    plt.figure(figsize=(7, 5))
    sns.barplot(y='category', x='join_date', data=df_dist, hue='category', palette=category_colors, dodge=False)

    plt.grid(alpha=0.2)
    plt.xlim(2011, 2016)
    plt.xlabel('Average join date')
    plt.ylabel('')
    plt.title('Average join date for each category')

    plt.show()


#****************************************************************************************************************
# Plot of the ratio views/subscribers by category
def plot_ratio_views_subscribers(data, category_colors):
    data = data.copy()
        # Creation of the ratio (Views per videos / subscribers)
    data['ratio_views_subs'] = data['views'] / data['subscribers_cc']
    df_ratio = data.groupby('category_cc')['ratio_views_subs'].mean().reset_index()
    df_ratio = df_ratio.sort_values(by='ratio_views_subs', ascending=False)
        # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_ratio, y='category_cc', x='ratio_views_subs', hue='category_cc', palette=category_colors, dodge=False)
    plt.xlabel('Views per videos / Subscribers per Channel (Ratio)')
    plt.ylabel('')
    plt.title('Views per videos / Subscribers per Channel by Category')
    plt.grid(alpha=0.3)
    plt.show()


#****************************************************************************************************************
# Plot of the typical evolution in time of a given metric by category
def plot_monthly_metric_by_category(data, metric='delta_views'):
    
    data['month'] = data['datetime'].dt.to_period('M')
    
    if metric not in ['delta_views', 'delta_subs']:
        raise ValueError("Metric must be either 'delta_views' or 'delta_subs'")
    
    monthly_data = data.groupby(['category', 'month'])[metric].sum().reset_index()
    
    monthly_metric_pivot = monthly_data.pivot(index='month', columns='category', values=metric)

    title = "Total Monthly " + ("Views" if metric == 'delta_views' else "Subscriber Gains") + " by Category"
    ylabel = "Total " + ("Views" if metric == 'delta_views' else "Subscriber Gains")
    
    monthly_metric_pivot.plot(figsize=(12, 6), title=title)
    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()


#****************************************************************************************************************
# Plot of the channel counts by category
def plot_smoothed_monthly_channel_counts_by_category(data, window=3):

    data['join_date'] = pd.to_datetime(data['join_date'], errors='coerce')
    data['month'] = data['join_date'].dt.to_period('M')
    monthly_counts = data.groupby(['category_cc', 'month']).size().unstack(fill_value=0)
    
    monthly_counts_smoothed = monthly_counts.T.rolling(window=window, min_periods=1).mean()
    
    plt.figure(figsize=(12, 6))
    monthly_counts_smoothed.plot(kind='line', figsize=(12, 6), title=f"Number of channels created per month by category (moving average of {window} months)")
    plt.xlabel("Month")
    plt.ylabel("Number of channels created")
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()


#****************************************************************************************************************

# Given a co-occurrence matrix, this method plots the corresponding heatmap
def plot_co_occurrence_heatmap(co_occurrence_matrix):
    # Extract all unique categories from the co-occurrence dictionary
    categories = sorted(set([cat for pair in co_occurrence_matrix.keys() for cat in pair]))
    
    # Initialize a DataFrame with zeros for all category pairs
    df = pd.DataFrame(0, index=categories, columns=categories)
    
    # Fill in the DataFrame with the co-occurrence counts
    for (cat1, cat2), count in co_occurrence_matrix.items():
        df.loc[cat1, cat2] = count
        df.loc[cat2, cat1] = count  # Ensure symmetry
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=False, cmap="YlGnBu", square=True, cbar_kws={"shrink": 0.75})
    plt.title("Co-Occurrence Heatmap of Categories")
    plt.show()

#****************************************************************************************************************

# Method that plots the co-occurrence network
def plot_co_occurrence_network(co_occurrence_matrix):
    """
    Visualizes the co-occurrence network of categories. Categories that appear together often
    will be positioned closer together, and their co-occurrence is represented by edges.
    
    Args:
    - co_occurrence_matrix: A dictionary where the keys are pairs of categories and the values
                             are the number of times those categories co-occur.
    """
    # Create a graph from the co-occurrence matrix
    G = nx.Graph()
    
    # Add edges with weights from the co-occurrence matrix
    for (cat1, cat2), count in co_occurrence_matrix.items():
        if count > 5000:  # Ignore pairs with low co-occurrence counts
            G.add_edge(cat1, cat2, weight=count)
    
    # Use a spring layout to arrange the nodes. This layout places connected nodes closer to each other.
    pos = nx.spring_layout(G, k=0.18,weight='weight', iterations=1000)
    
    # Prepare the plot
    plt.figure(figsize=(12, 8))
    
    # Draw the nodes with color based on their degree (how many connections they have)
    node_size = [v * 100 for v in dict(G.degree()).values()]  # Node size based on degree
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="skyblue", alpha=0.7)
    
    # Draw edges with thickness based on the weight of the co-occurrence
    edge_width = [0.5 for u, v, d in G.edges(data=True)]  # Scale edge width by weight
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.5, edge_color="gray")
    
    # Draw labels for the categories
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")
    
    # Title and show plot
    plt.title("Co-Occurrence Network of Categories")
    plt.axis('off')  # Turn off the axis
    plt.show()


#****************************************************************************************************************
# Interactive co-occurrence network plot
def plot_interactive_co_occurrence_network(co_occurrence_matrix):
    # Create a graph from the co-occurrence matrix
    G = nx.Graph()
    for (cat1, cat2), count in co_occurrence_matrix.items():
        if count>5000:
            G.add_edge(cat1, cat2, weight=count)
    
    # Calculate positions using spring layout
    pos = nx.spring_layout(G, k=0.15, weight='weight', iterations=1000)
    
    # Define node sizes based on the degree (number of connections)
    node_sizes = [G.degree(node) * 20 for node in G.nodes()]
    
    # Set node colors based on communities
    community_map = community_louvain.best_partition(G, resolution=1.2)
    
    # Create Plotly edge traces
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    # Create Plotly node traces
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=[],
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
        )
    )
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        
        # Append node information for hover text
        node_trace['text'] += (f'{node}<br># of Connections: {G.degree(node)}',)
        node_trace['marker']['color'] += (community_map[node],)
    
    # Create the Plotly figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Co-Occurrence Network of Categories',
                        titlefont_size=16,
                        showlegend=False,
                        width=1200,
                        height=600,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        annotations=[dict(
                            text="Hover over nodes to see details",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    # Show the plot
    fig.show()

#****************************************************************************************************************
# Plot the distribution of a set of values
def plot_distribution(values, title, xlabel):
    # Create a seaborn style plot for better aesthetics
    sns.set(style="whitegrid")

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(values, kde=True, bins=10, color='skyblue', stat='density', linewidth=0)

    # Adding labels and title
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Density", fontsize=12)

    # Show the plot
    plt.show()

#****************************************************************************************************************
# Method that plots the QQ-plots for 4 candidate laws : Exponential, Truncated Normal Distribution, Beta, Gamma.
def qq_plots(values):
    # Convert the values to a list so we can use numpy easily
    data = list(values)

    # Scale the data to [0, 1] for fair comparison between the potential candidates for the generative law
    data_scaled = (np.array(data) - np.min(data)) / (np.max(data) - np.min(data))

    # Plot setup
    plt.figure(figsize=(14, 12))

    # 1. QQ-Plot for Exponential Distribution
    plt.subplot(2, 2, 1)
    stats.probplot(data_scaled, dist="expon", plot=plt)
    plt.title('QQ-Plot for Exponential Distribution')

    # 2. QQ-Plot for Truncated Normal Distribution
    a, b = (min(data_scaled) - np.mean(data_scaled)) / np.std(data_scaled), (max(data_scaled) - np.mean(data_scaled)) / np.std(data_scaled)
    params_truncnorm = stats.truncnorm.fit(data_scaled, a, b)
    plt.subplot(2, 2, 2)
    stats.probplot(data_scaled, dist="truncnorm", sparams=(a, b, params_truncnorm[0], params_truncnorm[1]), plot=plt)
    plt.title('QQ-Plot for Truncated Normal Distribution')

    # 3. QQ-Plot for Beta Distribution
    params_beta = stats.beta.fit(data_scaled)  # Fit the Beta distribution
    plt.subplot(2, 2, 3)
    stats.probplot(data_scaled, dist="beta", sparams=params_beta, plot=plt)
    plt.title('QQ-Plot for Beta Distribution')

    # 4. QQ-Plot for Gamma Distribution
    params_gamma = stats.gamma.fit(data_scaled)  # Fit the Gamma distribution
    plt.subplot(2, 2, 4)
    stats.probplot(data_scaled, dist="gamma", sparams=params_gamma, plot=plt)
    plt.title('QQ-Plot for Gamma Distribution')

    # Layout adjustments
    plt.tight_layout()
    plt.show()


#****************************************************************************************************************
# Plot information in the metrics dataframe (Log_Likelihood, QQ-MSE)
def plot_metrics_data(df_metrics):
    # Sort metrics for each plot
    df_metrics_loglik = df_metrics.sort_values(by='Log-Likelihood', ascending=False)
    df_metrics_mse = df_metrics.sort_values(by='QQ MSE', ascending=True)

    # Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Heatmap for Log-Likelihood
    sns.heatmap(df_metrics_loglik[['Log-Likelihood']], annot=True, cmap='YlGnBu', linewidths=0.5, ax=axes[0])
    axes[0].set_title('Log-Likelihood (Higher is Better)')
    axes[0].set_ylabel('')
    best_loglik_dist = df_metrics_loglik['Log-Likelihood'].idxmax()
    axes[0].annotate(f'Best: {best_loglik_dist}', xy=(0.5, -0.1), xycoords='axes fraction',
                    ha='center', va='center', fontsize=12, color='green')

    # Plot 2: Heatmap for QQ MSE
    sns.heatmap(df_metrics_mse[['QQ MSE']], annot=True, cmap='YlGnBu', linewidths=0.5, ax=axes[1])
    axes[1].set_title('QQ MSE (Lower is Better)')
    axes[1].set_ylabel('')
    best_mse_dist = df_metrics_mse['QQ MSE'].idxmin()
    axes[1].annotate(f'Best: {best_mse_dist}', xy=(0.5, -0.1), xycoords='axes fraction',
                    ha='center', va='center', fontsize=12, color='red')

    plt.suptitle('Comparison of Distribution Fit Metrics', fontsize=16)
    plt.tight_layout()
    plt.show()

#********************************************************************************************************************
# Plot of the subscribers distribution between the three diversity categories specifies in the given subscribers_dict
def plot_subscribers_comparison_with_boxes(subscribers_dict):
    """
    Plots the comparison of subscriber counts between the three categories (low, moderate, high),
    with a boxplot showing the distributions and no individual points.
    
    Args:
    - subscribers_dict: A dictionary where keys are 'low', 'moderate', 'high', 
                        and values are lists of subscriber counts.
    """
    
    # Convert the dictionary to a pandas DataFrame for easier plotting
    data = []
    for category, subscribers in subscribers_dict.items():
        data.extend(subscribers)
    
    categories = []
    for category, subscribers in subscribers_dict.items():
        categories.extend([category] * len(subscribers))
    
    df = pd.DataFrame({
        'Subscribers': data,
        'Category': categories
    })
    
    # Set the figure size
    plt.figure(figsize=(10, 6))
    
    # Create the boxplot with no individual points or caps
    sns.boxplot(x='Category', y='Subscribers', data=df, palette="Set3", showmeans=False, showcaps=False, showfliers=False)
    
    # Set plot title and labels
    plt.title('Comparison of Subscriber Counts Between Categories')
    plt.xlabel('Diversity Categories')
    plt.ylabel('Number of Subscribers')
    
    # Show the plot
    plt.show()

#****************************************************************************************************************
# Plot the mean growth of two given time series data
    # Method that computes the mean growth
def compute_mean_growth(cluster_data,interest_column):
    """
    Computes the mean growth across all channels for a given cluster.
    
    Args:
    - cluster_data: Dictionary where keys are channel_ids and values are lists of entries.
    
    Returns:
    - mean_growth: A DataFrame with the mean growth per aligned week.
    """
    # Create a DataFrame to hold all the data
    combined_df = pd.DataFrame()

    # Concatenate all channel data into a single DataFrame with aligned weeks
    for entries in cluster_data.values():
        df = pd.DataFrame(entries)
        combined_df = pd.concat([combined_df, df[['aligned_week', interest_column]]])

    # Calculate the mean smoothed views per aligned week
    mean_growth = combined_df.groupby('aligned_week')[interest_column].mean().reset_index()
    
    return mean_growth

    # Method that plots the figures side by side within a given time frame
def plot_mean_growth_comparison(column_of_interest, cluster1_data, cluster2_data, category1, category2, 
                                xlim_start=None, xlim_end=None, log_scale=False):
    """
    Plots the typical mean growth evolution for two clusters side by side.
    
    Args:
    - Column_of_interest: Viewership, subscriptions...
    - cluster1_data: Data for the first cluster.
    - cluster2_data: Data for the second cluster.
    - category1: Name of the first category.
    - category2: Name of the second category.
    - xlim_start: Start of the range for aligned weeks (optional).
    - xlim_end: End of the range for aligned weeks (optional).
    - log_scale: Boolean to use logarithmic scaling for the y-axis.
    """
    # Compute the mean growth for each cluster
    mean_growth1 = compute_mean_growth(cluster1_data,column_of_interest)
    mean_growth2 = compute_mean_growth(cluster2_data,column_of_interest)
    
    # Extract the data for plotting
    weeks1 = mean_growth1['aligned_week']
    growth1 = mean_growth1[column_of_interest]
    weeks2 = mean_growth2['aligned_week']
    growth2 = mean_growth2[column_of_interest]
    
    # Filter data based on the specified range
    if xlim_start is not None and xlim_end is not None:
        mask1 = (weeks1 >= xlim_start) & (weeks1 <= xlim_end)
        mask2 = (weeks2 >= xlim_start) & (weeks2 <= xlim_end)
        
        weeks1, growth1 = weeks1[mask1], growth1[mask1]
        weeks2, growth2 = weeks2[mask2], growth2[mask2]

    # Set up the figure and axes for side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot for Cluster 1
    axes[0].plot(weeks1, growth1, label=f"{category1} (Mean)", color='blue')
    axes[0].set_title(f"Growth Evolution: {category1}")
    axes[0].set_xlabel("Aligned Week")
    axes[0].set_ylabel(f"Smoothed Growth in {column_of_interest} (%)")
    axes[0].grid(True)
    if log_scale:
        axes[0].set_yscale('log')
    if xlim_start is not None and xlim_end is not None:
        axes[0].set_xlim(xlim_start, xlim_end)
        axes[0].set_ylim(growth1.min() * 0.9, growth1.max() * 1.1)  # Adjust y-axis based on data range

    # Plot for Cluster 2
    axes[1].plot(weeks2, growth2, label=f"{category2} (Mean)", color='orange', linestyle='--')
    axes[1].set_title(f"Growth Evolution: {category2}")
    axes[1].set_xlabel("Aligned Week")
    axes[1].grid(True)
    if log_scale:
        axes[1].set_yscale('log')
    if xlim_start is not None and xlim_end is not None:
        axes[1].set_xlim(xlim_start, xlim_end)
        axes[1].set_ylim(growth2.min() * 0.9, growth2.max() * 1.1)  # Adjust y-axis based on data range

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

#****************************************************************************************************************

category_colors = {
    'Film and Animation': '#4C72B0',   # Soft Blue
    'Entertainment': '#DD8452',         # Warm Orange
    'Music': '#55A868',                 # Soft Green
    'Comedy': '#C44E52',                # Deep Red
    'Gaming': '#8172B2',                # Muted Purple
    'Science & Technology': '#8C564B',  # Warm Brown
    'Sports': '#E377C2',                # Vibrant Pink
    'Education': '#7F7F7F',             # Neutral Grey
    'People & Blogs': '#BCBD22',        # Fresh Yellow-green
    'Nonprofits & Activism': '#17BECF', # Cool Light Blue
    'Howto & Style': '#FFB74D',         # Light Orange
    'News & Politics': '#F28E2B',       # Bold Red-Orange
    'Travel & Events': '#C5B0D5',       # Lavender Purple
    'Autos & Vehicles': '#9E7C4D',      # Earthy Tan
    'Pets & Animals': '#F7B6D2'         # Soft Light Pink
}