import community as community_louvain
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import db_connection
import db_methods

def compute_co_occurrence(channel_categories_dict):
    """
    Computes the co-occurrence matrix for categories across all channels.
    
    Args:
    - channel_categories_dict: A dictionary where keys are channel_ids and values are lists of categories.
    
    Returns:
    - global_co_occurrence: A dictionary where keys are pairs of categories and values are their co-occurrence count.
    """
    global_co_occurrence = defaultdict(int)
    
    # Iterate through each channel and their associated categories
    for channel_id, categories in channel_categories_dict.items():
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                cat_1 = categories[i]
                cat_2 = categories[j]
                
                # Increment the co-occurrence count for each category pair
                global_co_occurrence[(cat_1, cat_2)] += 1
                #global_co_occurrence[(cat_2, cat_1)] += 1  # Make the relationship undirected
                
    return global_co_occurrence


def detect_communities(co_occurrence_matrix):
    """
    Performs community detection using the Louvain method on a co-occurrence matrix.
    
    Args:
    - co_occurrence_matrix: A dictionary containing pairs of categories and their co-occurrence count.
    
    Returns:
    - partition: A dictionary where keys are categories and values are community assignments.
    """
    # Create a graph from the co-occurrence matrix
    G = nx.Graph()
    
    # Add nodes and edges
    for (cat1, cat2), count in co_occurrence_matrix.items():
        G.add_edge(cat1, cat2, weight=count)
    
    # Apply Louvain method for community detection
    partition = community_louvain.best_partition(G)
    
    return partition



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
        if count > 1000:  # Ignore pairs with low co-occurrence counts
            G.add_edge(cat1, cat2, weight=count)
    
    # Use a spring layout to arrange the nodes. This layout places connected nodes closer to each other.
    pos = nx.spring_layout(G, k=0.15,weight='weight', iterations=20)
    
    # Prepare the plot
    plt.figure(figsize=(100, 100))
    
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



if __name__ == "__main__":
    print("\nConnection")
    collection = db_connection.connect_to_mongodb()
    print("\nDB query")
    channel_categories = db_methods.retrieve_categories_from_mongodb(collection)
    print("\nCo-occurrence matrix")
    co_occurrence_matrix = compute_co_occurrence(channel_categories)
    #print("\nCommunity detection")
    #partition = detect_communities(co_occurrence_matrix)
    print("\nCo-occurrence network")
    plot_community_network(co_occurrence_matrix)

