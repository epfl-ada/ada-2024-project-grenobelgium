# Cross-Category Channels Performance: A Deep Dive into Content Diversity on YouTube

## Abstract

This project investigates how content diversity within YouTube channels influences audience growth, engagement, and loyalty by analysing the YouNiverse dataset. By clustering channels based on the variety of content they produce, we aim to compare the performance of diverse-content channels (those covering multiple topics) with single-topic channels. We expect that while channels with mixed content will attract broader audiences and show higher growth, specialised channels might foster stronger viewer loyalty. The study will contribute to understanding the trade-offs in content variety and explore whether there’s a tipping point where excessive diversity begins to fragment audiences.

## Research Questions

1. Do YouTube channels with diverse content attract larger audiences and higher growth rates than those focused on a single topic?
2. How does content diversity affect viewer engagement and loyalty, especially within niche audiences?
3. Is there a tipping point in content variety where audience fragmentation outweighs engagement benefits?

## Data

### Primary Dataset
The YouNiverse dataset, which includes metadata for over 136,000 English-language YouTube channels and 72.9 million videos, provides a rich basis for exploring audience engagement and growth patterns. The dataset’s channel-level time-series data on weekly subscriber and view counts will allow us to observe and compare growth trends over time.

### Additional Datasets
Currently, no additional datasets are planned. However, we are open to integrating external data sources if they provide useful context on broader YouTube or social media trends.

## Methods

### Data Exploration and Initial Analysis
The project pipeline begins with the exploration of the YouNiverse dataset to ensure data handling feasibility and clarity on variable distributions, missing values, and potential correlations. Initial data exploration is conducted in three main Jupyter notebooks:
- **`metadata_exploration.ipynb`**: Analyses channel metadata, identifying key fields for clustering (e.g., channel category and video types).
- **`channels_timeseries_exploration.ipynb`**: Investigates subscriber and view count trajectories, focusing on growth trends.
- **`exploration.ipynb`**: A broader analysis notebook to refine data transformations and feature engineering for clustering.

### Clustering Channels by Content Variety
Using clustering techniques, we group channels based on the diversity of their content. Channels are classified based on their dominant video categories (e.g., entertainment, gaming, vlogs), with clustering helping to identify those with varied versus single-topic content. Clustering methods under consideration include k-means and hierarchical clustering, with metrics derived from channel metadata.

### Time-Series Analysis for Growth Patterns
After clustering, we conduct time-series analysis on subscriber and view counts to evaluate growth patterns within each group. By applying rolling averages and growth rate calculations, we aim to measure long-term audience trends and identify factors tied to either growth or stagnation across diverse and specialised channels.

### Community Detection (script: `community_detection.py`)
For a more granular view of content interaction, we apply community detection algorithms to explore interconnected topics within channels. This analysis will help reveal any intra-cluster consistency or variation in content preferences, adding depth to our understanding of audience fragmentation versus engagement.

### Database Connection and Management
The project relies on modular scripts (`db_connection.py` and `db_methods.py`) for efficient database interactions, ensuring streamlined access and manipulation of the YouNiverse data. Configuration details are managed through `config.yaml` for easy adjustments.

## Proposed Timeline

- **Week 1**: Dataset exploration, initial clustering analysis, and refinement of data pipeline (Milestone P2 submission).
- **Week 2**: Complete clustering and validate growth trends through time-series analysis.
- **Week 3**: Apply community detection and deepen analysis on audience segmentation.
- **Week 4**: Synthesis of findings, begin drafting visualisations for data storytelling.
- **Week 5**: Finalise the project repository, complete the report and visualisations.

## Organisation within the Team

- **Data Exploration and Preprocessing**: Team member A and B
- **Clustering and Time-Series Analysis**: Team member C
- **Community Detection and Audience Segmentation**: Team member D
- **Documentation and Visualisations**: Team member E

## Questions for TAs

- Are there specific metrics or statistical models you recommend for identifying potential tipping points in content variety versus audience engagement?