# Cross-Category Channels Performance: A Deep Dive into Content Diversity on YouTube

## Abstract

This project investigates how content diversity within YouTube channels influences audience growth, engagement, and loyalty by analyzing the YouNiverse dataset. By clustering channels based on the variety of content they produce, we aim to compare the performance of diverse-content channels (those covering multiple topics) with single-topic channels. We expect that while channels with mixed content will attract broader audiences and show higher growth, specialized channels might foster stronger viewer loyalty. The study will contribute to understanding the trade-offs in content variety and explore whether there’s a tipping point where excessive diversity begins to fragment audiences.

## Research Questions

1. Do YouTube channels with diverse content attract larger audiences and higher growth rates than those focused on a single topic?
2. How does content diversity affect viewer engagement and loyalty?
3. Is there a tipping point in content variety where audience fragmentation outweighs engagement benefits?
4. Are there any relationships between categories?
5. Which combination of categories is most effective for audience growth?

## Data


The **YouNiverse dataset**, which includes metadata for over 136,000 English-language YouTube channels and 72.9 million videos, provides a rich basis for exploring audience engagement and growth patterns. The dataset’s channel-level time-series data on weekly subscriber and view counts allows us to observe and compare growth trends over time. It contains channels with more than 10k subscribers and at least 10 videos.

#### Composition of the dataset

- **Channels dataset**: Contains information about YouTube channels.  
- **TimeSeries dataset**: Tracks the weekly evolution of metrics for each channel.  
- **Video metadata dataset**: Contains metadata for every video from the YouTube channels.

> **Note**: Since the dataset is large, it will not be uploaded to GitHub.

Currently, no additional datasets are planned.

## Project Structure

```plaintext
YouTube Data Analysis/
├── src/                     # Source code directory
│   ├── database/            # Database-related files
│   │   ├── db_connection.py 
│   │   ├── db_methods.py
│   │   └── config.yaml
│   ├── data_cleaning/       # Data cleaning file and notebook
│   │   └── data_cleaning.ipynb      
│   ├── plot_generation.py 
│   └── methods.py           
├── tests/                   # tests directory
│   └── (empty)
├── results.ipynb            # Notebook for exploring and visualizing results
├── README.md                
└── requirements.txt
```
$$\\$$

- **`db_connection.py`**: Code to make the connection with the mongodb database.
- **`db_methods.py`**: Methods to work with the mongodb database.
- **`config.yaml`**: configuration file for the connection with the database
$$\\$$
- **`data_cleaning.ipynb`**: This notebook shows how we clean the data.
$$\\$$
- **`plot_generation.py`**: Methods to display graphics.
- **`methods.py`**: Methods used to do the analysis.
$$\\$$
- **`results.ipynb`**: Complet analysis of the data.


### Summary of our work

The data cleaning process involves handling missing values, correcting data types, and extracting useful features. Missing values in key columns are either dropped or filled, and any incorrectly formatted columns are recalculated or converted to appropriate types. Date columns are transformed into a consistent datetime format, and additional temporal features like year, month, and day are created. These preprocessing steps ensure the data is clean and ready for analysis.

Our analysis aims to extract meaningful insights from the datasets, including a global comparison of categories based on views, subscribers, and video counts to identify trends and patterns. We also explore community detection to reveal clusters of frequently co-occurring categories, shedding light on content ecosystems and diversification. Additionally, we propose a channel diversity metric to quantify content variety within channels and analyze growth dynamics across clusters using time series data to highlight differences in viewership and subscriber trends.


#### Data Storage Approach

We use **MongoDB**, a NoSQL database, to manage our large-scale JSONL data efficiently. Its flexible schema and indexing capabilities enable seamless querying of subsets by categories, channels, or time ranges, without the limitations of traditional relational databases. This approach allows for scalable, dynamic analysis of millions of documents without memory constraints.




## Proposed Timeline

Investigate the possibility of predicting the category using the time-series evolution of a channel.

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

