# Unveiling the secrets behind Youtube categories

## Link to data story
https://epfl-ada.github.io/ada-2024-project-grenobelgium/


If you wish to run the project, make sure you set up your MongoDB database with the expected collection names.

In order to get all the requirements need for the project, run the following command :

```pip install -r requirements.txt```

## Abstract

This project investigates how content diversity within YouTube channels influences audience growth, engagement, and loyalty by analyzing the YouNiverse dataset. By classifying channels based on the variety of content they produce, we aim to compare the performance of diverse-content channels (those covering multiple topics) with single-topic channels. We expect that while channels with mixed content will attract broader audiences and show higher growth, specialized channels might foster stronger viewer loyalty. The study will contribute to understanding the trade-offs in content variety and explore whether there’s a tipping point where excessive diversity begins to fragment audiences.

## Research Questions

1. Do YouTube channels with diverse content attract larger audiences and higher growth rates than those focused on a single topic?
2. How does content diversity affect viewer engagement and loyalty?
3. Is there a tipping point in content variety where audience fragmentation outweighs engagement benefits?
4. Are there any relationships between categories?
5. Which combination of categories is most effective for audience growth?
6. Is it possible to predict the categorical cluster of a channel giving timeseries data ?

## Data

The **YouNiverse dataset**, which includes metadata for over 136,000 English-language YouTube channels and 72.9 million videos, provides a rich basis for exploring audience engagement and growth patterns. The dataset’s channel-level time-series data on weekly subscriber and view counts allows us to observe and compare growth trends over time. It contains channels with more than 10k subscribers and at least 10 videos.

#### Composition of the dataset

- **Channels dataset**: Contains information about YouTube channels.
- **TimeSeries dataset**: Tracks the weekly evolution of metrics for each channel.
- **Video metadata dataset**: Contains metadata for every video from the YouTube channels.

> **Note**: Since the dataset is large, it will not be uploaded to GitHub.

No additional datasets were used.

## Project Structure

```plaintext
YouTube Data Analysis/
├── images/
│   └── cooccurrence_network.png
├── src/                          # Source code directory
│   ├── database/                 # Database-related files
│   │   ├── db_connection.py
│   │   ├── db_methods.py
│   │   └── config.yaml
│   ├── data_cleaning/            # Data cleaning file and notebook
│   │   └── data_cleaning.ipynb
│   ├── prediction/            # Prediction part utils
│   │   └── prediction_utils.py
│   ├── plot_generation.py      # Methods that were used to generate the different plots
│   └── community_detection.py    # Community detection functions
├── results.ipynb                 # Notebook for exploring and visualizing results
├── README.md
├── requirements.txt
└── .gitignore
```


- **`db_connection.py`**: Code to make the connection with the mongodb database.
- **`db_methods.py`**: Methods to work with the mongodb database.
- **`prediction_utils.py`**: Preprocessing, Feature Engineering and Benchmark functions for the prediction part.
- **`config.yaml`**: configuration file for the connection with the database

- **`data_cleaning.ipynb`**: This notebook shows how we clean the data.

- **`plot_generation.py`**: Methods to display graphics.
- **`methods.py`**: Methods used to do the analysis.

- **`results.ipynb`**: Complet analysis of the data.


### Summary of our work

The data cleaning process involves handling missing values, correcting data types, and extracting useful features. Missing values in key columns are either dropped or filled, and any incorrectly formatted columns are recalculated or converted to appropriate types. Date columns are transformed into a consistent datetime format, and additional temporal features like year, month, and day are created. These preprocessing steps ensure the data is clean and ready for analysis.

The analysis in this project is organized around five key areas: First, we perform a **global analysis of categories**, where we compare them based on various metrics such as views, subscribers, number of videos, and their evolution over time. This allows us to identify significant trends and patterns in content performance across different categories. Second, we apply **community detection** techniques to identify clusters of frequently co-occurring categories. This reveals content ecosystems on YouTube and helps us understand how different types of content interact and diversify. Third, we propose a **Channel Diversity Metric**, which quantifies the diversity of content within channels. This metric offers a new perspective on content variety and its impact on channel performance, providing valuable insights into creative strategies for content creators. Next, we analyze and compare the **evolution of channels across clusters** using time series data. This helps us track growth dynamics over time, focusing on subscriber and view trends, and revealing differences in growth trajectories between clusters. Last but not least, we try to **predict a channel's cluster based on its time series data**. This will eventualy serve as a proof that time series data may require more complex architectures to be able to extract meaningful information from it.

For data storage, we opted for **MongoDB**, a NoSQL database that is well-suited for handling large-scale, unstructured data. MongoDB’s document-based architecture integrates seamlessly with our dataset stored in JSONL (JSON Lines) format, enabling efficient storage, querying, and retrieval. The flexibility of MongoDB’s schema allows us to easily adapt as our data evolves, without the constraints of traditional relational databases. With MongoDB, we can efficiently manage millions of documents, perform complex queries, and filter data by categories, channels, or time ranges without needing to load entire datasets into memory. This makes MongoDB an ideal choice for handling the large and dynamic dataset we are working with, ensuring that our analysis remains scalable and efficient.


## Novel approach

In this project, we introduce a novel metric for measuring channel diversity, combining both **category distribution** and **category relatedness**. A channel is considered more diverse if it covers categories that rarely co-occur, rather than just having a wide range of topics. Our metric is based on **Entropy** and **Normalized Mutual Information (NMI)**.

### i. Entropy: Measuring Distribution

**Entropy** captures the spread of content across different categories. A higher entropy value indicates a more diverse distribution of content. It's calculated as:

$$
H = - \sum_{i=1}^{n} p_i \log(p_i)
$$

Where \( p_i \) is the proportion of content in category \( i \), and \( n \) is the total number of categories.

### ii. Normalized Mutual Information (NMI): Adjusting for Category Relationships

**NMI** adjusts for relatedness between categories. It penalizes channels that focus on categories with high co-occurrence, reflecting that they are less independent. The NMI between two categories \( i \) and \( j \) is:

$$
\text{NMI}(i,j) = \frac{I(i,j)}{\max(H(i), H(j))}
$$

Where \( I(i,j) \) is the mutual information between categories \( i \) and \( j \).

### iii. Full Diversity Metric

The final **Channel Diversity Metric** combines entropy and NMI adjustments:

$$
\text{Channel Diversity} = \frac{- \sum_{i\neq j} p_i \log(p_i) \cdot (1 - \text{NMI}(i,j))}{\log(n)}
$$

This formula encourages a broad distribution of content while penalizing category overlaps, providing a comprehensive measure of channel diversity.


## Organisation within the Team

We followed the Agile methodology, holding weekly meetings at the start of each week to allocate tasks and define clear, actionable goals. At the end of each week, we conduct a review to evaluate our progress and then reassign tasks for the upcoming week, ensuring that everyone gains experience in different aspects of the project. This approach allows each team member to contribute to all areas, enhancing their skills across the board. As a result, it is hard to attribute specific tasks to only one individual, but if we were to look at the time spent on each task, the organization of work would be as follows:

*Aymane: Community Detection, Diversity Metric, Data Story

*Grégoire: Global Analysis of Categories, Prediction, Data Story

*Antoine: Global Analysis of Categories, Evolution of Channels Across Clusters, Data Story

*Redouane: Community Detection, Evolution of Channels Across Clusters, Prediction

*Noah: Global Analysis of Categories, Prediction, Data Story
