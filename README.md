# Unveiling the secrets behind Youtube categories

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

Currently, no additional datasets are planned.

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
│   ├── plot_generation.py
│   └── community_detection.py
├── tests/                        # tests directory
│   └── (empty)
├── results.ipynb                 # Notebook for exploring and visualizing results
├── README.md
├── requirements.txt
└── .gitignore
```


- **`db_connection.py`**: Code to make the connection with the mongodb database.
- **`db_methods.py`**: Methods to work with the mongodb database.
- **`config.yaml`**: configuration file for the connection with the database

- **`data_cleaning.ipynb`**: This notebook shows how we clean the data.

- **`plot_generation.py`**: Methods to display graphics.
- **`methods.py`**: Methods used to do the analysis.

- **`results.ipynb`**: Complet analysis of the data.


### Summary of our work

The data cleaning process involves handling missing values, correcting data types, and extracting useful features. Missing values in key columns are either dropped or filled, and any incorrectly formatted columns are recalculated or converted to appropriate types. Date columns are transformed into a consistent datetime format, and additional temporal features like year, month, and day are created. These preprocessing steps ensure the data is clean and ready for analysis.

The analysis in this project is organized around four key areas: First, we perform a **global analysis of categories**, where we compare them based on various metrics such as views, subscribers, number of videos, and their evolution over time. This allows us to identify significant trends and patterns in content performance across different categories. Second, we apply **community detection** techniques to identify clusters of frequently co-occurring categories. This reveals content ecosystems on YouTube and helps us understand how different types of content interact and diversify. Third, we propose a **Channel Diversity Metric**, which quantifies the diversity of content within channels. This metric offers a new perspective on content variety and its impact on channel performance, providing valuable insights into creative strategies for content creators. Lastly, we analyze and compare the **evolution of channels across clusters** using time series data. This helps us track growth dynamics over time, focusing on subscriber and view trends, and revealing differences in growth trajectories between clusters.

For data storage, we opted for **MongoDB**, a NoSQL database that is well-suited for handling large-scale, unstructured data. MongoDB’s document-based architecture integrates seamlessly with our dataset stored in JSONL (JSON Lines) format, enabling efficient storage, querying, and retrieval. The flexibility of MongoDB’s schema allows us to easily adapt as our data evolves, without the constraints of traditional relational databases. With MongoDB, we can efficiently manage millions of documents, perform complex queries, and filter data by categories, channels, or time ranges without needing to load entire datasets into memory. This makes MongoDB an ideal choice for handling the large and dynamic dataset we are working with, ensuring that our analysis remains scalable and efficient.



## Proposed Timeline

- **Week 1**: Apply more community detection and deepen analysis on audience segmentation.
- **Week 2**: Complete clustering and validate growth trends through time-series analysis.
- **Week 3**: Synthesis of findings, begin drafting visualisations for data storytelling.
- **Week 4**: Explore the possibility of predicting the categorical cluster of a channel using timeseries data
- **Week 5**: Finalise the project repository, complete the online report and visualisations.

## Organisation within the Team

We work using the Agile methodology, holding a weekly meeting at the beginning of the week to divide tasks and set clear and precise objectives. The following week, we do a review and start over by assigning different topics to each team member so that everyone gains expertise in various aspects.

## Questions for TAs
Do you have any advice for us for the rest of the project ? Thank you.
