import pandas as pd
from collections import defaultdict
from datetime import datetime


def retrieve_all_channels_from_mongodb(collection):
    """
    Retrieves all videos from MongoDB.
    
    Returns:
    - A dataframe containing all channels information.
    """
    result = collection.find()
    return pd.DataFrame(result)


def retrieve_categories_from_mongodb(collection):
    """
    Retrieves all video categories grouped by channel_id from MongoDB.
    
    Returns:
    - channel_categories_dict: A dictionary where keys are channel_ids and values are lists of categories.
    """
    
    # Fetch data from MongoDB and process categories per channel
    pipeline = [
        {"$unwind": "$categories"},
        {"$group": {"_id": "$channel_id", "categories": {"$addToSet": "$categories"}}}
    ]
    
    result = collection.aggregate(pipeline)
    
    # Process the result into a dictionary
    channel_categories_dict = {}
    for doc in result:
        channel_categories_dict[doc['_id']] = doc['categories']
        
    return channel_categories_dict


def retrieve_categories_with_redundancy(collection):
    """
    Retrieves all video categories grouped by channel_id from MongoDB with redundancy.
    
    Returns:
    - channel_categories_dict: A dictionary where keys are channel_ids and values are lists of categories (including repetitions).
    - unique_categories: A set containing all unique categories across all channels.
    """
    
    # Step 1: Fetch data from MongoDB and process categories per channel with redundancy
    pipeline = [
        {"$unwind": "$categories"},
        {"$group": {"_id": "$channel_id", "categories": {"$push": "$categories"}}}
    ]
    
    result = collection.aggregate(pipeline)
    
    # Step 2: Process the result into a dictionary and build a set of unique categories
    channel_categories_dict = {}
    unique_categories = set()
    
    for doc in result:
        channel_id = doc['_id']
        categories = doc['categories']
        channel_categories_dict[channel_id] = categories
        
        # Add categories to the unique set
        unique_categories.update(categories)
    
    # Convert the set to a list for further processing if needed
    unique_categories = list(unique_categories)
    
    return channel_categories_dict, unique_categories


def retrieve_subscribers_for_categories(channel_categories, collection):
    """
    Retrieves the number of subscribers for channels in each category (low, moderate, high).
    
    Args:
    - channel_categories: A dictionary where keys are 'low', 'moderate', 'high' and values are lists of channel_ids.
    - collection: The MongoDB collection containing channel data with fields `channel_id` and `subscribers`.
    
    Returns:
    - subscribers_dict: A dictionary where keys are 'low', 'moderate', 'high', 
                        and values are lists of subscriber counts corresponding to the channel_ids.
    """
    
    # Initialize dictionary to hold the subscribers count for each category
    subscribers_dict = {'low': [], 'moderate': [], 'high': []}
    
    # Iterate over the categories and their respective channel_ids
    for category, channel_ids in channel_categories.items():
        # Query MongoDB for the subscriber count of the given channel_ids
        pipeline = [
            {"$match": {"channel_id": {"$in": channel_ids}}},
            {"$project": {"channel_id": 1, "subscribers": 1}}
        ]
        
        # Fetch data from MongoDB for this category
        result = collection.aggregate(pipeline)

        # Extract subscriber counts from the result and append them to the appropriate category
        for doc in result:
            subscribers_dict[category].append(doc['subscribers'])
    
    return subscribers_dict


def retrieve_timeseries_data_by_cluster(collection, categories):
    """
    Fetches time series data from MongoDB for the specified categories, limited to the top 5000 channels by total views.
    
    Args:
    - collection: MongoDB collection object.
    - categories: List of categories to filter.
    
    Returns:
    - data: A list of dictionaries containing the time series data.
    """

    # Step 1: Fetch the top 5000 channels by total views
    pipeline_top_channels = [
        {"$match": {"category": {"$in": categories}}},
        {"$group": {
            "_id": "$channel",
            "total_views": {"$sum": "$views"}
        }},
        {"$sort": {"total_views": -1}},
        {"$limit": 5000}
    ]
    
    # Get the list of top 5000 channels
    top_channels = list(collection.aggregate(pipeline_top_channels))
    top_channel_ids = {doc['_id'] for doc in top_channels}
    print("Retrieved the list of the top 5000 channels.")

    
    # Step 2: Fetch time series data for the top 5000 channels
    pipeline_timeseries = [
        {"$match": {"category": {"$in": categories}, "channel": {"$in": list(top_channel_ids)}}},
        {"$sort": {"channel": 1, "datetime": 1}}
    ]
    
    cursor = collection.aggregate(pipeline_timeseries)
    print("Fetched time series data.")
    
    # Step 3: Organize data into a dictionary keyed by 'channel'
    data = defaultdict(list)
    for doc in cursor:
        try:
            doc['datetime'] = datetime.strptime(doc['datetime'], '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            continue  # Skip if there's an issue with datetime conversion
        data[doc['channel']].append(doc)

    print("Converted strings to datetime objects")
    
    return data