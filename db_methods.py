import pandas as pd

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