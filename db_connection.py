import pymongo
import yaml

# Function to load configuration from the config.yaml file
def load_config(config_file="config/config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

# Function to establish a connection to MongoDB
def connect_to_mongodb(collection_name,config_file="config/config.yaml"):
    # Load the configuration
    config = load_config(config_file)

    # Retrieve MongoDB configuration from the YAML file
    mongodb_config = config.get("mongodb", {})
    
    # Extract connection details
    host = mongodb_config["host"]
    port = mongodb_config["port"]
    database = mongodb_config["database"]
    username = mongodb_config["username"]
    password = mongodb_config["password"]
    auth_db = mongodb_config["auth_db"]

    # Create the MongoDB URI for authentication (if necessary)
    if username and password:
        uri = f"mongodb://{username}:{password}@{host}:{port}/{database}?authSource={auth_db}"
    else:
        uri = f"mongodb://{host}:{port}/{database}"

    # Establish connection
    client = pymongo.MongoClient(uri)

    # Access the specified database and collection
    db = client[database]
    collection = db[collection_name]
    
    print(f"Connected to MongoDB at {host}:{port}, Database: {database}, Collection: {collection_name}")
    return collection