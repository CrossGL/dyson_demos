import pandas as pd
import dyson
from dyson import DysonRouter
import os

os.environ["dyson_api"] = "dyson_api_key_here"
router = DysonRouter()
from google.cloud import storage
import logging


def titanic_etl(data):
    """
    ETL function that:
    1. Downloads Titanic dataset from a URL
    2. Preprocesses the data (cleaning, feature engineering)

    Args:
        csv_url (str): URL to the Titanic CSV file
        bucket_name (str): GCP bucket name to store the processed data
    Returns:
        pandas.DataFrame: The processed DataFrame
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("titanic_etl")

    # Step 1: Extract - Download the CSV file

    # Step 2: Transform - Preprocess the Titanic dataset
    logger.info("Preprocessing Titanic dataset")
    try:
        # Make a copy to avoid modifying the original
        processed_df = data.copy()

        # 2.1 Handle missing values
        # Fill missing age with median
        processed_df["Age"] = processed_df["Age"].fillna(processed_df["Age"].median())

        # Fill missing embarked with most common value
        most_common_embarked = processed_df["Embarked"].mode()[0]
        processed_df["Embarked"] = processed_df["Embarked"].fillna(most_common_embarked)

        # Fill missing cabin with 'Unknown'
        processed_df["Cabin"] = processed_df["Cabin"].fillna("Unknown")

        # 2.2 Feature engineering
        # Extract title from name
        processed_df["Title"] = processed_df["Name"].str.extract(
            " ([A-Za-z]+)\.", expand=False
        )

        # Group rare titles
        rare_titles = [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ]
        processed_df.loc[processed_df["Title"].isin(rare_titles), "Title"] = "Rare"
        processed_df.loc[processed_df["Title"] == "Mlle", "Title"] = "Miss"
        processed_df.loc[processed_df["Title"] == "Ms", "Title"] = "Miss"
        processed_df.loc[processed_df["Title"] == "Mme", "Title"] = "Mrs"

        # Create family size feature
        processed_df["FamilySize"] = processed_df["SibSp"] + processed_df["Parch"] + 1

        # Create is_alone feature
        processed_df["IsAlone"] = (processed_df["FamilySize"] == 1).astype(int)

        # Extract deck from cabin
        processed_df["Deck"] = processed_df["Cabin"].str[0]
        processed_df["Deck"] = processed_df["Deck"].fillna("U")

        # 2.3 Categorical encoding
        # Convert categorical features to numeric
        processed_df["Sex"] = processed_df["Sex"].map({"male": 0, "female": 1})

        # One-hot encode embarked
        embarked_dummies = pd.get_dummies(processed_df["Embarked"], prefix="Embarked")
        processed_df = pd.concat([processed_df, embarked_dummies], axis=1)

        # One-hot encode title
        title_dummies = pd.get_dummies(processed_df["Title"], prefix="Title")
        processed_df = pd.concat([processed_df, title_dummies], axis=1)

        # One-hot encode deck
        deck_dummies = pd.get_dummies(processed_df["Deck"], prefix="Deck")
        processed_df = pd.concat([processed_df, deck_dummies], axis=1)

        # 2.4 Drop unnecessary columns
        columns_to_drop = ["Name", "Ticket", "Cabin", "Embarked", "Title", "Deck"]
        processed_df = processed_df.drop(columns=columns_to_drop)

        logger.info(
            f"Preprocessing complete. Final dataframe shape: {processed_df.shape}"
        )

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

    return processed_df


hardware = router.route_hardware(
    titanic_etl,
    mode="energy-efficient",
    judge=5,
    run_type="log",
    complexity="medium",
    precision="normal",
    multi_device=False,
)

hardware["spec"]
hardware["hardware_type"]

import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/your_key.json"
from google.cloud import storage


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(f"File {source_blob_name} downloaded to {destination_file_name}.")


download_blob("dysontest", "train.csv", "gcp_train.csv")

# Example CSV URL (Titanic dataset)
df = pd.read_csv("gcp_train.csv")

import dyson

compiled_func = dyson.run(titanic_etl, target_device=hardware["hardware_type"])

# Run the ETL function
processed_data = compiled_func(df)

# Display the first few rows of the processed data
processed_data.head()
