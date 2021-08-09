# importing necessary libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

# ------------------------------------------------------------------------------
# loading data from csv
def load_data(messages_filepath, categories_filepath):

    """
    Function that permits loading of data and merging in one df
    args :
        categories_file: path for the categories raw csv file
    returns :
        message_file : path for the messages file
    """

    # reading the csv files
    categories = pd.read_csv("{}".format(categories_filepath))
    messages = pd.read_csv("{}".format(messages_filepath))

    # merging the two csv
    df = pd.merge(messages, categories, on="id")

    return df


# ------------------------------------------------------------------------------
# cleaning the dataframe after loading from csv
def clean_data(df):

    """
    Function that expand categories column , clean it and merge the final df for saving
    args :
        df: dataframe to prepare from load data function
    returns :
        df: final df prepared for saving
    """

    # expanding the categories column by spliting the values based on comma
    cat_expanded = df["categories"].str.split(";", expand=True)

    # looping through the expanded df and replacing all values to 1 or 0
    col_dict = {}

    for col in cat_expanded.columns:
        col_dict[
            "{}".format(cat_expanded[col].str.split("-", expand=True)[0].unique()[0])
        ] = (cat_expanded[col].str.split("-", expand=True)[1].replace("2", "1"))

    # dictionary as dataframe for export
    categories_ready = pd.DataFrame.from_dict(col_dict, dtype=int)

    # making sur that our output is int type
    for col in categories_ready.columns:
        categories_ready[col] = categories_ready[col].astype(int)

    # concatenating the message and prepared categories
    df = pd.concat([df[["message", "genre"]], categories_ready], axis=1)

    # droping duplicates
    df = df.drop_duplicates()

    return df


# ------------------------------------------------------------------------------
# saving data as a database after loading and cleaning
def save_data(df, database_filepath):
    """
    Function that saves the final df to database
    args:
        df: dataframe to save
    returns:
        saved database
    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df.to_sql("messages", con=engine, if_exists="replace")


# ------------------------------------------------------------------------------
# execution of the full process of loading, cleaning and saving
def main():
    """
    Function that leverage all the other function and execute the full ETL pipeline
    args:
        None
    returns:
        Preprocessed and cleaned data in a database
    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python data/process_data.py "
            "data/disaster_messages.csv data/disaster_categories.csv "
            "data/DisasterResponse.db"
        )


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
