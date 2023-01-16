# -*- coding: utf-8 -*-
import re
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(
    input_filepath: str = "../data/raw/train.csv.zip",
    output_filepath: str = "../data/processed/train_processed.csv",
) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    print("Loading zip files..")
    data = pd.read_csv(input_filepath, compression="zip")
    # labels = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
    data["total_classes"] = data.iloc[:, 2:8].apply(lambda x: sum(x), axis=1)
    data["non_toxic"] = data.iloc[:, 2:8].apply(
        lambda x: 1 if (sum(x) == 0) else 0, axis=1
    )
    cleaned_data = data.copy()

    # Removing Hyperlinks from text
    # The below code is a re pattern that matches URLs beginning with http or https, followed by :// and one or more
    # non-whitespace characters, or URLs beginning with www., followed by one or more non-whitespace characters.

    print("Cleaning the data..")
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(
        lambda x: re.sub(r"https?://\S+|www\.\S+", "", x)
    )
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(
        lambda x: re.sub(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            "",
            x,
            flags=re.UNICODE,
        )
    )

    # matches of css style,class elements and attributes
    idxs_css = []
    for i in range(len(cleaned_data)):
        if re.findall(r"[{][|].+\n", cleaned_data.loc[i, "comment_text"]):
            idxs_css.append(i)

    # Comments Containing Css style, class and attributes
    cleaned_data.loc[idxs_css, "total_classes"].value_counts()
    cleaned_data.drop(idxs_css, inplace=True)

    # Removing html tags from text
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(
        lambda x: re.sub(r"<.*?>", "", x)
    )
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(
        lambda x: re.sub(r"\"\"", '"', x)
    )  # replacing "" with "
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(
        lambda x: re.sub(r"^\"", "", x)
    )  # removing quotation from start and the end of the string
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(
        lambda x: re.sub(r"\"$", "", x)
    )

    # Removing Punctuation / Special characters (;:'".?@!%&*+) which appears more than twice in the text
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(
        lambda x: re.sub(r"[^a-zA-Z0-9\s][^a-zA-Z0-9\s]+", " ", x)
    )

    # Removing Special characters
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(
        lambda x: re.sub(r"[^a-zA-Z0-9\s\"\',:;?!.()]", " ", x)
    )

    # Removing extra spaces in text
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(
        lambda x: re.sub(r"\s\s+", " ", x)
    )
    cleaned_data.reset_index(inplace=True)
    cleaned_data.drop(columns=["index"], inplace=True)
    Final_data = cleaned_data.copy()

    # Keeping 15291 comments from non_toxic comments and dropping the rest
    C_indexs = cleaned_data[cleaned_data["total_classes"] == 0].index
    drop_indxs = np.random.choice(C_indexs, size=127696, replace=False)

    # Randomly selecting and dropping non_toxic comments from datasets and leaving 5000 behind
    Final_data.drop(drop_indxs, inplace=True)

    Final_data.to_csv(output_filepath, index=False)
    print("Saved file in data\processed")

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
