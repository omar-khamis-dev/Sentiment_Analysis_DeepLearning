import pandas as pd
import requests, zipfile, io

url = "https://nyc3.digitaloceanspaces.com/ml-files-distro/v1/sentiment-analysis-is-bad/data/training.1600000.processed.noemoticon.csv.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
df = pd.read_csv(z.open("training.1600000.processed.noemoticon.csv"), encoding="latin-1")

# Datast columns
df.columns = ["target", "ids", "date", "flag", "user", "text"]
df = df[["target", "text"]]

# Convert target from 0=negative, 4=positive
df["target"] = df["target"].replace({0:0, 4:1})

print(df.head())
print(df.shape)