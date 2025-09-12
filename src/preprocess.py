import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

def clean_text(text): 
    text = text.lower() 
    text = re.sub(r"http\S+", "", text) # Delete links 
    text = re.sub(r"@\w+", "", text) # Delete mention 
    text = re.sub(r"[^a-z\s]", "", text) # Delete symbols 
    tokens = [w for w in text.split() if w not in stopwords.words("english")] 
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)
print(df["clean_text"].head())