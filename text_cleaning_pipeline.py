from helpers.helper_module import pd, re, Tokenizer, fugashi


# Load data
data = pd.read_csv('reviews.csv')
text_data = data['Review']

def clean_text(text):
    """
    Clean the input text by removing numbers, extra spaces, and symbols.
    """
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Apply text cleaning
data['cleaned_text'] = data['Review'].apply(clean_text)

# Initialize tokenizers
tokenizer = Tokenizer()
tagger = fugashi.Tagger()

def tokenize_text(text, use_fugashi=True):
    """
    Tokenize the input text using either Fugashi or custom Tokenizer.
    """
    if use_fugashi:
        tokens = tagger(text)
    else:
        tokens = tokenizer.tokenize(text)
    return ' '.join([token.surface for token in tokens])

# Apply tokenization
data['tokenized_text'] = data['cleaned_text'].apply(lambda x: tokenize_text(x, use_fugashi=True))

# Define stop words
STOP_WORDS = set([
    'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる',
    'も', 'する', 'から', 'な', 'こと', 'として', 'い', 'や', 'れる', 'など', 'なっ', 'ない',
    'あっ', 'よう', 'まし', 'その', 'あ', 'これ', 'それ'
])

def remove_stopwords(text):
    """
    Remove stop words from the input text.
    """
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token not in STOP_WORDS]
    return ' '.join(filtered_tokens)

# Apply stop word removal
data['filtered_text'] = data['tokenized_text'].apply(remove_stopwords)

# Save processed data
data.to_csv('cleaned_data.csv', index=False)