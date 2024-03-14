from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

with open('keywords.txt') as file:
    sentences = [kwd for kwd in file]

# Initialize Tf-idf vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Compute the tf-idf scores for each sentence
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

# Compute the cosine similarity between each pair of sentences
similarity_matrix = cosine_similarity(tfidf_matrix)

# Print the similarity matrix
print(similarity_matrix)

# Initialize an empty list to store the grouped sentences
grouped_sentences = []

# Initialize a list to keep track of which sentences have been added to a group
added_sentences = [False] * len(sentences)

# Iterate through each sentence
for i in range(len(sentences)):
    if not added_sentences[i]:
        # Initialize a new group for this sentence
        new_group = [sentences[i]]
        added_sentences[i] = True
        # Iterate through the remaining sentences
        for j in range(i+1, len(sentences)):
            if not added_sentences[j]:
                # Check the similarity between the current sentence and the next sentence
                if similarity_matrix[i, j] > 0.5:
                    # Add the sentence to the current group
                    new_group.append(sentences[j])
                    added_sentences[j] = True
        # Add the group to the list of grouped sentences
        grouped_sentences.append(new_group)

# Print the grouped sentences

# print(grouped_sentences)
# print('\n'.join(map(str, grouped_sentences)))

#print out the resulting list with groups separated by space
# for item in grouped_sentences:
#  print(*item,"\n\n", sep='\n')

#print out the resulting list as a dictionary 
# for item in grouped_sentences:
#  result = {item[0]: ', '.join(map(str, item[1:]))  for item in grouped_sentences}
#  result_clean = {}
#  for key,value in result.items():
#    key_clean = key.replace('\n','')
#    value_clean = value.replace('\n','')
#    result_clean[key_clean] = value_clean
#  print(result_clean)


#add the resulting dictionary to a apandas dataframe with keys and values as columns
for item in grouped_sentences:
 result = {item[0]: item[1:] for item in grouped_sentences}
 # result_clean = {}
 # for key,value in result.items():
 # key_clean = key.replace('\n','')
 # value_clean = value.replace('\n','')
 # result_clean[key_clean] = value_clean

dataframes = []

# Iterate over the dictionary
for key, values in result.items():
    # Convert the key and values to a DataFrame
    df = pd.DataFrame({'keys': key, 'values': values})
    dataframes.append(df)

# Concatenate all dataframes
final_df = pd.concat(dataframes, ignore_index=True)
final_df['keys'] = final_df['keys'].replace('\n', '', regex=True)
final_df['values'] = final_df['values'].replace('\n', '', regex=True)

final_df.to_csv('topics_grouped.csv', index=False)
#print the resulting dataframe
# print(final_df)

# This code uses the Tf-idf Vectorizer from the scikit-learn library to compute the similarity between each pair of sentences, and the cosine_similarity function to compute the cosine similarity between the tf-idf vectors for each sentence. The code then uses a nested for loop to iterate through the sentences, comparing the similarity of each sentence to every other sentence. If the similarity between two sentences is greater than a certain threshold (0.5 in this case), the second sentence is added to the first sentence's group. The code then prints the grouped sentences.

# The threshold can be adjusted as per requirement and also can use different similarity measure like Jaccard Similarity, etc. The above code is just an example and you can use other clustering algorithms like K-means, Hierarchical clustering, etc. to group sentences based on their common meaning.