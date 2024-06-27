import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
plt.style.use("ggplot")

#Recommendation System - Part 1
amazon_ratings = pd.read_csv('C:/Users/jorda/OneDrive/Desktop/Nxt24/2 E-commerce Product Recommendation System/CSV/Amazon.csv')
amazon_ratings = amazon_ratings.dropna()
print(amazon_ratings.head())
print(amazon_ratings.shape)

popular_products = pd.DataFrame(amazon_ratings.groupby('ProductId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
print(most_popular.head(10))

# Plotting the top 30 items as a bar chart
most_popular.head(30).plot(kind='bar')

# Display the plot
plt.xlabel('Product ASIN')
plt.ylabel('Frequency')
plt.title('Top 30 Most Popular Products')
plt.show()

#Recommendation System - Part 2
# Creating a subset of the first 10,000 rows
amazon_ratings1 = amazon_ratings.head(10000)
# Print the shape to confirm the subset size
print(amazon_ratings1.shape)
# Optionally, display the first few rows to confirm
print(amazon_ratings1.head())

# Creating the ratings utility matrix
ratings_utility_matrix = amazon_ratings1.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)

# Display the first few rows of the utility matrix
print(ratings_utility_matrix.head())
print(ratings_utility_matrix.shape)

# Transpose the ratings utility matrix
X = ratings_utility_matrix.T
# Display the first few rows of the transposed matrix
print(X.head())
print(X.shape)
X1 = X
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X1)
print(decomposed_matrix.shape)
correlation_matrix = np.corrcoef(decomposed_matrix)
print(correlation_matrix.shape)
print(X.index[99])

i = "6117036094"
product_names = list(X.index)
product_ID = product_names.index(i)
print(product_ID)

correlation_product_ID = correlation_matrix[product_ID]
print(correlation_product_ID.shape)
recommend = list(X.index[correlation_product_ID > 0.90])
# It removes the item already bought by the customer
recommend.remove(i) 
print(recommend[:9])

#Recomendation system 3
product_descriptions = pd.read_csv('C:/Users/jorda/OneDrive/Desktop/Nxt24/2 E-commerce Product Recommendation System/CSV/product_descriptions.csv')
print(product_descriptions.shape)
# Missing values
product_descriptions = pd.read_csv('C:/Users/jorda/OneDrive/Desktop/Nxt24/2 E-commerce Product Recommendation System/CSV/product_descriptions.csv')
print(product_descriptions.shape)
print(product_descriptions.head())
product_descriptions1 = product_descriptions.head(500)
# product_descriptions1.iloc[:,1]
print(product_descriptions1["product_description"].head(10))

#Converting the text in product description into numerical data for analysis
vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(product_descriptions1["product_description"])
print(X1)

# Fitting K-Means to the dataset
X = X1
kmeans = KMeans(n_clusters=10, init='k-means++')
y_kmeans = kmeans.fit_predict(X)
plt.plot(y_kmeans, ".")
plt.show()

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
def print_cluster(i):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print()

# # Optimal clusters is
true_k = 10

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X1)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

def print_cluster(i):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print()

for i in range(true_k):
    print_cluster(i)

#Predicting clusters based on key search words
def show_recommendations(product):
    Y = vectorizer.transform([product])
    prediction = model.predict(Y)
    print_cluster(prediction[0])

show_recommendations("Your sample product description goes here.")

print("Test 1")
show_recommendations("cutting tool")
print("Test 2")
show_recommendations("spray paint")
print("Test 3")
show_recommendations("steel drill")
print("Test 4")
show_recommendations("water")
