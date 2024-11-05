import joblib 
from sklearn.datasets import load_iris 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()

# Split the dataset into features (X) and target (y)
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100)

# Train the classifier
clf.fit(X_train, y_train)

# Save the trained classifier to a file
joblib.dump(clf, 'iris_classifier.joblib')