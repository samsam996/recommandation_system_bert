
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def xgb_classifier(df):

    # # need to embed the text data into a vector
    # # Feature extraction with TF-IDF
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(df["title"])
    y_true = df["label"]

    # split the training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

    # 2. Initialize the XGBoost classifier
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    # 3. Train the model
    model.fit(X_train, y_train)

    # 4. Make predictions on the test set
    y_pred = model.predict(X_test)

    # 5. Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

