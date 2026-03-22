from flask import Flask, render_template, request
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
}

model_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_scores[name] = accuracy_score(y_test, y_pred)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    selected_model = 'Random Forest'
    error = None

    if request.method == 'POST':
        try:
            selected_model = request.form.get('model', 'Random Forest')
            features = [
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width']),
            ]
            clf = models[selected_model]
            pred_index = clf.predict([features])[0]
            prediction = iris.target_names[pred_index]
        except Exception as e:
            error = str(e)

    return render_template(
        'index.html',
        model_scores=model_scores,
        prediction=prediction,
        error=error,
        selected_model=selected_model,
    )

if __name__ == '__main__':
    app.run(debug=True)

