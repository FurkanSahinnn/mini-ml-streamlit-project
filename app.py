import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


# You can remove the comments within each commented code block if you want to visualize each task.
class App:
    def __init__(self):
        self.classifier_name = None
        self.data = None

        self.params = dict()
        self.clf = None
        self.X, self.Y = None, None
        self.X_train, self.Y_train, self.X_test, self.Y_test = None, None, None, None

    def run(self):
        self.task_5()

    def task_1(self):
        uploaded_file = st.sidebar.file_uploader("Upload")
        if uploaded_file is None:
            return False
        self.data = pd.read_csv(uploaded_file)
        if self.data is None:
            return False

        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Naive-Bayes')
        )

        """
        st.markdown("<h2 style='text-align: center; color: white;'>Task - 1 </h2>",
                    unsafe_allow_html=True)
        """
        # st.write(self.data.head(10))
        # st.write(self.data.columns)
        return True

    def task_2(self):
        """
        st.markdown("<h2 style='text-align: center; color: white;'>Task - 2 </h2>",
                    unsafe_allow_html=True)
        """
        # Delete the 'id' and 'Unnamed' columns. We don't need to use them.
        self.data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)  # Clean unnecessary columns.
        # st.write(self.data.tail(10)) # Display the last 10 row values of the data.

        self.data["diagnosis"] = self.data["diagnosis"].map({"M": 1, "B": 0})

        self.Y = self.data["diagnosis"].values
        self.X = self.data.drop(columns=["diagnosis"], axis=1)

        malignant_data = self.data[self.data["diagnosis"] == 1]
        benign_data = self.data[self.data["diagnosis"] == 0]

        fig = plt.figure(figsize=(10, 8))
        plt.scatter(malignant_data.radius_mean, malignant_data.texture_mean, color="red", label="Kotu", alpha=0.3)
        plt.scatter(benign_data.radius_mean, benign_data.texture_mean, color="green", label="Iyi", alpha=0.3)
        plt.xlabel("radius_mean")
        plt.ylabel("texture_mean")
        plt.legend()
        # st.pyplot(fig) # Remove the comment line to visualize the 2nd task.

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y,
                                                                                test_size=0.2, random_state=1234)

    def task_3(self):
        self.add_parameter_ui()
        self.get_classifier()
        self.generate()

    def add_parameter_ui(self):
        if self.classifier_name == "SVM":
            self.params = {"C": [0.01, 0.1, 1, 10, 100]}
        elif self.classifier_name == "KNN":
            self.params = {"n_neighbors": [1, 3, 5, 7, 9]}
        elif self.classifier_name == "Naive-Bayes":
            self.params = {"alpha": np.arange(0, 1.1, 0.1)}  # alpha_values

    def get_classifier(self):
        # Control possible error cases with the error_score parameter.
        if self.classifier_name == 'SVM':
            svc = SVC()
            self.clf = GridSearchCV(estimator=svc, param_grid=self.params, error_score='raise')
        elif self.classifier_name == 'KNN':
            knn = KNeighborsClassifier()
            self.clf = GridSearchCV(estimator=knn, param_grid=self.params, error_score='raise')
        else:
            cnb = ComplementNB(alpha=self.params)
            self.clf = GridSearchCV(estimator=cnb, param_grid=self.params, cv=5, error_score='raise')

    def generate(self):
        self.get_classifier()

        # You cannot train the Naive Bayes model with negative data. Therefore, standardizing is not applied.
        # If we replace the negative data resulting from standardization with 0,
        # the training results of the model will be lower compared to the normal ones.
        if not self.classifier_name == "Naive-Bayes":
            # Standardizing data with StandardScaler() function
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

        #### TRAINING ####
        self.clf.fit(self.X_train, self.Y_train)
        # Find the best parameters.
        st.write("Best parameters:", self.clf.best_params_)

        self.clf = self.clf.best_estimator_  # Select the best parameter.
        self.clf.fit(self.X_train, self.Y_train)  # Train again for the best parameter.

    def task_4(self):
        #### CLASSIFICATION ####
        y_pred = self.clf.predict(self.X_test)

        accuracy = accuracy_score(self.Y_test, y_pred)
        predision = precision_score(self.Y_test, y_pred)
        recall = recall_score(self.Y_test, y_pred)
        f1 = f1_score(self.Y_test, y_pred)

        st.write(f"Classifier = {self.classifier_name}")
        st.write(f"Accuracy = {accuracy}")
        st.write(f"Precision = {predision}")
        st.write(f"Recall = {recall}")
        st.write(f"F1 = {f1}")

        #### PLOT DATASET ####
        cm = confusion_matrix(self.Y_test, y_pred)

        # Visualize the Confusion Matrix.
        fig, ax = plt.subplots(figsize=(5, 5))

        sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("Y_true")
        st.pyplot(fig)

    def task_5(self):
        if not self.task_1():
            return
        st.markdown("<h2 style='text-align: center; color: white;'>Task - 5 </h2>",
                    unsafe_allow_html=True)
        self.task_2()
        self.task_3()
        self.task_4()
