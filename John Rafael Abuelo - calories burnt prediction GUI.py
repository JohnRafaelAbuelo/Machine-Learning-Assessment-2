#Importing the necessary libraries and functions for model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
#importing libraries for GUI 
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import numpy as np
#importing libraries for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

#loading the data from csv file to pandas data frame for reading
calories=pd.read_csv('calories.csv')

#displaying and reviewing datasets

print(calories.shape)

calories.head()

excercise=pd.read_csv('exercise.csv') 

excercise.shape

excercise.head()

calories_data=pd.concat([excercise,calories['Calories']],axis=1)

calories_data.head()

calories_data.shape

#displaying more information from the data
calories_data.info()

#checking for missing values in the dataset

calories_data.isnull().sum()

#analyzing the given data using descriptions

calories_data.describe()

sns.set()

#plotting gender column to number of males and females using plot chart

sns.countplot(calories_data,x='Gender')

#finding the distribution of "Age" column
sns.distplot(calories_data['Age'])

sns.displot(calories_data['Age'])

sns.histplot(calories_data['Age'])

#finding the distribution of "Height" column
sns.distplot(calories_data['Height'])

#finding the distribution of "Weight" column
sns.distplot(calories_data['Weight'])

#finding the distribution of "Heart-Rate" column
sns.distplot(calories_data['Heart_Rate'])

#finding the distribution of "Duration" column
sns.displot(calories_data['Duration'])

#one hot encoding the categorical data received

calories_data['Gender']=pd.Categorical(calories_data['Gender'])
calories_data_encoded=pd.get_dummies(calories_data,columns=['Gender'])

correlation_matrix=calories_data_encoded.corr()

#constructing a heatmap to understsnad the correlation between variables

plt.figure(figsize=(10,10))
sns.heatmap(correlation_matrix,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Greens')

#categorizing the calories data

calories_data['Gender'] = pd.Categorical(calories_data['Gender'])


categorical_columns = calories_data.select_dtypes(include=['category']).columns

#excluding categorical columns from correlation calculation
numeric_columns = calories_data.drop(columns=categorical_columns)
correlation = numeric_columns.corr()

#constructing a heatmap to understsnad the correlation between variables

plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')

calories_data.head()

calories_data.replace({'Gender':{'male':0,'female':1}},inplace=True)

calories_data.head()

X=calories_data.drop(columns=['User_ID','Calories'],axis=1)
Y=calories_data['Calories']

X.head()

Y

#splitting the test and training data

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

X_train

X_train=X_train.astype(float)

print(X_train.head)

#utilizing XGBRegressor for a predictive regression model

model=XGBRegressor()

model.fit(X_train,Y_train)

#prediction for test data

test_data_prediction=model.predict(X_test)

print(test_data_prediction) 

mean_abs_error=metrics.mean_absolute_error(Y_test,test_data_prediction)

#evaluating the mean absolute error to see the accuracy of the model

print("Mean Absolute error:",mean_abs_error)


#initializing a hyperparameter tuning using GridSearchCV

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

#initiating XGBoost model
model = XGBRegressor()

#creating GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5)
grid_search.fit(X_train, Y_train)

#get the best parameters
best_params = grid_search.best_params_

#training the model with the best parameters
best_model = XGBRegressor(**best_params)
best_model.fit(X_train, Y_train)

#prediction using the tuned model
test_data_prediction = best_model.predict(X_test)

#mean Absolute Error with tuned model
mean_abs_error_tuned = metrics.mean_absolute_error(Y_test, test_data_prediction)

#display the best parameters for the model
print("Best Hyperparameters:", best_params)

#display the mean absolute error with tuned model
print("Mean Absolute Error (Tuned Model):", mean_abs_error_tuned)

#creating tkinter GUI for the model

def display_results(predictions, actual_values, mean_absolute_error, features):
    root = tk.Tk()
    root.title("Calories Prediction Results")
    root.geometry("1000x600")
    root.configure(bg="#2E2E2E")

    #styling the tkinter GUI
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview",
                    background="#404040",
                    foreground="white",
                    rowheight=20,
                    font=('Helvetica', 10),
                    fieldbackground="#404040")

    #creating a frame
    frame = ttk.Frame(root, style="Treeview", borderwidth=5, relief="sunken")
    frame.grid(row=0, column=0, padx=10, pady=10)

    #create a figure and axis for the plot
    fig = Figure(figsize=(10, 6), dpi=80, facecolor="#2E2E2E")
    ax1 = fig.add_subplot(231, facecolor="#2E2E2E")
    ax2 = fig.add_subplot(232, facecolor="#2E2E2E")
    ax3 = fig.add_subplot(233, facecolor="#2E2E2E")
    ax4 = fig.add_subplot(234, facecolor="#2E2E2E")
    ax5 = fig.add_subplot(235, facecolor="#2E2E2E")

    #plotting the predicted vs actual values
    ax1.plot(predictions, label='Predicted Calories', marker='o', color='#5DADE2')
    ax1.plot(actual_values.values, label='Actual Calories', marker='o', color='#E74C3C')
    ax1.set_xlabel('Index', color='white')
    ax1.set_ylabel('Calories', color='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.legend()

    #creating a bar chart for predicted and actual values for the GUI
    bar_labels = ['Predicted', 'Actual']
    bar_values = [np.mean(predictions), np.mean(actual_values)]
    ax2.bar(bar_labels, bar_values, color=['#5DADE2', '#E74C3C'])
    ax2.set_ylabel('Mean Calories', color='white')
    ax2.tick_params(axis='y', colors='white')

    #creating correlation heatmap
    correlation_matrix = features.corr()
    sns.heatmap(correlation_matrix, ax=ax3, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8},
                cmap='coolwarm')

    # Distribution plot for actual calorie values
    sns.histplot(actual_values, kde=True, ax=ax4, color='#5DADE2')
    ax4.set_title('Actual Calories Distribution', color='white')

    # Distribution plot for predicted calorie values
    sns.histplot(predictions, kde=True, ax=ax5, color='#E74C3C')
    ax5.set_title('Predicted Calories Distribution', color='white')

    # Embedding the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    # Adding a toolbar for navigation
    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()
    toolbar.pack(side=tk.TOP, fill=tk.BOTH)

    #adding MAE label
    mae_label = ttk.Label(root, text=f"Mean Absolute Error: {mean_absolute_error:.2f}", font=('Helvetica', 12),
                          background="#2E2E2E", foreground="white")
    mae_label.grid(row=1, column=0, columnspan=2, pady=10)

    #creating a frame for the table
    table_frame = ttk.Frame(root, style="Treeview", borderwidth=5, relief="sunken", padding=(10, 10))
    table_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

    #creating columns for the table
    columns = ["Index", "Actual Calories", "Predicted Calories"]

    #creating a treeview widget for the table for organization
    table = ttk.Treeview(table_frame, columns=columns, show='headings', height=8, style="Treeview")

    #adding column headings
    for col in columns:
        table.heading(col, text=col, anchor='center')

    #adding data to the table
    for i, (actual, predicted) in enumerate(zip(actual_values.values, predictions)):
        table.insert("", "end", values=(i, actual, predicted))

    #packing the table to make it more compact
    table.pack()

    #run the tkinter event loop
    root.mainloop()

#extracting features for correlation heatmap
features_for_correlation = X_test[['Age', 'Height', 'Weight', 'Heart_Rate', 'Duration']]

#display the results in the tkinter GUI
display_results(test_data_prediction, Y_test, mean_abs_error, features_for_correlation)
