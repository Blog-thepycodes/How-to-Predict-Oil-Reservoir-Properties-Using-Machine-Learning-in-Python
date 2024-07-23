import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import *
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
 
# Global variables
data = None
X = None
y = None
X_train = None
X_test = None
y_train = None
y_test = None
lr_model = None
rf_model = None
 
 
# Function to load the dataset
def load_data(file):
   data = pd.read_csv(file)
   return data
 
 
# Function to upload a file
def upload_file():
   file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
   if file:
       global data
       data = load_data(file)
       messagebox.showinfo("File Upload", "File uploaded successfully")
       show_data()
       preprocess_data()
   else:
       messagebox.showerror("File Upload Error", "Can You Please upload a valid CSV file")
 
 
# Function to preprocess data
def preprocess_data():
   global data
   data = data.ffill()
 
 
# Function to show data
def show_data():
   if data is not None:
       top = Toplevel()
       text = Text(top)
       text.insert(INSERT, str(data.head()))
       text.pack()
 
 
# Function to show correlation heatmap
def show_heatmap():
   if data is not None:
       plt.figure(figsize=(10, 8))
       sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
       plt.show()
   else:
       messagebox.showerror("Data Error", "Can You Please upload a valid CSV file first")
 
 
# Function to split data
def split_data():
   global X, y, X_train, X_test, y_train, y_test
   if 'target_property' in data.columns:
       X = data.drop('target_property', axis=1)
       y = data['target_property']
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       messagebox.showinfo("Data Split", "Data split into training and testing sets")
   else:
       messagebox.showerror("Data Error", "The CSV file must contain a 'target_property' column")
 
 
# Function to build models
def build_models():
   global lr_model, rf_model, X_train, y_train
   if X_train is not None and y_train is not None:
       lr_model = LinearRegression()
       rf_model = RandomForestRegressor()
       lr_model.fit(X_train, y_train)
       rf_model.fit(X_train, y_train)
       messagebox.showinfo("Model Training", "Models trained successfully")
       evaluate_models()
   else:
       messagebox.showerror("Model Training Error", "Please split the data first")
 
 
# Function to evaluate models
def evaluate_models():
   global X_test, y_test, lr_model, rf_model
   lr_predictions = lr_model.predict(X_test)
   rf_predictions = rf_model.predict(X_test)
   lr_mae = mean_absolute_error(y_test, lr_predictions)
   rf_mae = mean_absolute_error(y_test, rf_predictions)
   lr_mse = mean_squared_error(y_test, lr_predictions)
   rf_mse = mean_squared_error(y_test, rf_predictions)
   lr_r2 = r2_score(y_test, lr_predictions)
   rf_r2 = r2_score(y_test, rf_predictions)
   results = f"""
   Linear Regression MAE: {lr_mae:.4f}
   Random Forest MAE: {rf_mae:.4f}
   Linear Regression MSE: {lr_mse:.4f}
   Random Forest MSE: {rf_mse:.4f}
   Linear Regression R2 Score: {lr_r2:.4f}
   Random Forest R2 Score: {rf_r2:.4f}
   """
   messagebox.showinfo("Model Evaluation", results)
 
 
# Function to predict reservoir property
def predict_property():
   global X
   user_input = entry.get()
   try:
       input_array = np.array([float(i) for i in user_input.split(',')]).reshape(1, -1)
       if input_array.shape[1] != X.shape[1]:
           messagebox.showerror("Input Error", "Incorrect number of inputs. Please enter data for all features: GR, ILD, PHI, RHOB, NPHI, RT.")
           return
       prediction = rf_model.predict(input_array)
       messagebox.showinfo("Prediction", f"Predicted Property: {prediction[0]:.4f}")
   except Exception as e:
       messagebox.showerror("Input Error", f"Error in input data: {e}")
 
 
# Main window setup
root = Tk()
root.title("Predict Oil Reservoir Properties Using Machine Learning - The Pycodes")
root.geometry("600x400")
 
 
upload_btn = Button(root, text="Upload CSV File", command=upload_file)
upload_btn.pack(pady=10)
 
 
heatmap_btn = Button(root, text="Show Correlation Heatmap", command=show_heatmap)
heatmap_btn.pack(pady=10)
 
 
split_btn = Button(root, text="Split Data", command=split_data)
split_btn.pack(pady=10)
 
 
train_btn = Button(root, text="Train Models", command=build_models)
train_btn.pack(pady=10)
 
 
label = Label(root, text="Enter well log data (comma-separated): GR, ILD, PHI, RHOB, NPHI, RT")
label.pack(pady=10)
 
 
entry = Entry(root, width=50)
entry.pack(pady=10)
 
 
predict_btn = Button(root, text="Predict", command=predict_property)
predict_btn.pack(pady=10)
 
 
show_data_btn = Button(root, text="Show Data", command=show_data)
show_data_btn.pack(pady=10)
 
 
root.mainloop()
