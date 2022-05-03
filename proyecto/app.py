from flask import Flask, request, render_template
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

# Main function here
@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        clf = joblib.load("clf.pkl")
        
        # Get values through input bars
        Altura = request.form.get("Altura")
        Peso = request.form.get("Peso")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[Altura, Peso]], columns = ["Altura", "Peso"])
        
        # Get prediction 
        prediction = clf.predict(X)[0]
        res="El sexo de la persona es: "+prediction
        
    else:
        res = ""
        
    return render_template("website.html", output = res)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)
