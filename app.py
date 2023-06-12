from flask import Flask,request,render_template,jsonify
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            area=float(request.form.get('area')),
            perimeter = float(request.form.get('perimeter')),
            compactness = float(request.form.get('compactness')),
            length_kernel = float(request.form.get('length_kernel')),
            width_kernel = float(request.form.get('width_kernel')),
            asymmetry_coeff = float(request.form.get('asymmetry_coeff')),
            length_of_kernel_groove = request.form.get('length_of_kernel_groove')
            
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        result=pred

        #return render_template('result.html',final_result=result)

        if result == 1:
            return render_template("result.html",final_result = "Variety(classification) of wheat is Kama")
        elif result ==2:
            return render_template("result.html",final_result = "Variety(classification) of wheat is Rosa")
        elif result == 3:
            return render_template("result.html",final_result = "Variety(classification) of wheat is Canadian")






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True,port=5000)