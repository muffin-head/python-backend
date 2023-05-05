# from flask import Flask ,jsonify, request
# from flask_cors import CORS
# import work_2
# import work_5
# import work_7
# app = Flask(__name__)
# CORS(app)
# @app.route("/getcoordinates",methods=['POST'])
# def hello():
#     data = request.get_json()
#     gapCoord= work_2.getCoordinates(data)
#     # print("fdn",gapCoord)
#     response = jsonify(gapCoord)
#     response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
#     return response

# if __name__ == "__main__":
#     app.run(debug=True)




# ik-edit below
from flask import Flask ,jsonify, request
from flask_cors import CORS
import work_11

app = Flask(__name__)
CORS(app)
@app.route("/getcoordinates",methods=['POST'])
def hello():
    data = request.get_json() 
    
    gapCoord= work_11.getCoordinates(data['imgData'])
    print('re')
    print("final cords",gapCoord)
    response = jsonify(gapCoord)
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
    return response

if __name__ == "__main__":
    app.run(debug=True)
