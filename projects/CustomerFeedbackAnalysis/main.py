from src.trigger_training import trigger_training
from src.trigger_prediction import trigger_prediction
from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
from flask_cors import CORS, cross_origin
import os
import json
import flask_monitoringdashboard as dashboard


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')
port = int(os.getenv("PORT", 5000))
app = Flask(__name__)
dashboard.bind(app)
CORS(app)


@app.route("/train", methods=['GET'])
@cross_origin()
def start_training():
    try:
        trigger_training()
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)
    return Response("Training successful!!")


@app.route("/predict", methods=['GET'])  # If sending data via postman change method to POST
@cross_origin()
def start_prediction():
    try:
        trigger_prediction()
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)
    return Response("Predictions done!!")


if __name__ == "__main__":
    host = '0.0.0.0'
    httpd = simple_server.make_server(host, port, app)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
