import tornado.escape  #for escaping/unescaping methods for HTML, JSON, URLs, etc
import tornado.ioloop  #event loop for non-blocking sockets
import tornado.options  #command line parsing module to define options
import tornado.web  #provides a simple web framework with asynchronous featuresfrom tornado.options import define, options
import tornado.websocket  #implementation of the WebSocket protocol and bidirectional communication
import logging
import os.path
import uuid
import websockets
import sys
import ssl
import logging
import tensorflow as tf
import numpy as np
import scipy.signal as sps
import io
import soundfile as sf
import json
import train_handler
import random
import string
from pathlib import Path
from tornado.options import define, options
from sklearn.metrics import auc
import glob

define("port", default=8080, type=int)  #setting the port to 8000 which can easily be changed

sys.path.append("./multilingual_kws/")
from multilingual_kws.embedding import transfer_learning, input_data

settings = None
silence_threshold = 20 / 32767    
sample_rate=16000

model_data = {
    "LV_small": {
        "path": "models/TildePls_lv_4568shot_distilled",
        "wakeword": "Tilde, lūdzu",
        "displayName": "LV-small ('Tilde, lūdzu')",
        "threshold": 98.5
    },
    "LT_small": {
        "path": "models/TildePls_lt_4044shot_distilled",
        "wakeword": "Tilde, prašau",
        "displayName": "LT-small ('Tilde, prašau')",
        "threshold": 99.0
    },
    "ET_small": {
        "path": "models/TildePls_et_4044shot_distilled",
        "wakeword": "Tilde, palun",
        "displayName": "ET-small ('Tilde, palun')",
        "threshold": 99.0
    },
    "EN_small": {
        "path": "models/TildePls_en_4200shot_distilled",
        "wakeword": "Tilde, please",
        "displayName": "EN-small ('Tilde, please')",
        "threshold": 99.0
    },
    "RU_small": {
        "path": "models/TildePls_ru_4200shot_distilled",
        "wakeword": "Тилдe, пожалуйста",
        "displayName": "RU-small ('Тилдe, пожалуйста')",
        "threshold": 99.0
    }
}

models = {
    # "LV": tf.keras.models.load_model('models/TildePls_lv_4568shot'),
    # "LT": tf.keras.models.load_model('models/TildePls_lt_4044shot'),
    # "ET": tf.keras.models.load_model('models/TildePls_et_4044shot'),
    # "EN": tf.keras.models.load_model('models/TildePls_en_4200shot'),
    # "RU": tf.keras.models.load_model('models/TildePls_ru_4200shot'),
    "LV_small": tf.keras.models.load_model('models/TildePls_lv_4568shot_distilled'),
    "LT_small": tf.keras.models.load_model('models/TildePls_lt_4044shot_distilled'),
    "ET_small": tf.keras.models.load_model('models/TildePls_et_4044shot_distilled'),
    "EN_small": tf.keras.models.load_model('models/TildePls_en_4200shot_distilled'),
    "RU_small": tf.keras.models.load_model('models/TildePls_ru_4200shot_distilled')
}

class Application(tornado.web.Application):  #setting a tornado class as a web application
    def __init__(self):
        handlers = [(r"/", MainHandler), 
                    (r"/websocket", WakeWordSocketHandler), 
                    (r"/train", train_handler.TrainHandler), 
                    (r"/model", ModelHandler),
                    (r"/record", RecordHandler),
                    (r"/record/(.*)", tornado.web.StaticFileHandler, {
                        "path": "templates/recorder/"
                    }),
                    (r"/upload", UploadHandler),
                    (r"/evaluate", EvaluateHandler)]
        settings = dict(
            cookie_secret="szLzTfFYkh7bm)9L@NpnRgLpdE!KHC",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),  #providing the templates path
            static_path=os.path.join(os.path.dirname(__file__), "static"),  #providing the static folder's path
            xsrf_cookies=True,
        )
        super().__init__(handlers, **settings)

class RecordHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("recorder/index.html")

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class ModelHandler(tornado.web.RequestHandler):
    def get(self):
        self.write(model_data)
    
    def post(self):
        data = json.loads(self.request.body)
        wakeword = data.get("name")
        new_models = data.get("models")
        for model in new_models:
            path = model.get("path")
            print(f"Registering model {path}")
            key = path.split("/")[-1].replace(" ", "_")
            model_data[key] = {
                "path": path,
                "wakeword": wakeword,
                "displayName": f"{key} ('{wakeword}')",
                "threshold": model.get("threshold") * 100
            }
            models[key] = tf.keras.models.load_model(path)
        self.write(f"Registered {len(new_models)} models")

class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        wakeword = self.get_argument("ww")
        output_dir = f"./dataset/user/{wakeword}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for file_field in self.request.files.keys():
            file = self.request.files[file_field][0]
            original_fname = file['filename']
            extension = os.path.splitext(original_fname)[1]
            fname = ''.join(random.choice(string.ascii_lowercase + string.digits) for x in range(6))
            final_filename = fname + extension
            output_file = open(f"{output_dir}/{final_filename}", "wb")
            output_file.write(file['body'])
            
        self.finish(output_dir)
        
class EvaluateHandler(tornado.web.RequestHandler):
    def get(self):
        THRESHOLD_STEP=0.0001
        THRESHOLD_START=0.001
        THRESHOLD_END=0.999
        langcode = self.get_argument("langcode", "lv")
        model_id = self.get_argument("model", None)
        
        prediction_model = models[model_id]
        
        settings = input_data.standard_microspeech_model_settings(label_count=1)
        
        original_positive = glob.glob(f".\\dataset\\dataset_processed\\augmented\\*\\converted\\wakeword\\dataset\\{langcode}\\positive\\*.wav")
        augmented_positive = glob.glob(f".\\dataset\\dataset_processed\\converted\\wakeword\\dataset\\{langcode}\\positive\\*.wav")
        positive_set = original_positive + augmented_positive
        positive_set = [os.path.normpath(path) for path in positive_set]
        positive_set_spectrograms = np.array([input_data.file2spec(settings, str(f)) for f in positive_set])
        positive_set_predictions = prediction_model.predict(positive_set_spectrograms)
        
        all_processed = glob.glob("./dataset/dataset_processed/**/*.wav", recursive=True)        
        all_processed = [os.path.normpath(path) for path in all_processed]
        negative_set = [wav for wav in all_processed if wav not in positive_set]        
        negative_set_spectrograms = np.array([input_data.file2spec(settings, str(f)) for f in negative_set])
        negative_set_predictions = prediction_model.predict(negative_set_spectrograms)
        
        false_alarms = []
        false_rejects = []
        true_positives = []
        
        for threshold in np.arange(THRESHOLD_START, THRESHOLD_END, THRESHOLD_STEP):
          negative_count = 0
          for p in positive_set_predictions:
            if p[2] < threshold:
              negative_count += 1
          false_negative_rate = negative_count / len(positive_set)
          false_rejects.append(false_negative_rate)
          true_positives.append(1-false_negative_rate)

          positive_count = 0
          for p in negative_set_predictions:
            if p[2] >= threshold:
              positive_count += 1

          false_positive_rate = positive_count / len(negative_set)
          false_alarms.append(false_positive_rate)
        
        auc_val = auc(false_alarms, false_rejects)
        self.write({"auc": auc_val, "test_set_examples_positive": len(positive_set), "test_set_examples_negative": len(negative_set), "false_alarms": false_alarms, "false_rejects": false_rejects})

class WakeWordSocketHandler(tornado.websocket.WebSocketHandler):  #creating our main websocket class

    def get_spectrogram(self, data):
        return input_data.to_micro_spectrogram(settings, data)

    def detect_triggerword_spectrum(self, model, x):
        return model.predict(np.array([x]))

    def on_message(self, message):
        obj = json.loads(message)        
        audio_bytes = bytes(obj["audio"])
        audio, message_sample_rate = sf.read(io.BytesIO(audio_bytes))
        if np.abs(audio).mean() < silence_threshold:
            logging.info("Signal too quiet")
            self.write_message("0")
        else:
            if message_sample_rate != sample_rate:
                samples = round(len(audio) * float(sample_rate) / message_sample_rate)
                audio = sps.resample(audio, samples)
            
            spectrogram = self.get_spectrogram(audio)
            predictions = self.detect_triggerword_spectrum(models[obj["model"]], spectrogram)
            wakeword_detected_pred = predictions[0][2]
            logging.info(f"Prediction: {wakeword_detected_pred}")
            self.write_message(str(wakeword_detected_pred))

    def open(self):
        logging.info("WebSocket opened")

    def on_close(self):
        logging.info("WebSocket closed")
		
def main():  #and to close up everything
    tornado.options.parse_command_line()
    app = Application()

    # start HTTPS
    ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_ctx.load_cert_chain("dummy.cert",
                            "dummy.key")
    https_server = tornado.httpserver.HTTPServer(app, ssl_options=ssl_ctx)

    https_server.bind(8083)
    https_server.start(1)

    app.listen(options.port)
    
    global settings
    settings = input_data.standard_microspeech_model_settings(label_count=1)
    logging.info("Started")
    
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
