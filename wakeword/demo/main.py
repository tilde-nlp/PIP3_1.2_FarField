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
import logging
import tensorflow as tf
import numpy as np
import scipy.signal as sps
import io
import soundfile as sf
import json
from tornado.options import define, options

define("port", default=8080, type=int)  #setting the port to 8000 which can easily be changed

sys.path.append("./multilingual_kws/")
from multilingual_kws.embedding import transfer_learning, input_data

settings = None
silence_threshold = 20 / 32767    
sample_rate=16000
models = {
    "LV": tf.keras.models.load_model('models/TildePls_lv_4568shot'),
    "LT": tf.keras.models.load_model('models/TildePls_lt_4044shot'),
    "ET": tf.keras.models.load_model('models/TildePls_et_4044shot'),
    "EN": tf.keras.models.load_model('models/TildePls_en_4200shot'),
    "RU": tf.keras.models.load_model('models/TildePls_ru_4200shot'),
    "LV_small": tf.keras.models.load_model('models/TildePls_lv_4568shot_distilled'),
    "LT_small": tf.keras.models.load_model('models/TildePls_lt_4044shot_distilled'),
    "ET_small": tf.keras.models.load_model('models/TildePls_et_4044shot_distilled'),
    "EN_small": tf.keras.models.load_model('models/TildePls_en_4200shot_distilled'),
    "RU_small": tf.keras.models.load_model('models/TildePls_ru_4200shot_distilled')
}  

class Application(tornado.web.Application):  #setting a tornado class as a web application
    def __init__(self):
        handlers = [(r"/", MainHandler), (r"/websocket", WakeWordSocketHandler)] #setting the nesassary urls
        settings = dict(
            cookie_secret="szLzTfFYkh7bm)9L@NpnRgLpdE!KHC",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),  #providing the templates path
            static_path=os.path.join(os.path.dirname(__file__), "static"),  #providing the static folder's path
            xsrf_cookies=True,
        )
        super().__init__(handlers, **settings)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


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
            self.write_message(str(wakeword_detected_pred))

    def open(self):
        logging.info("WebSocket opened")

    def on_close(self):
        logging.info("WebSocket closed")
		
def main():  #and to close up everything
    tornado.options.parse_command_line()
    app = Application()
    app.listen(options.port)
    
    global settings
    settings = input_data.standard_microspeech_model_settings(label_count=1)
    logging.info("Started")
    
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()