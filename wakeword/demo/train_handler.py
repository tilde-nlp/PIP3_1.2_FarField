import tornado.web  #provides a simple web framework with asynchronous features
import json
import os
import os.path

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/multilingual_kws/")
from multilingual_kws.embedding import transfer_learning, input_data

import tensorflow as tf
import numpy as np
from pathlib import Path
import subprocess
import csv
import glob
import shutil
import sklearn.preprocessing
import random
from pydub import AudioSegment
from audiomentations import Compose, AddGaussianNoise, ApplyImpulseResponse, RoomSimulator, Gain, PitchShift, Normalize, TimeStretch
import librosa
import soundfile as sf
import shutil
from distiller import Distiller
from tensorflow import keras
from keras import layers, models


class TrainHandler(tornado.web.RequestHandler):
        
    def post(self):
        data = json.loads(self.request.body)
        KEYWORD = data.get("name")
        
        model_path = f"./models/{KEYWORD}"
        if (os.path.exists(model_path)):
            self.set_status(500)
            self.set_header("Content-Type", "text/plain")
            self.write("Error: model already exists")
            return
        
        input_files = glob.glob(data.get('path') + "/*.wav")
        
        negative_examples = self.get_negative_examples(input_files, KEYWORD)
        negative_count = len(negative_examples)
        
        positive_examples = self.get_positive_examples(input_files, KEYWORD)
        positive_count = len(positive_examples)
        
        # repeat positive examples
        positive_examples = positive_examples * (negative_count // positive_count)
        
        print(f"Positive examples: {len(positive_examples)}; Negative examples: {negative_count}")
        
        base_model_path = "models/multilingual_context_73_0.8011"
        
        background_noise = "./dataset/_background_noise_/"

        print("---Training model---")
        model_settings = input_data.standard_microspeech_model_settings(3)
        _, model, _ = transfer_learning.transfer_learn(
            target=KEYWORD,
            train_files=positive_examples,
            val_files=[positive_examples[0]],
            unknown_files=negative_examples,
            num_epochs=4,
            #num_epochs=8,
            num_batches=1,
            batch_size=64,
            primary_lr=0.001,
            #backprop_into_embedding=False,
            #embedding_lr=0,
            backprop_into_embedding=True,
            embedding_lr=0.001,
            model_settings=model_settings,
            base_model_path=base_model_path,
            base_model_output="dense_2",
            UNKNOWN_PERCENTAGE=50.0,
            bg_datadir=background_noise,
            csvlog_dest=None,
        )
        
        
        avg_threshold = self.test_model(model, input_files)  
        threshold = avg_threshold - 0.01
        model.save(model_path)
        
        self.set_header("Content-Type", "application/json")
        if ("distill" in data and data.get("distill")):
            distilled_model_path, distilled_model = self.distill_model(model, base_model_path, model_settings, background_noise, positive_examples, negative_examples, KEYWORD)            
            avg_distilled_threshold = self.test_model(distilled_model, input_files)            
            distilled_threshold = avg_distilled_threshold - 0.01
            self.write({'models': [{'path': model_path, 'threshold': threshold}, {'path': distilled_model_path, 'threshold': distilled_threshold}]})
        else:
            self.write({'models': [{'path': model_path, 'threshold': threshold}]})
        
        shutil.rmtree(f"./dataset/user/{KEYWORD}")
        
    def get_negative_examples(self, input_files, keyword):
        voice_noise = glob.glob("./dataset/noise/*.wav")
        mswc_words = glob.glob("./dataset/mswc/*/*.wav")
        random.shuffle(mswc_words)
        
        mswc_selection = mswc_words[0:20]
        ww_parts2 = self.generate_negative_examples_from_ww(input_files, 2, keyword)
        ww_parts4 = self.generate_negative_examples_from_ww(input_files, 4, keyword)
        
        negative_examples = mswc_words + ww_parts2 + ww_parts4 + voice_noise
        random.shuffle(negative_examples)        
        return negative_examples
        
    # split wakeword example into 4 equal parts, pad them with silence and save
    def generate_negative_examples_from_ww(self, input_files, split_parts, keyword):
        audio_length_ms = 1000
        split_size = audio_length_ms / split_parts
        output_dir = f"./dataset/user/{keyword}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for ww in input_files:
            dir_name, file_name = os.path.split(ww)
            sound = AudioSegment.from_file(ww)
            for i in range(0, split_parts):
                start = i * split_size
                end = start + split_size
                sound_part = AudioSegment.silent(duration=(audio_length_ms - split_size), frame_rate=16000) + sound[start:end]
                sound_part.export(f"{output_dir}/{i}_{split_parts}_{file_name}", format="wav")
                
        return glob.glob(f"{output_dir}/*.wav")
        
    def get_positive_examples(self, input_files, keyword):
        augmented_wakewords = self.generate_augmented_wakeword_examples(input_files, keyword)
        positive_examples = input_files + augmented_wakewords
        random.shuffle(positive_examples)
        return positive_examples
        
    def generate_augmented_wakeword_examples(self, input_files, keyword):
        output_dir = f"./dataset/user/{keyword}/augmented"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        augmenter_noise = Compose([
            AddGaussianNoise(p=1)
        ])
        
        augmenter_noise_gain = Compose([
            AddGaussianNoise(p=1),
            Gain(min_gain_in_db=-30, max_gain_in_db=-20, p=1)
        ])
        
        augmenter_noise_gain_ir = Compose([
            AddGaussianNoise(p=1),
            Gain(min_gain_in_db=-30, max_gain_in_db=-20, p=1),
            ApplyImpulseResponse(ir_path="./dataset/rirs", leave_length_unchanged=True, p=1)
        ])
        
        augmenter_pitch_shift = Compose([
            PitchShift(p=1)                             
        ])

        augmenter_normalize = Compose([
            Normalize(p=1)                             
        ])

        augmenter_time_stretch = Compose([
            TimeStretch(leave_length_unchanged=True, min_rate=1, max_rate=1.5, p=1)                         
        ])
        
        def augment(signal, sample_rate, augmenter, output_file, check_existing=False):
          Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
          if check_existing and glob.glob(output_file):
            return
          success = False
          while (not success):
            try:
              augmented = augmenter(signal, sample_rate)
              success = True
            except IndexError:
              pass
          #cut to 1s
          augmented = augmented[:sample_rate]
          sf.write(output_file, augmented, sample_rate)
          
        for input_file in input_files:
          signal, sample_rate = librosa.load(input_file, sr=16000)
          dir_name, file_name = os.path.split(input_file)
          augment(signal, sample_rate, augmenter_noise, f"{output_dir}/noise/{file_name}")
          augment(signal, sample_rate, augmenter_noise_gain, f"{output_dir}/noise_gain/{file_name}")
          augment(signal, sample_rate, augmenter_noise_gain_ir, f"{output_dir}/noise_gain_ir/{file_name}")
          augment(signal, sample_rate, augmenter_pitch_shift, f"{output_dir}/pitch/{file_name}")
          augment(signal, sample_rate, augmenter_normalize, f"{output_dir}/normalize/{file_name}")
          augment(signal, sample_rate, augmenter_time_stretch, f"{output_dir}/time_stretch/{file_name}")
        
        return glob.glob(f"{output_dir}/*/*.wav")
        
    def distill_model(self, teacher, base_model_path, model_settings, background_noise_dir, positive_examples, negative_examples, keyword):
        print("Distilling model")
        trained_base_model = tf.keras.models.load_model(base_model_path)

        x = trained_base_model.layers[-6].output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation="relu")(x)
        # layers.Dropout(0.5)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, kernel_initializer="lecun_normal", activation="selu")(x)
        # must use alpha-dropout if dropout is desired with selu
        logits = layers.Dense(3)(x)

        embedding_model = models.Model(inputs=trained_base_model.input, outputs=logits)
        embedding_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        xfer = tf.keras.models.Model(
            name="EmbeddingModel",
            inputs=embedding_model.inputs,
            outputs=embedding_model.layers[-1].output,
        )
        xfer.trainable = True

        # dont use softmax unless losses from_logits=False
        CATEGORIES = 3  # silence + unknown + target_keyword
        student = tf.keras.models.Sequential(
            [
                xfer,
                tf.keras.layers.Dense(units=18, activation="tanh"),
                tf.keras.layers.Dense(units=CATEGORIES, activation="softmax"),
            ],
            name="student"
        )
        
        distiller = Distiller(student=student, teacher=teacher)
        distiller.compile(
            optimizer=keras.optimizers.Adam(),
            #metrics=[keras.metrics.SparseCategoricalAccuracy()],
            metrics=["accuracy"],
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=0.1,
            temperature=10,
        )
        
        audio_dataset = input_data.AudioDataset(
            model_settings=model_settings,
            commands=[keyword],
            background_data_dir=background_noise_dir,
            unknown_files=negative_examples,
            unknown_percentage=50.0,
            spec_aug_params=input_data.SpecAugParams(percentage=80),
        )

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        init_train_ds = audio_dataset.init_single_target(
            AUTOTUNE, positive_examples, is_training=True
        )
        init_val_ds = audio_dataset.init_single_target(
            AUTOTUNE, [positive_examples[0]], is_training=False
        )
        train_ds = init_train_ds.shuffle(buffer_size=1000).repeat().batch(64)
        val_ds = init_val_ds.batch(64)

        small_model = distiller.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=64 * 1,
            #epochs=4,
            epochs=8,
            callbacks=[],
            verbose=True,
        )        
        
        output_path = f"./models/{keyword}_distilled"
        distiller.student.save(output_path)
        
        return output_path, distiller.student
        
    def test_model(self, model, test_examples):
        settings = input_data.standard_microspeech_model_settings(label_count=1)
        test_spectrograms = np.array([input_data.file2spec(settings, str(f)) for f in test_examples])
        test_predictions = model.predict(test_spectrograms)
        probabilities = list(map(lambda p: p[2], test_predictions))
        average = sum(probabilities) / len(probabilities)
        print(f"Probabilities {probabilities}; Average: {average}")
        return average
        
