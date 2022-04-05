# import csv
# import pandas
# from tqdm import tqdm

URL_PATH = "https://raw.githubusercontent.com/MohsenShamas/TopicalChatDataset/main/translated_dataset/preprocessed.csv"

# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Arabic Empathetic Dialogues"""

from __future__ import absolute_import, division, print_function

import csv

import datasets
from arabert.preprocess import ArabertPreprocessor

model_name="bert-base-arabert"
arabert_prep = ArabertPreprocessor(model_name=model_name, keep_emojis=False)

_DESCRIPTION = """\
The Arabic TopicalChat dataset contains ~188K samples of open domain utterances and responses with defined sentiment.
"context" refers to the utterance
"response" refers to the response
"sentiment" refers to the emotion in the utterance
"""

_URL = "https://raw.githubusercontent.com/MohsenShamas/TopicalChatDataset/main/translated_dataset/preprocessed.csv"


class ArabicTopicalChat(datasets.GeneratorBasedBuilder):
    """ArabicTopicalChat: A Dataset of Open-Domain Human-to-Chatbot Dialogues"""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="arabic_topical_chat",
            version=datasets.Version("1.0.0"),
            description="Full training set",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "context": datasets.Value("string"), "response": datasets.Value("string"), "sentiment": datasets.Value("string"),
                }),
            supervised_keys=None,
            homepage="https://github.com/MohsenShamas/TopicalChatDataset",
        )

    def _split_generators(self, dl_manager):
        downloaded_file = dl_manager.download(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_file},
            ),
        ]

    def _generate_examples(self, filepath):
        with open("arabic-empathetic-conversations.csv",'r',encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for i , row in enumerate(csv_reader):
                if i==0:
                    continue
                yield i-1, {
                    "context": arabert_prep.preprocess(row[0]),
                    "response": arabert_prep.preprocess(row[1]),
                    "sentiment": row[2]
                }