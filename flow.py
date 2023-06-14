from jina import Flow,Executor,requests,DocumentArray
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image
from Utils import dbimutils
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
import torch
from tqdm import tqdm
from pprint import pprint
from text2vec import SentenceModel, EncoderType

class PreImage(Executor):
    def __init__(self, device: str = 'cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_repo = 'SmilingWolf/wd-v1-4-swinv2-tagger-v2'
        model_filename = 'model.onnx'
        hf_token = 'hf_JMxVsgXZvlELeRCAdrsk CHCSTXaLGLwuvA'
        path = huggingface_hub.hf_hub_download(
            model_repo, model_filename, use_auth_token=hf_token
        )
        self.model = rt.InferenceSession(path)
        label_filename = 'selected_tags.csv'
        path = huggingface_hub.hf_hub_download(
            model_repo, label_filename, use_auth_token=hf_token
        )
        df = pd.read_csv(path)
        self.tag_names = df["name"].tolist()
        self.rating_indexes = list(np.where(df["category"] == 9)[0])
        self.general_indexes = list(np.where(df["category"] == 0)[0])
        self.character_indexes = list(np.where(df["category"] == 4)[0])
        # print(f'tag_names', self.tag_names)
        # print(f'rating_indexes', self.rating_indexes)
        # print(f'general_indexes', self.general_indexes)
        # print(f'character_indexes', self.character_indexes)

    @requests
    def predict(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            general_threshold = 0.5
            character_threshold = 0.85
            img_path = doc.uri
            print(img_path)
            image = Image.open(doc.uri)
            rawimage = image
            _, height, width, _ = self.model.get_inputs()[0].shape

            # Alpha to white
            image = image.convert("RGBA")
            new_image = Image.new("RGBA", image.size, "WHITE")# 创建一个新的RGBA图像并将所有像素设置为白色
            new_image.paste(image, mask=image)# 使用原始图像作为蒙版将所有不透明像素从新图像复制到新图像
            image = new_image.convert("RGB")# 将图像转换为RGB模式
            image = np.asarray(image) # 将图像转换为NumPy数组
            # print(image)

            # PIL RGB to OpenCV BGR
            image = image[:, :, ::-1]

            image = dbimutils.make_square(image, height)
            image = dbimutils.smart_resize(image, height)
            image = image.astype(np.float32)
            image = np.expand_dims(image, 0)

            input_name = self.model.get_inputs()[0].name
            label_name = self.model.get_outputs()[0].name
            probs = self.model.run([label_name], {input_name: image})[0]

            labels = list(zip(self.tag_names, probs[0].astype(float)))
            # labels = list(zip(self.tag_names, probs[0].astype(float)))[:10]
            # print(labels)
            # ratings_names = [labels[i] for i in self.rating_indexes]
            # rating = dict(ratings_names)
            # print("===========================")
            # print(rating)

            # Then we have general tags: pick anywhere prediction confidence > threshold
            general_names = [labels[i] for i in self.general_indexes]
            general_res = [x for x in general_names if x[1] > general_threshold]
            general_res = dict(general_res)
            tags_str = ','.join([tag for tag in general_res.keys()])

            b = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
            a = (
                ", ".join(list(b.keys()))
                .replace("_", " ")
                .replace("(", "\(")
                .replace(")", "\)")
            )
            print(a)
            doc.text = a
            print("===========================")
            print(general_res)

            # Everything else is characters: pick any where prediction confidence > threshold
            # character_names = [labels[i] for i in self.character_indexes]
            # character_res = [x for x in character_names if x[1] > character_threshold]
            # character_res = dict(character_res)
            # print("===========================")
            # print(character_res)
    @requests(on="/search")
    def search(self, docs: DocumentArray, **kwargs):
        return docs


class EnglishToChineseTranslator(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode_name = 'liam168/trans-opus-mt-en-zh'
        model = AutoModelWithLMHead.from_pretrained(mode_name)
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name)
        self.translation = pipeline("translation_en_to_zh", model=model, tokenizer=self.tokenizer)

    @requests
    def encode(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            # 执行翻译并将结果添加到文档
            translated_text = self.translation(doc.text, max_length=400)[0]['translation_text']
            doc.text = translated_text
            print(doc.summary())
            print(doc.text)

    @requests(on="/search")
    def search(self, docs: DocumentArray, **kwargs):
        return docs

class TextEncoder(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model = SentenceModel("shibing624/text2vec-base-chinese", encoder_type=EncoderType.FIRST_LAST_AVG,
                                    device=device)
        self._da = DocumentArray(
            storage='sqlite', config={'connection': 'example.db', 'table_name': 'images'}
        )

    @requests
    def bar(self, docs: DocumentArray, **kwargs):
        print('start to index')
        print(f"Length of da_encode is {len(self._da)}")
        for doc in tqdm(docs):
            doc.embedding = self._model.encode(doc.text)
            print(doc.summary())
            with self._da:
                self._da.append(doc)
                self._da.sync()
                print(f"Length of da_encode is {len(self._da)}")
        print(f"Length of da_encode is {len(self._da)}")

    @requests(on='/search')
    def search(self, docs: DocumentArray, **kwargs):
        self._da.sync()  # Call the sync method
        print(f"Length of da_search is {len(self._da)}")
        for doc in docs:
            doc.embedding = self._model.encode(doc.text)
            print(doc.text)
            print(doc.summary())
            doc.match(self._da, limit=6, exclude_self=True, metric='cos', use_scipy=True)
            pprint(doc.matches[:, ('text', 'uri', 'scores__cos')])


f = Flow().config_gateway(protocol='http', port=12345) \
    .add(name='predict', uses=PreImage, timeout_ready=36000000) \
    .add(name='translate', uses=EnglishToChineseTranslator, needs='predict') \
    .add(name='text_encoder', uses=TextEncoder, needs='translate')

with f:
    f.block()