from jina import Flow,Executor,requests,DocumentArray
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image
from Utils import dbimutils
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForTokenClassification,pipeline,BlipProcessor,BlipForConditionalGeneration,AutoModelForSeq2SeqLM,BertTokenizerFast,AutoModelForCausalLM
import torch
from tqdm import tqdm
from pprint import pprint
from text2vec import SentenceModel, EncoderType
import re
from ckip_transformers import __version__
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker


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
        # path = huggingface_hub.hf_hub_download(
        #     model_repo, label_filename, use_auth_token=hf_token
        # )
        df = pd.read_csv('selected_tags.csv')
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
        print("进入图片预处理步骤================")
        for doc in docs:
            general_threshold = 0.7
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
            new_tags = {re.sub(r'\(.*?\)', '', key.replace('_', ' ')).strip(): value for key, value in b.items()}
            print(f'doc.tags=', new_tags)
            doc.tags = new_tags
            print("===========================")


            # Everything else is characters: pick any where prediction confidence > threshold
            # character_names = [labels[i] for i in self.character_indexes]
            # character_res = [x for x in character_names if x[1] > character_threshold]
            # character_res = dict(character_res)
            # print("===========================")
            # print(character_res)
    @requests(on="/search")
    def search(self, docs: DocumentArray, **kwargs):
        return docs

class Caption(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    @requests
    def caption(self, docs: DocumentArray, **kwargs):
        for doc in tqdm(docs):
            img_path = doc.uri
            print(img_path)
            raw_image = Image.open(img_path).convert('RGB')
            # unconditional image captioning
            inputs = self.processor(raw_image, return_tensors="pt")

            out = self.model.generate(**inputs)
            print(self.processor.decode(out[0], skip_special_tokens=True))
            doc.text = self.processor.decode(out[0], skip_special_tokens=True)

    @requests(on="/search")
    def search(self, docs: DocumentArray, **kwargs):
        return docs

class CaptionToTag(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("wietsedv/xlm-roberta-base-ft-udpos28-en")
        self.model = AutoModelForTokenClassification.from_pretrained("wietsedv/xlm-roberta-base-ft-udpos28-en")

    @requests
    def encode(self, docs: DocumentArray, **kwargs):
        print(f"in CaptionToTag")
        for doc in docs:
            chinese_tag = self.tokenizer(doc.text)
            print(f'chinsese_tag:',chinese_tag)
            print(f'doc.text:',doc.text)
            print(f'doc.tags:',doc.tags)
            print(doc.summary())
    @requests(on="/search")
    def search(self, docs: DocumentArray, **kwargs):
        return docs
class EnglishToChineseTranslator(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode_name = 'kiddyt00/yt-tags-en-zh-v2'
        # mode_name = 'liam168/trans-opus-mt-en-zh'
        # mode_name = 'nkuAlexLee/Pokemon_EN_to_ZH'
        # mode_name = 'Helsinki-NLP/opus-mt-en-zh'
        # model = AutoModelWithLMHead.from_pretrained(mode_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(mode_name)
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name)
        # self.translation = pipeline("translation_en_to_zh", model=model, tokenizer=self.tokenizer)
        self.translation = pipeline("translation_en_to_zh", model=model, tokenizer=self.tokenizer)

    @requests
    def encode(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            # 执行翻译并将结果添加到文档
            out_key = 'translation_text'
            # out_key = 'generated_text'
            translated_text = self.translation(doc.text, max_length=400)[0][out_key]
            for tag_name, tag_value in doc.tags.copy().items():
                print(f'{tag_name}: {tag_value}')
                translated_tag = self.translation(tag_name, max_length=400)[0][out_key]
                doc.tags[translated_tag] = tag_value
                print(f'{translated_tag}: {tag_value}')
            doc.text = translated_text
            print(doc.summary())
            print(doc.text)


    @requests(on="/search")
    def search(self, docs: DocumentArray, **kwargs):
        return docs

class ChineseTextToTag(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # mode_name = 'kiddyt00/yt-tags-en-zh-v2'
        # model = AutoModelForSeq2SeqLM.from_pretrained(mode_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(mode_name)
        # # self.translation = pipeline("translation_en_to_zh", model=model, tokenizer=self.tokenizer)
        #
        # casual language model (GPT2)
        # self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        # self.model = AutoModelForCausalLM.from_pretrained('ckiplab/albert-tiny-chinese-ws')
        # self.translation = pipeline("translation_en_to_zh", model=self.model, tokenizer=self.tokenizer)
        self.ws_driver = CkipWordSegmenter(model="bert-base")
        self.pos_driver = CkipPosTagger(model="bert-base")


    @requests
    def encode(self, docs: DocumentArray, **kwargs):
        print(f"in ChineseTextToTag")
        ws = self.ws_driver(docs.texts)
        pos = self.pos_driver(ws)
        print()

        print(docs.summary())
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
            print("==============in embedding state")

            pre_emedding_text = doc.text
            for key, value in doc.tags.items():
                pre_emedding_text = pre_emedding_text + key
            print(f'pre_emedding_text:',pre_emedding_text)
            doc.embedding = self._model.encode(pre_emedding_text)
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
        # tmp_docs = DocumentArray()
        # for doc in docs:
        #     tag_key = doc.text
        #     if any(key.startswith(tag_key) or tag_key in key for key in doc.tags):
        #         tmp_docs.append(doc)
        #     doc.match(tmp_docs, limit=6, exclude_self=True, metric='cos', use_scipy=True)
        for doc in docs:
            doc.embedding = self._model.encode(doc.text)
            print(doc.text)
            print(doc.summary())
            doc.match(self._da, limit=20, exclude_self=True, metric='cos', use_scipy=True)
            pprint(doc.matches[:, ('text', 'uri', 'scores__cos')])


f = Flow().config_gateway(protocol='http', port=12345) \
    .add(name='predict', uses=PreImage) \
    .add(name='caption', uses=Caption, needs='predict') \
    .add(name='translate', uses=EnglishToChineseTranslator, needs='caption') \
    .add(name='chinesetotag', uses=ChineseTextToTag, needs='translate') \
    .add(name='text_encoder', uses=TextEncoder, needs='chinesetotag')

with f:
    f.block()

# with f:
#     for doc in f.index():
#         if not doc.tags:
#             f.delete(doc)