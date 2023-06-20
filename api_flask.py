from flask import Flask, jsonify,request
from jina import Client, DocumentArray, Document


app = Flask(__name__)


@app.route('/index', methods=['POST'])
def hello():
    data = request.get_json()
    img_uri = data['img_url']
    host = "http://172.66.1.189:12345"
    doc = Document(uri=img_uri)
    docs = DocumentArray()
    docs.append(doc)
    result_list = []
    # 发送 POST 请求并获取响应数据
    c = Client(host=host)
    ca = c.post(on='/', inputs=docs, show_progress=True, timeout=3600)
    print(ca.summary())
    for a in ca:
        # 提取 tags 属性中的所有键名
        tag_keys = list(a.tags.keys())
        print(f"Tags: {tag_keys}")
        result_list.append({'text': a.text, 'tags': tag_keys})
    return jsonify(result_list)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
