## 環境
macbook air (m3)
Python 3.12.4

## 内容
WebSocketで指差しの画像を取得してどちらの方向を指しているのかを推論して返すWebSocketサーバです。
モデルは事前学習済みのRenNetを転移学習しました。

## 環境構築(pipenvを使用した場合)
このディレクトリで
```
pipenv install
```

modelsというフォルダを作りそこにモデルの重みファイルを配置してください。

重みファイルはonnx形式にしてください。

## 実行方法(pipenvを使用した場合)
pipenv run python server.py

