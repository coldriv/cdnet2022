# cdnet2022
Change Detection Neural Network [2022]

## チュートリアル
レポジトリをクローンします。

    git clone https://github.com/coldriv/cdnet2022.git  

weightsファイルをダウンロードして一つ階層の上の場所にフォルダごと置いてください。  
Gドライブに保存しています。
[こちら](https://drive.google.com/drive/folders/1ImNeaXaMnB4tZDzrpohfshCgVxkO38gc?usp=sharing/)
。

    python 0.main.py

## 独自の画像を推論
「0.main.py」の実行部分は以下の通りです。ファイルのパスを適宜記述しなおしてください。

    m = predictor(device="cuda",
                  weight_cdnet="../weights/model.pth",
                  weight_yolo5="../weights/yolov5x6.pt",
                  weight_depth="../weights/depth.pt")

    allrun(m,
           p0s = ["../_tmp/20210726145000.jpg",
                  "../_tmp/20210726150001.jpg",
                  "../_tmp/20210726151002.jpg",
                  "../_tmp/20210726152002.jpg"],
           p1  =  "../_tmp/20210726153001.jpg",
           pm  =  "mask/mask_004.jpg")
