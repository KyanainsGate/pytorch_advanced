# pytorch_advanced

- Unofficial update version of `つくりながら学ぶ! PyTorchによる発展ディープラーニング` for Ampere Architecture GPUs
- オリジナルがUbuntu 16.04なので、最近のもの（WSL2）に追従

## 環境構築
- checkoutする
    ```sh
    git checkout main_20.04
    ```

- Anacondaの利用を推奨(`conda install`のほうが安定するケースが多いため)
- また`torchvision`と`torchtext`がなかなか共存しない(2022 Feb.現在)
    - 自動でinstallされる`torch`のversionが一致しないためと予想
    - 仮想環境レベルで分けることを推奨。例えば、torchvision用にkernel`cv`を
    ```bash
    # 1. Create new virtual environment
    conda create -n cv python=3.9
    # 2. After Created virtual environment
    conda install -c conda-forge torchvision # For CV chapter
    # 3. Install pip packages for running Jupyter
    pip install ipykernel notebook environment_kernels
    # 4. Create jupyter's kernel
    ipython kernel install --user --name=cv --display-name=cv
    ```

    - torchtext用にはkernel`nlp`を作成
    ```sh
    # 1. Create new virtual environment
    conda create -n nlp python=3.9
    # 2. After Created virtual environment
    conda install -c pytorch torchtext # For NLP chapter
    # 3. Install pip packages for running Jupyter
    pip install -r requreiments.txt
    # 4. Create jupyter's kernel
    ipython kernel install --user --name=nlp --display-name=nlp
    ```

    - Jupyter上で適切に選択（`起動 -> [Kernel] -> [Change Kernel] -> [cv] or [nlp]`）して使用

- 最終的には以下の構成でテスト済み
    - 共通
        - OS: Ubuntu 20.04 @ WSL2 (Windows 11)
        - CUDA: 11.1 (ただし [CUDA on Windows Subsystem for Linux (WSL)](https://developer.nvidia.com/cuda/wsl) )
        - Python: 3.9.7
    - 環境ごと

      | Env. name | PyTorch Dependent Framework | PyTorch Ver.  | Chapter | 
      | ------ | ------ | ------ | ------ | 
      | `cv` | torchvision == `0.10.0a0` | 1.9.0 | Except 7 and 8 | 
      | `nlp` | torchtext == `0.11.2` | 1.10.2 | 7 and 8 | 

## 主なソース変更点

- Ch.1: `ipywidgets`不調時のトラブルシューティングコメント挿入
- Ch.5&6: 互換性対応(`sklearn`, `nn.BCEWithLogitsLoss()`の引数型対応)
- Ch.7&8: Download Link更新、互換性対応(`MeCab`install方法, `torchtext`legacy化, `gensim`, ...)
- Ch.9: kinetics dataset自動DL用のrequirements.txt追加

# 以下公式のREADME

## つくりながら学ぶ! PyTorchによる発展ディープラーニング

<div align="center">
<img src="./etc/pytorch_advanced.png" alt="PyTorchによる発展ディープラーニング" title="PyTorchによる発展ディープラーニング" width=60%>
</div>

<br>

[書籍「つくりながら学ぶ! PyTorchによる発展ディープラーニング」（小川雄太郎、マイナビ出版 、19/07/29) ](https://www.amazon.co.jp/dp/4839970254/)

のサポートリポジトリです。

<br>

### 1. 本書で扱うタスク内容とディープラーニングモデル

- 第1章 画像分類と転移学習（VGG）
- 第2章 物体認識（SSD）
- 第3章 セマンティックセグメンテーション（PSPNet）
- 第4章 姿勢推定（OpenPose）
- 第5章 GANによる画像生成（DCGAN、Self-Attention GAN）
- 第6章 GANによる異常検知（AnoGAN、Efficient GAN)
- 第7章 自然言語処理による感情分析（Transformer）
- 第8章 自然言語処理による感情分析（BERT）
- 第9章 動画分類（3DCNN、ECO）

本書の詳細な内容については、以下で別途、解説しております。

[「Qiita記事：PyTorchによる発展ディープラーニング、各章の紹介」](https://qiita.com/sugulu/items/07253d12b1fc72e16aba)

<br>

### 2. 疑問点・修正点はIssueにて管理しています

本GitHubのIssueにて、疑問点や修正点を管理しています。

不明な点などがございましたら、こちらをご覧ください。

https://github.com/YutaroOgawa/pytorch_advanced/issues

### 3. 誤植について

書籍中の誤植一覧はこちらになります。 大変申し訳ございません。

[誤植一覧](https://github.com/YutaroOgawa/pytorch_advanced/issues?q=is%3Aissue+is%3Aopen+label%3A%E8%AA%A4%E6%A4%8D)
