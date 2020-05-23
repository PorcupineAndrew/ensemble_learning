# Ensemble Learning

#### 作者： 陈熠豪

#### 学号：2017011486

#### 邮箱： chenyiha17@mails.tsinghua.edu.cn

## 如何复现

-   建立运行环境

    -   下面提供一个基于 docker 建立运行环境的方法。你也可以使用任何其他方式构建运行环境，注意本仓库需要 Python3.7 及以上，所需的 package 请见 `./docker/requirements.txt`

    -   如果使用 docker，请注意替换 docker 源为国内源，否则可能导致镜像拉取失败

    -   运行下面的命令，构建一个 docker image。这个脚本命令会拉取 Python3 镜像，替换 Apt 和 pip3 源，安装本仓库所需的所有 package，并建立必要的用户和用户组

        ```bash
        ./docker/docker_script.sh build
        ```

    -   运行下面的命令，建立运行时 container。这个脚本命令会使用刚刚构建的镜像来建立容器，并运行 bash。得到的容器会挂载本仓库的文件目录，可以直接交互，容器在退出后会自动删除

        ```bash
        ./docker/docker_script.sh run
        ```

-   放置数据文件

    -   请将数据文件放置在 `./data` 下

        ```
        data
        ├── train.csv
        └── test.csv
        ```

-   运行实验

    -   运行下面的命令，得到 bagging + svm 结果

        ```
        ./src/main.py -d 300 -t svm -c 10 -f bagging -n {NUM_WORKERS} -o {OUTPUT_PATH}
        ```

        其中 NUM_WORKER 表示多进程的最大并行数，请根据运行机器的实际情况选择合适的数值

        OUTPUT_PATH 为输出文件的路径

    -   运行下面的命令，得到 bagging + dt 结果

        ```
        ./src/main.py -d 300 -t dt -c 10 -f bagging -n {NUM_WORKERS} -o {OUTPUT_PATH}
        ```

        其中 NUM_WORKER 表示多进程的最大并行数，请根据运行机器的实际情况选择合适的数值

        OUTPUT_PATH 为输出文件的路径

    -   运行下面的命令，得到 adaboost + svm 结果

        ```
        ./src/main.py -d 300 -t svm -i 20 -f adaboost -n {NUM_WORKERS} -o {OUTPUT_PATH}
        ```

        其中 NUM_WORKER 表示多进程的最大并行数，请根据运行机器的实际情况选择合适的数值

        OUTPUT_PATH 为输出文件的路径

    -   运行下面的命令，得到 adaboost + dt 结果

        ```
        ./src/main.py -d 300 -t dt -i 20 -f adaboost -n {NUM_WORKERS} -o {OUTPUT_PATH}
        ```

        其中 NUM_WORKER 表示多进程的最大并行数，请根据运行机器的实际情况选择合适的数值

        OUTPUT_PATH 为输出文件的路径

    -   运行下面的命令，训练 cnn 分类器

        ```
        ./src/cnn/main.py -p {PREFIX}
        ```

        其中 PREFIX 是用于区分的前缀，自定即可

        运行下面的命令，使用训练好的 cnn 分类器进行 predict

        ```
        ./src/cnn/main.py --is-pred -l {MODEL_NAME} -w {WEIGHT_NAME} -o {OUTPUT_PATH}
        ```

        其中 MODEL_NAME 是训练的模型，即 {PREFIX}model.h5

        WEIGHT_NAME 是训练的神经网络权重，即 {PREFIX}weights.best.hdf5

        OUTPUT_PATH 为输出文件的路径

    -   一些参考结果存放于 `./result/`中

        集成学习结果的命名格式为 `{框架}.{迭代次数}.{分类器}.{词向量维度}[.{决策树深度}].csv`

        -   adaboost.20.dt.d300.depth5.csv
        -   adaboost.20.dt.d300.depth10.csv
        -   adaboost.20.dt.d300.depth20.csv
        -   adaboost.20.svm.d300.csv
        -   bagging.10.dt.d300.depth5.csv
        -   bagging.10.dt.d300.depth10.csv
        -   bagging.10.dt.d300.depth20.csv
        -   bagging.10.svm.d300.csv
        -   cnn.summary.csv
        -   cnn.reviewText.csv

## 仓库文件组成

```
.
├── doc
│   └── report.md
├── docker
│   ├── Dockerfile
│   ├── docker_script.sh
│   ├── requirements.txt
│   └── sources.list
├── Readme.md
├── report.pdf
├── result
│   ├── adaboost.20.dt.d300.depth10.csv
│   ├── adaboost.20.dt.d300.depth20.csv
│   ├── adaboost.20.dt.d300.depth5.csv
│   ├── adaboost.20.svm.d300.csv
│   ├── bagging.10.dt.d300.depth10.csv
│   ├── bagging.10.dt.d300.depth20.csv
│   ├── bagging.10.dt.d300.depth5.csv
│   ├── bagging.10.svm.d300.csv
│   ├── cnn.reviewText.csv
│   └── cnn.summary.csv
└── src
    ├── adaboost.py
    ├── bagging.py
    ├── classifier.py
    ├── cnn
    │   ├── main.py
    │   ├── text_cnn.py
    │   └── text_processing_util.py
    ├── data_load.py
    ├── decision_tree.py
    └── main.py
```
