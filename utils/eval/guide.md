## 运行评测脚本所需参数

1. 第一步 `eval_net.py` 脚本会遍历测试集把每个图片的模型置信度输出与对应的ground truth保存到一个npz文件中

    - `-g` 选择所使用的GPU序号，若为-1则使用CPU
    - `-p` caffe模型的定义文件 `.prototxt`
    - `-m` caffe模型的参数文件 `.caffemodel`
    - `-i` 测试集的图片及label列表文件
    - `-o` 指定输出的npz文件的目录
    - `--scale` 输入图片的scale，需与训练时保持一致 默认为1
    - `--mean` 一个三位的数列，图片BGR通道
    - `--batch_size` 运行时的batch size 默认为24
    - `--result_layer` 模型输出的结果层名，需与输入的测试集中label顺序保持一致

1. 第二步 `eval_multi_score.py` 脚本会读取第一步得到的npz文件，计算不同阈值下的precision，recall，badcase等结果并绘制pr曲线。最终会在console中输出recall为0.8和0.9时各个label的precison值，可选的参数为：

    - `--attrnamefile` 一个各个label对应的语义名称的文件，每行为一个label名，排列顺序同测试集中label的顺序