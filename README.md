# 智能单人cut
本工具可以根据提供的素材，学习并识别影视节目中的某位角色，并且根据识别的结果，对提供的视频进行单人CUT

基础工具:

1. ffmpeg(需要配置到环境变量)
2. Python 2.7-3.7

Python库:

1. dlib(gpu版本可能需要自行编译)
2. face_recognition
3. opencv-python

使用方法:

1. 将目标嘉宾的照片放到images文件夹
2. 运行 python main.py generate_data_set
3. 在data文件夹人工筛选不合格的照片，保留大约2000张(作者并没有实验过其他数值)
4. 运行 python main.py generate_face_encodings
5. 在videos文件夹放入目标视频
6. 调整threshold_recognition使得cut结果令人满意
