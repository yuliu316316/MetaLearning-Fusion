%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This package contains the source code which is associated with the following paper:

Huafeng Li, Yueliang Cen, Yu Liu, Xun Chen, Zhengtao Yu, "Different Input Resolutions and Arbitrary Output Resolution: A Meta-Learning Based Deep Framework for Infrared and Visible Image Fusion", IEEE Transactions on Image Processing, vol. 30, pp. 4070-4083, 2021.

Edited by Yueliang Cen and Yu Liu.   

Usage of this code is free for research purposes only. 

Please refer to the above publication if you use this code. Thank you.

Thank you.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Requirements:
    CUDA  10
    Python  3.7
    Pytorch  1.3.1
    torchvision  0.4
    numpy  1.17
    cv2  4.1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Install and run demo:
    1. download the code
        git clone https://github.com/yuliu316316/MetaLearning-Fusion.git
        cd MetaLearning-Fusion

    2. run the demo file: 
        python demo.py
        
    3. Train:
        3.1. Train the whole network:
            python main.py

        3.2. And then fine-turn the fusion branch with the contrast loss:
            python main.py --finetune_AAF

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Contact:

    Don't hesitate to contact me if you meet any problems when using this code.

    Yu Liu
    Department of Biomedical Engineering
    Hefei University of Technology                                                            
    Email: yuliu@hfut.edu.cn; lyuxxz@163.com
    Homepage: https://sites.google.com/site/yuliu316316; https://github.com/yuliu316316


Last update: 27-July-2022
