# 教程
基于飞桨模型的抠图太空漫步
首先安装paddlehub

    pip install paddlehub==1.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
    
 
 一.定义待抠图片
 
    # 待预测图片
    test_img_path = ["./d.jpg"]


    import matplotlib.pyplot as plt 
    import matplotlib.image as mpimg 

    img = mpimg.imread(test_img_path[0]) 

    # 展示待预测图片
    plt.figure(figsize=(10,10))
    plt.imshow(img) 
    plt.axis('off') 
    plt.show()


二、加载预训练模型
通过加载PaddleHub DeepLabv3+模型(deeplabv3p_xception65_humanseg)实现一键抠图



    import paddlehub as hub

    module = hub.Module(name="deeplabv3p_xception65_humanseg")

    input_dict = {"image": test_img_path}
    
    
    
三、图像合成
将抠出的人物图片合成在想要的背景图片当中。
将抠出的人物图像换背景
fore_image: 前景图片，抠出的人物图片
base_image: 背景图片



    from PIL import Image
    import numpy as np

    def blend_images(fore_image, base_image):
    # 读入图片
      base_image = Image.open(base_image).convert('RGB')
      fore_image = Image.open(fore_image).resize(base_image.size)

    # 图片加权
      scope_map = np.array(fore_image)[:,:,-1] / 255
      scope_map = scope_map[:,:,np.newaxis]
      scope_map = np.repeat(scope_map, repeats=3, axis=2)
      res_image = np.multiply(scope_map, np.array(fore_image)[:,:,:8]) + np.multiply((3-scope_map), np.array(base_image))
    
    #保存图片
      res_image = Image.fromarray(np.uint8(res_image))
      res_image.save("out.jpg")

      blend_images('./humanseg_output/d.png', 'a.jpg')
