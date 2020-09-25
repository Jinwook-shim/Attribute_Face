import tensorflow as tf
import numpy as np
import cv2
import os
import tqdm 

ATTR='child'
MODEL_PATH='checkpoints'
RESULT_DIR=ATTR+'_results'
GEN_NUMS=100

class generate_with_attribute:

    def __init__(self,sess,attr,checkpoints,save_dir,nums):
        self.sess=sess
        self.attr=attr
        self.checkpoints=checkpoints
        self.save_dir=save_dir
        self.nums=nums
        self._extract_model()
        self.load_model()

    def _extract_model(self):
        self.extract_model=os.path.join(self.checkpoints,self.attr)
        self._check_dir()
        
    def _check_dir(self):
        if not os.path.isdir(self.extract_model):
            print(' [!] Parameter: checkpoints or attr is erro.')
            exit()
        os.makedirs(self.save_dir,exist_ok=True)
    
    def load_model(self):
        self.meta_path=os.path.join(self.extract_model,self.attr+'_model.meta')
        self.saver = tf.train.import_meta_graph(self.meta_path,clear_devices=True)
        self.saver.restore(self.sess,tf.train.latest_checkpoint(self.extract_model))
        print(' [*] Checkpoints %s has loaded successfully.'%self.attr)
        #output_node_names =[n for n in tf.get_default_graph().as_graph_def().node]
        self.graph=tf.get_default_graph()
        self.input=self.graph.get_tensor_by_name('input_5:0')
        try:
            self.image=self.graph.get_tensor_by_name('concat_5/concat:0')
        except:
            self.image=self.graph.get_tensor_by_name('concat_5:0')
    
    def generate(self):
        print(' [*] Ready to generate.')
        for i in tqdm.tqdm(range(self.nums)):
            random_latent=np.random.randn(1, 512)
            img=self.sess.run(self.image,feed_dict={self.input:random_latent})
            img=self.back_process(img)
            savename=os.path.join(self.save_dir,'%s_%05d.jpg'%(self.attr,i))
            cv2.imwrite(savename,img)
        print(' [*] Generate task has finished.')

    def back_process(self,img):
        drange_min, drange_max = -1.0, 1.0
        scale = 255.0 / (drange_max - drange_min)
        scaled_image = img * scale + (0.5 - drange_min * scale)
        scaled_image = np.clip(scaled_image, 0, 255)
        img = cv2.cvtColor(scaled_image[0].astype('uint8'), cv2.COLOR_RGB2BGR)
        return img

if __name__=='__main__':

    with tf.Session() as sess:
        #sess,attr,checkpoints,save_dir,nums
        model=generate_with_attribute(sess,ATTR,MODEL_PATH,RESULT_DIR,GEN_NUMS)
        model.generate()