##  Specific Attribute Face

This is a tensorflow implementation of the specific attribute generate. We propose the principle of perspective consistency. Which first learn  the distribution of specific attributes from a dataset and regenerate the image of the target attribute from the distribution rule. We carried out experiments on 10 kinds of face attributes, and all of them achieved good results.



### Installation requirements

We use GPU and CPU respectively for image generation, both of which can work normally. All the dependence have been packaged to **requirments.txt** and you can use it for installation.

```shell
pip install -r requirments.txt
```

### Test

First, you need to download our model file [**link**](https://pan.baidu.com/s/1y7I2M1Ejkg8n0usIJzYs9w)  code: **rbfz**, and place it into **./checkpoints** currently include: child, adult, aged and glasses. Other model files will come soon.

update: female, male, asian, smile were added, and model file [**link**](https://pan.baidu.com/s/13I6qiKefb51p5yKz9PBWxg) code: **ir60**

Once the model is downloaded, you also need to modify the parameters in the **test.py** file:

```python
ATTR='child'               # which attribute will be generated
MODEL_PATH='checkpoints'   # the dir of model path
RESULT_DIR=ATTR+'_results' # save path
GEN_NUMS=100               # the number of images generated
```

Then, you just need to run **python test.py**  to generate an image of a face with the target property.

### Results

![](./assert/figure1.png)

### *TODO*

training code

face expression: sad, surprise, natural,angry

head position: left, middle, right

race:  black , white

### **Others**

We use the model to generate a variety of face attributes, which can be used for attribute classification, image generation and other related tasks. The details are as follows:

| attribute | num    | link           |
| --------- | ------ | -------------- |
| chid      | 10,000 | to be released |
| adult     | 10,000 | to be released |
| old       | 10,000 | to be released |
| glassed   | 10,000 | to be released |
| female    | 10,000 | to be released |
| male      | 10,000 | to be released |
| asian     | 10,000 | to be released |
| smile     | 10,000 | to be released |

### Note

There is a certain probability that the generated image becomes illusory. We believe that improving the quality of the training data set and increasing the number of iterations can greatly alleviate this situation, but we have no time to try.