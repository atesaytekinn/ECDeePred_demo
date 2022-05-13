import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Libraries are loading....")

import tensorflow as tf
import numpy as np
import pickle
from sklearn.svm import SVC


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

subclasses_for_class1 = [1.1 , 1.11, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.2 , 1.3 , 1.4 ,
       1.5 , 1.6 , 1.7 , 1.8 , 1.9 , 1.97]
subclasses_for_class2 = [2.1, 2.3, 2.5, 2.6, 2.7, 2.8]
subclasses_for_class3 = [3.1, 3.2, 3.4, 3.5, 3.6]
subclasses_for_class4 = [4.1, 4.2, 4.3, 4.4, 4.6]
subclasses_for_class5 = [5.1, 5.2, 5.3, 5.4, 5.5, 5.6]
subclasses_for_class6 = [6.1, 6.2, 6.3, 6.4, 6.5]
subclasses_for_class7 = [7.1, 7.2, 7.4, 7.6]

subclasses_for_class7_1 = [1, 2]
subclasses_for_class7_2 = [1, 2]
subclasses_for_class6_3 = [1, 2, 4, 5]
subclasses_for_class5_1 = [1, 3, 99]
subclasses_for_class5_3 = [1, 2, 3, 4, 99]
subclasses_for_class5_4 = [2, 3, 99]
subclasses_for_class5_6 = [1, 2]
subclasses_for_class4_1 = [1, 2, 3, 99]
subclasses_for_class4_2 = [1, 2, 3]
subclasses_for_class4_3 = [1, 2, 3]
subclasses_for_class3_1 = [1, 13, 2, 21, 26, 27, 3, 4, 6]
subclasses_for_class3_2 = [1, 2]
subclasses_for_class3_4 = [11, 14, 16, 17, 19, 21, 22, 23, 24]
subclasses_for_class3_5 = [1, 2, 3, 4, 99]
subclasses_for_class3_6 = [1, 4, 5]
subclasses_for_class2_1 = [1, 2, 3]
subclasses_for_class2_3 = [1, 2, 3]
subclasses_for_class2_7 = [1, 10, 11, 12, 13, 2, 4, 6, 7, 8]
subclasses_for_class2_8 = [1, 2, 3, 4]
subclasses_for_class1_1 = [1, 3, 5, 99]
subclasses_for_class1_2 = [1, 3, 4, 7]
subclasses_for_class1_3 = [1, 3, 5, 7, 8, 98, 99]
subclasses_for_class1_4 = [1, 3, 4, 7, 99]
subclasses_for_class1_5 = [1, 3, 5, 8, 99]
subclasses_for_class1_6 = [2, 3, 5, 99]
subclasses_for_class1_7 = [1, 2, 3, 5]
subclasses_for_class1_8 = [1, 3, 4, 5, 7, 98, 99]
subclasses_for_class1_11 = [1, 2]
subclasses_for_class1_13 = [11, 12]
subclasses_for_class1_14 = [11, 12, 13, 14, 15, 18, 19, 99]
subclasses_for_class1_16 = [1, 3]
subclasses_for_class1_17 = [1, 4, 7, 99]
subclasses_for_class1_18 = [1, 6]

model = tf.keras.models.load_model("enzym_nonenzyme_classifier_2layers.h5")
mult_model = tf.keras.models.load_model("multiclass_classifier_2layers.h5")

class1_clf = pickle.load(open("level2/class1_classifier.sav", 'rb'))
class2_clf = pickle.load(open("level2/class2_classifier.sav", 'rb'))
class3_clf = pickle.load(open("level2/class3_classifier.sav", 'rb'))
class4_clf = pickle.load(open("level2/class4_classifier.sav", 'rb'))
class5_clf = pickle.load(open("level2/class5_classifier.sav", 'rb'))
class6_clf = pickle.load(open("level2/class6_classifier.sav", 'rb'))
class7_clf = pickle.load(open("level2/class7_classifier.sav", 'rb'))

class7_1_clf = pickle.load(open("level3/class7_1_classifier.sav", 'rb'))
class7_2_clf = pickle.load(open("level3/class7_2_classifier.sav", 'rb'))
class6_3_clf = pickle.load(open("level3/class6_3_classifier.sav", 'rb'))
class5_1_clf = pickle.load(open("level3/class5_1_classifier.sav", 'rb'))
class5_3_clf = pickle.load(open("level3/class5_3_classifier.sav", 'rb'))
class5_4_clf = pickle.load(open("level3/class5_4_classifier.sav", 'rb'))
class5_6_clf = pickle.load(open("level3/class5_6_classifier.sav", 'rb'))
class4_1_clf = pickle.load(open("level3/class4_1_classifier.sav", 'rb'))
class4_2_clf = pickle.load(open("level3/class4_2_classifier.sav", 'rb'))
class4_3_clf = pickle.load(open("level3/class4_3_classifier.sav", 'rb'))
class3_1_clf = pickle.load(open("level3/class3_1_classifier.sav", 'rb'))
class3_2_clf = pickle.load(open("level3/class3_2_classifier.sav", 'rb'))
class3_4_clf = pickle.load(open("level3/class3_4_classifier.sav", 'rb'))
class3_5_clf = pickle.load(open("level3/class3_5_classifier.sav", 'rb'))
class3_6_clf = pickle.load(open("level3/class3_6_classifier.sav", 'rb'))
class2_1_clf = pickle.load(open("level3/class2_1_classifier.sav", 'rb'))
class2_3_clf = pickle.load(open("level3/class2_3_classifier.sav", 'rb'))
class2_7_clf = pickle.load(open("level3/class2_7_classifier.sav", 'rb'))
class2_8_clf = pickle.load(open("level3/class2_8_classifier.sav", 'rb'))
class1_1_clf = pickle.load(open("level3/class1_1_classifier.sav", 'rb'))
class1_2_clf = pickle.load(open("level3/class1_2_classifier.sav", 'rb'))
class1_3_clf = pickle.load(open("level3/class1_3_classifier.sav", 'rb'))
class1_4_clf = pickle.load(open("level3/class1_4_classifier.sav", 'rb'))
class1_5_clf = pickle.load(open("level3/class1_5_classifier.sav", 'rb'))
class1_6_clf = pickle.load(open("level3/class1_6_classifier.sav", 'rb'))
class1_7_clf = pickle.load(open("level3/class1_7_classifier.sav", 'rb'))
class1_8_clf = pickle.load(open("level3/class1_8_classifier.sav", 'rb'))
class1_11_clf = pickle.load(open("level3/class1_11_classifier.sav", 'rb'))
class1_13_clf = pickle.load(open("level3/class1_13_classifier.sav", 'rb'))
class1_14_clf = pickle.load(open("level3/class1_14_classifier.sav", 'rb'))
class1_16_clf = pickle.load(open("level3/class1_16_classifier.sav", 'rb'))
class1_17_clf = pickle.load(open("level3/class1_17_classifier.sav", 'rb'))
class1_18_clf = pickle.load(open("level3/class1_18_classifier.sav", 'rb'))

print("Enter Q to quit!")

while True:

    print("Please enter your sequence:")
    seq = input()

    if seq == "Q":
        break

    prediction_file = open("sequence.txt", "w+")
    prediction_file.write(">sequence\n")
    prediction_file.write(seq)
    prediction_file.close()

    os.system("python iFeature-master/iFeature.py --file sequence.txt --type AAC --out features.txt >> temp.txt")

    x = ""
    while True:
        if os.path.exists("features.txt"):
            x = open("features.txt")
            x = x.read()

            os.remove("features.txt")
            os.remove("temp.txt")
            break
        else:
            continue

    x = x.split("\n")
    x.pop(0)
    x = x[0].split("\t")

    x_new = []
    for i in range(len(x)):
        if i != 0:
            x_new.append(float(x[i]))

    x_new = tf.constant(x_new, shape=(1,20), dtype='float32')

    prediction = model.predict(x_new)[0,0]

    if prediction >= 0.5:
        scores = mult_model.predict(x_new)
        prediction2 = np.argmax(scores) + 1 # np.argmax(mult_model.predict(x_new)) + 1
        prediction3 = np.max(scores)

        prediction4 = 0 # level 2 clasiffication
        prediction5 = 0 # level 3 classification

        if prediction2 == 1:
            prediction4 = subclasses_for_class1[class1_clf.predict(x_new)[0]]
            if prediction4 == 1.1:
                prediction5 = subclasses_for_class1_1[class1_1_clf.predict(x_new)[0]]
            elif prediction4 == 1.11:
                prediction5 = subclasses_for_class1_11[class1_11_clf.predict(x_new)[0]]
            elif prediction4 == 1.13:
                prediction5 = subclasses_for_class1_13[class1_13_clf.predict(x_new)[0]]
            elif prediction4 == 1.14:
                prediction5 = subclasses_for_class1_14[class1_14_clf.predict(x_new)[0]]
            elif prediction4 == 1.15:
                prediction5 = 1
            elif prediction4 == 1.16:
                prediction5 = subclasses_for_class1_16[class1_16_clf.predict(x_new)[0]]
            elif prediction4 == 1.17:
                prediction5 = subclasses_for_class1_17[class1_17_clf.predict(x_new)[0]]
            elif prediction4 == 1.18:
                prediction5 = subclasses_for_class1_18[class1_18_clf.predict(x_new)[0]]
            elif prediction4 == 1.2:
                prediction5 = subclasses_for_class1_2[class1_2_clf.predict(x_new)[0]]
            elif prediction4 == 1.3:
                prediction5 = subclasses_for_class1_3[class1_3_clf.predict(x_new)[0]]
            elif prediction4 == 1.4:
                prediction5 = subclasses_for_class1_4[class1_4_clf.predict(x_new)[0]]
            elif prediction4 == 1.5:
                prediction5 = subclasses_for_class1_5[class1_5_clf.predict(x_new)[0]]
            elif prediction4 == 1.6:
                prediction5 = subclasses_for_class1_6[class1_6_clf.predict(x_new)[0]]
            elif prediction4 == 1.7:
                prediction5 = subclasses_for_class1_7[class1_7_clf.predict(x_new)[0]]
            elif prediction4 == 1.8:
                prediction5 = subclasses_for_class1_8[class1_8_clf.predict(x_new)[0]]
            elif prediction4 == 1.9:
                prediction5 == 6
            else: # prediction4 == 1.97:
                prediction5 == 1
        
        elif prediction2 == 2:
            prediction4 = subclasses_for_class2[class2_clf.predict(x_new)[0]]
            if prediction4 == 2.1:
                prediction5 = subclasses_for_class2_1[class2_1_clf.predict(x_new)[0]]
            elif prediction4 == 2.3:
                prediction5 = subclasses_for_class2_3[class2_3_clf.predict(x_new)[0]]
            elif prediction4 == 2.5:
                prediction5 = 1
            elif prediction4 == 2.6:
                prediction5 = 1
            elif prediction4 == 2.7:
                prediction5 = subclasses_for_class2_7[class2_7_clf.predict(x_new)[0]]
            else: # prediction4 == 2.8
                prediction5 = subclasses_for_class2_8[class2_8_clf.predict(x_new)[0]]
        
        elif prediction2 == 3:
            prediction4 = subclasses_for_class3[class3_clf.predict(x_new)[0]]
            if prediction4 == 3.1:
                prediction5 = subclasses_for_class3_1[class3_1_clf.predict(x_new)[0]]
            elif prediction4 == 3.2:
                prediction5 = subclasses_for_class3_2[class3_2_clf.predict(x_new)[0]]
            elif prediction4 == 3.4:
                prediction5 = subclasses_for_class3_4[class3_4_clf.predict(x_new)[0]]
            elif prediction4 == 3.5:
                prediction5 = subclasses_for_class3_5[class3_5_clf.predict(x_new)[0]]
            else : # prediction4 == 3.6
                prediction5 = subclasses_for_class3_6[class3_6_clf.predict(x_new)[0]]
        
        elif prediction2 == 4:
            prediction4 = subclasses_for_class4[class4_clf.predict(x_new)[0]]
            if prediction4 == 4.1:
                prediction5 = subclasses_for_class4_1[class4_1_clf.predict(x_new)[0]]
            elif prediction4 == 4.2:
                prediction5 = subclasses_for_class4_2[class4_2_clf.predict(x_new)[0]]
            elif prediction4 == 4.3:
                prediction5 = subclasses_for_class4_3[class4_3_clf.predict(x_new)[0]]
            elif prediction4 == 4.4:
                prediction5 = 1
            else : # prediction4 == 4.6
                prediction5 = 1
        
        elif prediction2 == 5:
            prediction4 = subclasses_for_class5[class5_clf.predict(x_new)[0]]
            if prediction4 == 5.1:
                prediction5 = subclasses_for_class5_1[class5_1_clf.predict(x_new)[0]]
            elif prediction4 == 5.2:
                prediction5 = 1
            elif prediction4 == 5.3:
                prediction5 = subclasses_for_class5_3[class5_3_clf.predict(x_new)[0]]
            elif prediction4 == 5.4:
                prediction5 = subclasses_for_class5_4[class5_4_clf.predict(x_new)[0]]
            elif prediction4 == 5.5:
                prediction5 = 1
            else: # prediction4 == 5.6:
                prediction5 = subclasses_for_class5_6[class5_6_clf.predict(x_new)[0]]
        
        elif prediction2 == 6:
            prediction4 = subclasses_for_class6[class6_clf.predict(x_new)[0]]
            if prediction4 == 6.1:
                prediction5 = 1
            elif prediction4 == 6.2:
                prediction5 = 1
            elif prediction4 == 6.3:
                prediction5 = subclasses_for_class6_3[class6_3_clf.predict(x_new)[0]]
            elif prediction4 == 6.4:
                prediction5 = 1
            else: # prediction4 == 6.5
                prediction5 = 1
        
        else : # prediction2 == 7:
            prediction4 = subclasses_for_class7[class7_clf.predict(x_new)[0]]
            if prediction4 == 7.1:
                prediction5 = subclasses_for_class7_1[class7_1_clf.predict(x_new)[0]]
            elif prediction4 == 7.2:
                prediction5 = subclasses_for_class7_2[class7_2_clf.predict(x_new)[0]]
            elif prediction4 == 7.4:
                prediction5 = 2
            else: # prediction4 == 7.6
                prediction5 = 1

        print(seq + " is enzyme with EC Number " + str(prediction4) + "." + str(prediction5))

    else :
        print(seq + " is non-eznyme")

    os.remove("sequence.txt")
    print("\n")