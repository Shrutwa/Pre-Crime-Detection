import csv
import nltk
import re
from langdetect import detect
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from collections import defaultdict
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


train=[]
test=[]
data=[]
action_data = []
with open("action_data.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for lines in csv_reader:
        tokens = word_tokenize(lines[0])
        #print(tokens)
        for token, tag in pos_tag(tokens):
            lemma = wordnet_lemmatizer.lemmatize(token, tag_map[tag[0]])
            #print(token, "=>", lemma)
        action_data.append(lemma)
#print("Lemmatized action dataset:",action_data)

with open("action_data"+"_refined"+".csv", "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for line in action_data:
        writer.writerow([line])

pair_dict=dict()
with open("action_pairing.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for lines in csv_reader:
        if lines[0] not in pair_dict.keys():
            pair_dict[lines[0]]=[]
            q=lines[1].split(",")
            for a in q:
                pair_dict[lines[0]].append(a)
#print("action context sensitive pair dictionery",pair_dict)


#train  dataset phase 1
with open("datasetNLP.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i=0
    for lines in csv_reader:
        i += 1
        data.append((lines[0], lines[1]))

#print("training dataset",data)
#train = data[0:int(len(data)*0.8)]
#test = data[int(len(data)*0.8):len(data)]
train=data
test=data
#print(train)
#print(test)
#print("lendat",len(data))
#print("lentrain",len(train))
#print("lentest",len(test))

cl = NaiveBayesClassifier(train)
data1 = []
q = 0
l = 0
count=0
countarr=[]
action_verb_arr=[]

predict_data = []
print("Enter Dataset Name:")
Predict_dataset=input()
with open(str(Predict_dataset)+".csv", "r", encoding = "latin1") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i=0
    for lines in csv_reader:
        i += 1
        e = lines[0].encode("ascii", errors = "ignore")
        d = e.decode("ascii")
        lines[0]=d
        predict_data.append(d)

print(predict_data)

with open(str(Predict_dataset)+"_refined"+".csv", "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for line in predict_data:
        writer.writerow([line])

print("Does the dataset have other language words:(Y/N)")
input_lang=input()
with open(str(Predict_dataset)+"_refined.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for lines in csv_reader:
        l += 1
        '''
        if input_lang == "Y":
            if len(lines[0])>3:
                lang = TextBlob(lines[0])
                if lang.detect_language() != "en":
                    q+=1
                    continue
        '''
        if input_lang == "Y":
            try:
                if len(lines[0]) > 3:
                    if detect(lines[0])!= "en":
                        q += 1
                        continue
            except:
                q += 1
                continue

        #Action - context sensing phase
        tokens = nltk.word_tokenize(lines[0])
        tags = nltk.pos_tag(tokens)
        #print("Parts of speech:", tags)
        extracted_action_arr = []
        for a in tags:
            if a[1] == "VB" or a[1] == "VBD" or a[1] == "VBG" or a[1] == "VBN" or a[1] == "VBP" or a[1] == "VBZ":
                #print(a)
                extracted_action_arr.append(a[0])
        #print(extracted_action_arr, "\n")

        for i in range(len(extracted_action_arr)):
            tokens = word_tokenize(extracted_action_arr[i])
            #print(tokens)
            for token, tag in pos_tag(tokens):
                lemma = wordnet_lemmatizer.lemmatize(token, tag_map[tag[0]])
                #print(token, "=>", lemma)
            extracted_action_arr[i]=lemma
        #print(extracted_action_arr)

        listtemp = []
        if len(extracted_action_arr) !=0:
            for verb in extracted_action_arr:
                if verb in action_data and verb not in listtemp:
                    listtemp.append(verb)
            if len(listtemp) != 0:
                action_verb_arr.append(listtemp)

        #Finding probable continuity of doubtful statements
        label_identified = cl.classify(lines[0])
        #print(lines[0], label_identified, "\n")
        if label_identified == "pos" and l != 1:
            if count > 0:
                countarr.append(count)
            count = 0
        if label_identified == "neg":
            q += 1
            count += 1
if count != 0:
    countarr.append(count)
universal_answer_array=[]
#phase 1 output
final_ratio_phase1=q/l
print("un doubtful statements total ratio after detection from datset -phase1:", q/l)
if final_ratio_phase1*100 > 100-(final_ratio_phase1*100):
    universal_answer_array.append(["phase1",final_ratio_phase1*100,"un-doubtful"])
else:
    universal_answer_array.append(["phase1",100-(final_ratio_phase1 * 100), "doubtful"])
#phase 2 evaluation : post processing
print(countarr)
f=str(l)
r=len(f)
e=r-1
g=int(f[0])+1
u=2/5
if(int(l)<=10):
    method1_int=(int(l)/2)-2
elif(int(l)<=20 and int(l)>10):
    method1_int=(int(l)/2)*0.6
elif (int(l) <= 50 and int(l) > 20):
    method1_int = (int(l) / 2) * 0.4
elif(r<=2):
    method1_int=l/g
elif(r>=3):
    method1_int = l / g * (u**e)
flag = 1
print(method1_int)
int_final_count = 0
for a in countarr:
    if a > method1_int:
#        print("after phase2: un-doubtful")
        flag = 0
        int_final_count += 1
#if flag == 1:
#    print("after phase2: doubtful set")
final_ratio_phase2=int_final_count/len(countarr)
#print(final_ratio)
if final_ratio_phase2*100 > 30:
    print("after phase2: un-doubtful set")
else:
    print("after phase2: doubtful set")
if final_ratio_phase2*100 > 100-(final_ratio_phase2*100):
    universal_answer_array.append(["phase2", final_ratio_phase2*100,"un-doubtful"])
else:
    universal_answer_array.append(["phase2", 100-(final_ratio_phase2 * 100), "doubtful"])
#phase 3 evaluation : post processing
print("Extracted and filtered Action verbs from action_data:",action_verb_arr)
dump_list=[]
i=0
j=0
k=0
t=0
flag=1
final_list_pairs=[]
final_list_pairs_statement_index=[]
for i in range(len(action_verb_arr)):
    for j in range(len(action_verb_arr[i])):
        if action_verb_arr[i][j] in pair_dict.keys():
            for k in range(i+1, len(action_verb_arr)):
                for t in range(len(action_verb_arr[k])):
                    #print(i,j,k,t)
                    if action_verb_arr[k][t] in pair_dict[action_verb_arr[i][j]]:
                        l=[action_verb_arr[i][j],action_verb_arr[k][t]]
                        r=[i,k]
                        del action_verb_arr[i][j]
                        del action_verb_arr[k][t]
                        final_list_pairs.append(l)
                        final_list_pairs_statement_index.append(r)
                        #print(final_list_pairs)
                        #print(final_list_pairs_statement_index)
                        flag=0
                        break
                if flag==0:
                    break
        if flag ==0 :
            flag=1
            break
print("All Pairs:",final_list_pairs)
print("All Pairs with statement id:",final_list_pairs_statement_index)
count_context_sensitive=len(final_list_pairs_statement_index)
print("Total pairs detected",count_context_sensitive)
l=len(action_verb_arr)
f=str(l)
r=len(f)
e=r-1
g=int(f[0])+1
u=2/5
#print(l,f,r,e,g,u)
if(int(l)<=10):
    method1_int=(int(l)/2)-2
elif(int(l)<=20 and int(l)>10):
    method1_int=(int(l)/2)*0.6
elif (int(l) <= 50 and int(l) > 20):
    method1_int = (int(l) / 2) * 0.4
elif(r<=2):
    method1_int=l/g
elif(r>=3):
    method1_int = l / g * (u**e)
#print(count_context_sensitive,method1_int)
if count_context_sensitive > method1_int:
    universal_answer_array.append(["phase3","doubtful"])
else:
    universal_answer_array.append(["phase3","un-doubtful"])

#print("final_ratio_phase1",final_ratio_phase1)
#print("final_ratio_phase2",final_ratio_phase2)
#print("final_ratio_phase3",final_ratio_phase3)
# Classify some text
print(universal_answer_array)
#print("The Bird is in the nest -", cl.classify("The Bird is in the nest."))  # "pos"
#print("The parcel has been sent -", cl.classify("The parcel has been sent"))  # "pos"
#print("The apple is in the pie -", cl.classify("The apple is in the pie"))  # "pos"
print("Accuracy: {0}".format(cl.accuracy(test)*100))

#print("checking extra featurette..")

