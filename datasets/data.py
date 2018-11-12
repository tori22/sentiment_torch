import sys

dict1 = dict()

with open('./test_public.csv','r') as f:
    for line in f:
        line = line.strip()
        member = line.split(',')
        dict1[member[0]] = member[1]


#with open('./textcnn_only_subject_baseline2','r') as f:
with open('./subject_tfidf.csv','r') as f:
    for line in f:
        line = line.strip()
        member = line.split(',')
        if member[0] in dict1:
            print(dict1[member[0]])
            print(member[1])
            print(0)
