cat 1030.txt | python pinjie.py > 1030_part2.txt
paste -d',' ./subject_tfidf.csv 1030_part2.txt > 1030_part3.txt
awk -F',' '{print $1","$2","$5-1","$4}' 1030_part3.txt > 1030_part4.txt
