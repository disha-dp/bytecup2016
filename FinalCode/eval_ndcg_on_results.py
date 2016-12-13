'''
Evaluating ndcg score on the predicted results
Note: (i) ndcg.py is assumed to be in ./bytecup2016data and
(ii) change path to point to the csv file that has to be evaluated
'''
import sys
sys.path.insert(0, './bytecup2016data/')
import ndcg as ndcg
path = "final.csv"
try:
    path = sys.argv[1]
except:
    pass

finalcsv = pa.read_csv(path, sep=",",names=["qid","uid","label"] ) 
mydict = finalcsv.sort_values('label', ascending=False).groupby('qid')['uid'].apply(list).to_dict()
ndcg_5 = 0
ndcg_10 = 0
count = 0 
for key in mydict:
    user_list= mydict[key]
    ranking = []
    for user in user_list:
        value = training_data_formatted.loc[(training_data_formatted['qid'] == key) & (training_data_formatted['uid'] == user)]
        ranking.append(value['label'].values[0])
    ndcg_5 += ndcg.ndcg_at_k(ranking, 5)
    ndcg_10 += ndcg.ndcg_at_k(ranking, 10)
    count+=1
ndcg_5/=count
ndcg_10/=count
final_score = (ndcg_5+ndcg_10)*0.5
print final_score
