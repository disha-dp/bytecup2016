import math
import pickle

def get_prediction(w_a_u, a_user, user_counts_per_category, category_counts_per_user):

    active_category_count = user_counts_per_category[a_user]
    ra_bar = sum(active_category_count.values()) / len(active_category_count)
    p_a = {}
    # print 'category count for active user: '
    # print active_category_count
    # raw_input()
    for c in active_category_count:
        nr_sum = 0
        dr_sum = 0
        for u in w_a_u:
            user_category_count = user_counts_per_category[u]
            # print 'other user similar to active user: u:',u
            # print 'he answered this way for various categories: ',user_category_count
            if c in user_category_count:
                user_count = user_category_count[c]
            else:
                user_count = 0
            ru_bar = sum(user_category_count.values()) / len(user_category_count)
            nr_sum += abs(user_count-ru_bar)*w_a_u[u]
            dr_sum += w_a_u[u]
        if dr_sum!=0:
            p_a[c] = ra_bar+(float(nr_sum)/dr_sum)
        else:
            p_a[c] = 0
    # if len(p_a)!=0:
    mean = sum(p_a.values())
    mean_pred={}
    if mean!=0:
        for c in p_a:
            mean_pred[c]=float(p_a[c])/mean
    else:
        for c in p_a:
            mean_pred[c] = 0
    # print mean_pred
    # raw_input()
    return mean_pred

def find_similarity(active_user,user_counts_per_category,category_counts_per_user):
    active_category_count = user_counts_per_category[active_user]
    w_a_u = {}
    for u in user_counts_per_category:
        category_count = user_counts_per_category[u]
        user_category_count = user_counts_per_category[u]
        a_set = set(active_category_count)
        u_set = set(category_count)
        common_categories  = a_set.intersection(u_set)
        ra_bar = sum(active_category_count.values())/len(active_category_count)
        ru_bar = sum(user_category_count.values())/len(user_category_count)
        dr_first = 0
        dr_second = 0
        nr_sum = 0
        for c in common_categories:
            user_count_dictionary = category_counts_per_user[c]
            r_a_count = user_count_dictionary[active_user]
            r_u_count = user_count_dictionary[u]
            nr_sum += (r_a_count-ra_bar - r_u_count)*(r_u_count-ru_bar)
            dr_first += abs(r_a_count-ra_bar)**2
            dr_second += abs(r_u_count-ru_bar)**2
        denominator = math.sqrt(abs(dr_first*dr_second))
        if denominator!=0:
            w_a_u[u] = float (nr_sum)/denominator
        else:
            w_a_u[u] = 0

    return dict((k, v) for k, v in w_a_u.items() if v > 0)

def main():
    question_info_doc = './bytecup2016data/question_info.txt'
    invited_info_train = './bytecup2016data/invited_info_train.txt'
    doc = open(question_info_doc,'r')
    content = doc.read()
    doc.close()
    content_lines_list = content.strip().split('\n')
    question_category_mapping = {}
    for line1 in content_lines_list:
        q_id,category = line1.strip().split('\t')[:2]
        question_category_mapping[q_id] = category

    doc = open(invited_info_train,'r')
    content = doc.read()
    doc.close()
    content_lines_list = content.strip().split('\n')

    category_counts_per_user = {}
    user_counts_per_category = {}

    lines = map(lambda line: line.strip().split('\t'),content_lines_list)

    answered_quest_list = filter( lambda row: row[2] == '1', lines)
    for line1 in answered_quest_list:
        q_id = line1[0]
        u_id = line1[1]
        response = line1[2]

        category = question_category_mapping[q_id]

        if category not in category_counts_per_user:
            category_counts_per_user[category] = {}
            category_counts_per_user[category][u_id] = 0
        else:
            if u_id not in category_counts_per_user[category]:
                category_counts_per_user[category][u_id] = 0
        category_counts_per_user[category][u_id] += 1

        if u_id not in user_counts_per_category:
            user_counts_per_category[u_id] = {}
            user_counts_per_category[u_id][category] = 0
        else:
            if category not in user_counts_per_category[u_id]:
                user_counts_per_category[u_id][category] = 0
        user_counts_per_category[u_id][category] += 1

    '''
    executed the commented portion of code below, to get a pickle dump of predictions,
    then generated the final file using this dump
    '''

    '''
    w_a_u = {}
    prediction = {}
    for a_user in user_counts_per_category:
        print '.',
        w_a_u[a_user] = find_similarity(a_user,user_counts_per_category,category_counts_per_user)
        if w_a_u is not None:
            prediction[a_user] = get_prediction(w_a_u[a_user],a_user, user_counts_per_category, category_counts_per_user)

    print 'abcdefg'

    modelfile = open('results.txt', 'wb')
    pickle.dump(prediction, modelfile)
    modelfile.close()


    print prediction
    '''
    modelfile = open('results.txt', 'rb')
    training_acc_test = open('./bytecup2016data/training_res.txt','wb')
    prediction = pickle.load(modelfile)
    validation_file = open('./bytecup2016data/validate_nolabel.txt','rb')
    final_results = open('./bytecup2016data/final_results.txt','wb')
    validation_content=validation_file.read()
    vcontent_lines_list = validation_content.strip().split('\n')

    doc = open(invited_info_train, 'r')
    content = doc.read()
    doc.close()
    content_lines_list = content.strip().split('\n')

    for line2 in content_lines_list: #vcontent_lines_list,
        print line1
        q_id, u_id = line1.strip().split(',')
        category = question_category_mapping[q_id]
        if u_id in prediction:
            category_pred_list=prediction[u_id]
            if category in category_pred_list:
                pred = category_pred_list[category]
            else:
                pred = 0
        else:
            pred = 0
        final_results.write(q_id+','+u_id+','+str(pred)+'\n')

        #for training acc test
        q_id, u_id, label = line2.strip().split('\t')
        category = question_category_mapping[q_id]
        if u_id in prediction:
            category_pred_list = prediction[u_id]
            if category in category_pred_list:
                pred = category_pred_list[category]
            else:
                pred = 0
        else:
            pred = 0
        training_acc_test.write(q_id + ',' + u_id + ',' +label + ',' + str(pred) + '\n')




if __name__=='__main__':
        main()