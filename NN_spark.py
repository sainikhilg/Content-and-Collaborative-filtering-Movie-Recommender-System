import sys
from collections import defaultdict
from itertools import combinations
import numpy as np
from pyspark import SparkContext

def sampleInteractions(item_id,users_with_rating):

     
    return item_id, users_with_rating

def findUserPairs(item_id,users_with_rating):
    
    for user1,user2 in combinations(users_with_rating,2):
        return (user1[0],user2[0]),(user1[1],user2[1])

def calcSim(user_pair,rating_pairs):
    
    sum_xx, sum_xy, sum_yy, sum_x, sum_y, n = (0.0, 0.0, 0.0, 0.0, 0.0, 0)
    
    for rating_pair in rating_pairs:
        sum_xx += np.float(rating_pair[0]) * np.float(rating_pair[0])
        sum_yy += np.float(rating_pair[1]) * np.float(rating_pair[1])
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])
        # sum_y += rt[1]
        # sum_x += rt[0]
        n += 1

    cos_sim = cosine(sum_xy,np.sqrt(sum_xx),np.sqrt(sum_yy))
    return user_pair, (cos_sim,n)

def cosine(dot_product,rating_norm_squared,rating2_norm_squared):
    
    numerator = dot_product
    denominator = rating_norm_squared * rating2_norm_squared

    return (numerator / (float(denominator))) if denominator else 0.0

def keyOnFirstUser(user_pair,item_sim_data):
    
    (user1_id,user2_id) = user_pair
    return user1_id,(user2_id,item_sim_data)

def nearestNeighbors(user,users_and_sims,n):
    
    users_and_sims.sort(key=lambda x: x[1][0],reverse=True)
    return user, users_and_sims[:n]

def topNRecommendations(user_id,user_sims,users_with_rating,n):
    

    # initialize dicts to store the score of each individual item,
    # since an item can exist in more than one item neighborhood
    totals = defaultdict(int)
    sim_sums = defaultdict(int)

    for (neighbor,(sim,count)) in user_sims:

        # lookup the item predictions for this neighbor
        unscored_items = users_with_rating.get(neighbor,None)

        if unscored_items:
            for (item,rating) in unscored_items:
                if neighbor != item:

                    # update totals and sim_sums with the rating data
                    totals[neighbor] += sim * rating
                    sim_sums[neighbor] += sim

    # create the normalized list of scored items 
    scored_items = [(total/sim_sums[item],item) for item,total in totals.items()]

    # sort the scored items in ascending order
    scored_items.sort(reverse=True)

    # take out the item score
    ranked_items = [x[1] for x in scored_items]

    return user_id,ranked_items[:n]

if __name__ == "__main__":
    #if len(sys.argv) < 3:
     #   print >> sys.stderr, \
      #      "Usage: PythonUserCF <master> <file>"
        #exit(-1)

  sc = SparkContext(appName="PythonUserItemCF")
  lines = sc.textFile(sys.argv[2])
  
  complete_ratings_raw_data = sc.textFile(sys.argv[1])
  complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]
   # Parse
  complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header)\
      .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[1]),[int(tokens[0]),float(tokens[2])])).cache()
  
  movies_raw_data = sc.textFile(sys.argv[2])
  movies_raw_data_header = movies_raw_data.take(1)[0]
  movies_data = movies_raw_data.filter(lambda line: line!=movies_raw_data_header)\
        .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()
    
  movies_broadcast = sc.broadcast({
    k: v for (k, v) in movies_data.collect()
  })
    
    
  item_user_pairs = complete_ratings_data.groupByKey().map(
      lambda p: sampleInteractions(p[0],p[1])).cache()

    #item_user_pairs = lines.map(parseVectorOnItem).groupByKey().map(
     #   lambda p: sampleInteractions(p[0],p[1],500)).cache()

    
  pairwise_users = item_user_pairs.filter(
      lambda p: len(p[1]) > 1).map(
      lambda p: findUserPairs(p[0],p[1])).groupByKey()

    
  user_sims = pairwise_users.map(
        lambda p: calcSim(p[0],p[1])).map(
        lambda p: keyOnFirstUser(p[0],p[1])).groupByKey().map(
        lambda p: nearestNeighbors(p[0],list(p[1]),50))

    
  user_item_hist = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header)\
        .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),        [int(tokens[1]),float(tokens[2])])).groupByKey().collect()


    #user_item_hist = lines.map(parseVectorOnUser).groupByKey().collect()

  ui_dict = {}
  for (user,items) in user_item_hist: 
      ui_dict[user] = items

  uib = sc.broadcast(ui_dict)

    
  user_item_recs = user_sims.map(
        lambda p: topNRecommendations(p[0],p[1],uib.value,10)).collect()
    
  for i in list(range(0, 30)):
      a=[]
      for j in user_item_recs[i][1]:
          try:
              a.append(movies_broadcast.value[str(j)] )
          except:
              h=1
      print "User Id=",user_item_recs[i][0],"Movie List=",a,"\n"
