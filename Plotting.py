'''--------------Plotting Section-------------'''

'''-------Bargraph Of Sentiment--------'''

import seaborn as sns
see = []

for twt in tweets.sentiment:
    see.append(twt)

ax = sns.distplot(see,kde=False,bins=3)
ax.set(xlabel = 'Negative            Neutral             Positive'
       ,ylabel = '#Tweets',title = 'Sentiment Score of COVID-19 Tweets')




'''-----------Emotions of the Tweets----------'''


from collections import Counter

def con(sentence):
    emotion_list = []
    sentence = sentence.split(' ')
    with open('emotions.txt','r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",",'').replace("'",'').strip()
            word, emotion = clear_line.split(':')

            if word in sentence:
                emotion_list.append(emotion)
        w = Counter(emotion_list)
        return w



tweets['emotion'] = tweets['clean_text'].apply(lambda x: con(x))
tweets.head()

emotions=con(tweets['clean_text'].sum())



'''--------Plotting Emotion Bargraph--------'''

plt.figure(figsize = (15,10))
plt.bar(emotions.keys(),emotions.values())
plt.xticks(rotation = 90)
plt.show()



'''---------WordCloud-----------'''


from PIL import Image
from wordcloud import WordCloud,ImageColorGenerator
import urllib
import requests
import matplotlib.pyplot as plt
def generate_wordcloud(all_words):
    Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
    image_colors = ImageColorGenerator(Mask)
    wc = WordCloud(background_color='black', height=750, width=2000,mask=Mask).generate(all_words)
    plt.figure(figsize=(10,20))
    plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")
    plt.axis('off')
    plt.show()

'''------Positve Sentiment--------'''

all_words = ' '.join([text for text in tweets['clean_text'][tweets.sentiment == 1]])
generate_wordcloud(all_words)

'''------Negative Sentiment--------'''

all_words = ' '.join([text for text in tweets['clean_text'][tweets.sentiment == -1]])
generate_wordcloud(all_words)

'''------Neutral Sentiment--------'''

all_words = ' '.join([text for text in tweets['clean_text'][tweets.sentiment == 0]])
generate_wordcloud(all_words)



'''---------Pie Graph-----------'''

def percentage(part,whole):
    return 100*float(part)/float(whole)

positive = 0
negative = 0
neutral = 0
polarity = 0


for tweet in tweets.fully_clean_text:
    analyzer = TextBlob(tweet)
    polarity += analyzer.sentiment.polarity
    if analyzer.sentiment.polarity > 0:
        positive += 1
    elif analyzer.sentiment.polarity < 0:
        negative += 1
    else:
        neutral += 1
        
# print(positive)
# print(negative)
# print(neutral)
# print(polarity)

positive = percentage(positive,(positive + negative + neutral))
negative = percentage(negative,(positive + negative + neutral))
neutral = percentage(neutral,(positive + negative + neutral))

positive = format(positive,'.2f')
negative = format(negative,'.2f')
neutral = format(neutral,'.2f')

if (polarity > 0):
    print("Positive")
elif (polarity < 0):
    print("Negative")
elif (polarity == 0):
    print("Neutral")

labels = ['Positive ['+str(positive)+'%]', 'Negative ['+str(negative)+'%]', 
'Neutral ['+str(neutral)+'%]']
sizes = [positive, negative, neutral]
colors = ['lightskyblue','gold','lightcoral']
patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.legend(patches,labels,loc="best")
plt.title("Polarity Pie Chart")
plt.axis('equal')
plt.tight_layout()
plt.show()



'''-----------Tweets Location Bargraph----------'''

tweets['location'].value_counts().head(30).plot(kind='barh', figsize=(6,10))



'''-------Fluctuations of Polarity---------'''

fig = plt.figure(figsize=(18,7))
sns.distplot(tweets['polarity'],kde=False)
plt.ylim(0,10000)



'''-------Fluctuations of Subjectivity---------'''

fig = plt.figure(figsize=(18,7))
sns.distplot(tweets['subjectivity'],kde=False)
plt.ylim(0,10000)



'''---------Creating Hastag FreqDist----------'''

# function to collect hashtags
def hashtag_extract(text_list):
    hashtags = []
    # Loop over the words in the tweet
    for text in text_list:
        ht = re.findall(r"#(\w+)", text)
        hashtags.append(ht)
    return hashtags

def generate_hashtag_freqdist(hashtags):
    a = nltk.FreqDist(hashtags)
    d = pd.DataFrame({'Hashtag': list(a.keys()),
                      'Count': list(a.values())})
    # selecting top 15 most frequent hashtags     
    d = d.nlargest(columns="Count", n = 25)
    plt.figure(figsize=(16,7))
    ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
    plt.xticks(rotation=80)
    ax.set(ylabel = 'Count')
    plt.show()
    
hashtags = hashtag_extract(tweets['text'])
hashtags = sum(hashtags, [])

generate_hashtag_freqdist(hashtags)



'''---------Plotting Confusion Matrix-----------'''

conf_matrx = confusion_matrix(predictions,sent_test)
plt.figure(figsize = (10,7))
sns.heatmap(conf_matrx,annot=True)