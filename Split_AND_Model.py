
'''----------------------Model---------------------'''

'''------Splitting Data-------'''

from sklearn.model_selection import train_test_split

msg_train, msg_test, sent_train, sent_test = train_test_split(x,y, test_size=0.3,random_state = 88)
# msg_train.shape
# msg_test.shape


'''-------Fit to pipeline--------'''

# using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(msg_train, sent_train)


'''--------Predicting the accuracy and confusion_matrix---------'''

# predictions = pipeline.predict(msg_test)
predictions = classifier.predict(msg_test)

print(classification_report(predictions,sent_test))
print ('\n')
print(confusion_matrix(predictions,sent_test))
print("The accuracy of this model is")
print(accuracy_score(predictions,sent_test)*100)