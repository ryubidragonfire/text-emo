# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:11:04 2017

@author: cyyam
"""
import pickle
import json

### To use the vectorizer and model in the future
model_filename = "./results/random-forest-demo-word-10estimators-12gram-model.pkl"
with open(model_filename, 'rb') as fin:
    le, vectorizer, clf = pickle.load(fin)

### Deserialise json input
json_input = '{"text": "wie sch n"}'
text_input = json.loads(json_input)  

### testing
X_new = vectorizer.transform([text_input['text']])
X_new_preds = clf.predict(X_new); print(X_new_preds)

### Serialise predicted lable to json
output = {}
output["Predicted_label"] = str(le.inverse_transform(X_new_preds).item())
json_output = json.dumps(output); print(json_output)
    
### simple testing
#new_samples = ["wie sch n"]
#X_new = vectorizer.transform(new_samples); 
#X_new_preds = clf.predict(X_new); print(X_new_preds)

### Labels
# 'attractiveness' 0
# 'curiosity' 1
# 'disgust' 2
# 'fear' 3
# 'germanangst' 4
# 'happiness' 5
# 'indulgence' 6
# 'neutral' 7
# 'sadness' 8
# 'surprise' 9]