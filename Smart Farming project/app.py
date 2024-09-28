#import necessery library for app
from http.client import REQUEST_ENTITY_TOO_LARGE
import string
from urllib import response
from urllib.request import urlopen
from flask import Flask,request,render_template,jsonify
import numpy as np
import pickle
import requests

# chatbot library
import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np

#instantiate falsk 
app =Flask(__name__)


#load random model for fertilizer recommendation part
random_model= pickle.load(open('random_model.pkl','rb'))

#load decesion tree model crop recommendation part
reg_modele = pickle.load(open('DecisionTree.pkl','rb'))


#method to display the Home page 
@app.route('/',methods=["GET"])
def crop_home():   
     return render_template('Home.html')
     
@app.route('/crop',methods=["GET"])
def crop_recommendation_home():   
    return render_template('crop.html')

#method to predict the crop recommendation parameter variable 
@app.route("/predict", methods =["POST"])
def predict():   
    float_feature=[float(item) for item in request.form.values()]
    feature=[np.array(float_feature)]
    
    #predict model
    prediction=reg_modele.predict(feature)
    
    #return pridicted output into page 
    return render_template('crop.html',prediction_table=prediction)
    
    
    #fertilizer recommendation Home page method 
@app.route("/fertilizer",methods=["GET"])
def fertilizer_page():
    return render_template('fertilizer.html')
    
    
#fertilizer prediction method for input parameter 
@app.route("/fertilizer", methods =["POST"])
def Fertilizer_predict(): 
    
    #Accept all input from the from one by one 
     var1=request.form["Temp"]
     var2=request.form["humidity"]
     var3=request.form["moisturet"]
     var4=request.form["Soil type"]
     var5=request.form["Crop Type"]
     var6=request.form["N"]
     var7=request.form["K"]
     var8=request.form["P"]
      
     
     
      
    # convert input varaible into nupmpy array
     total_input=np.array([var1,var2,var3,var4,var5,var6,var7,var8])  
     total_input=total_input.reshape(1, -1)
     
     
    #  predict fertilizer using random model
     prediction2=random_model.predict(total_input)
    
    # return fertilizer recommendation output
     if int(prediction2)==0:
          
          return render_template("fertilizer.html",fertiizer="10/26/26")
   
     elif int(prediction2)==1:
         return render_template("fertilizer.html",fertiizer="14-35-14")
        
     elif int(prediction2)==2:
         return render_template("fertilizer.html",fertiizer="17-17-17")  
       
              
     elif int(prediction2)==3: 
          return render_template("fertilizer.html",fertiizer="20-20")
     
     elif int(prediction2)==4: 
          return render_template("fertilizer.html",fertiizer="28-28 ")
    
    
     elif int(prediction2)==5: 
          prediction2=="12-12-12"
          return render_template("fertilizer.html",fertiizer="DAP") 
     
     elif int(prediction2)==6: 
          prediction2=="12-12-12"
          return render_template("fertilizer.html",fertiizer="UREA")
     
     else:
         return render_template("fertilizer.html",fertiizer="You should use other appropriate inorganic fertilizer")         
         
    #  0-->10/26/26
    #  1-->14-35-14
    #  2-->117-17-17
    #  3-->'20-20'
    #  4-->'28-28'
    #  5-->'DAP'
    #  6-->'UREA'
    #  return render_template('fertilizer.html',
    #                         prediction_table=
    #                         'ለርሰወ አፈር የሚስማማው የመዳበርያ አይነት Fertilizer Type:{}'.format(prediction2))
     
       
@app.route("/crop_recommendation", methods =["GET"])   
def crop_recommendation():    
     return render_template('crop.html')
     
#open weather map Api         
@app.route("/weather_home", methods =["GET"])   
def weatehr_page():   
    

    return render_template('weather.html')
     
@app.route("/weather_result", methods =["POST"])   
def weatehr():
     weather_API="https://api.openweathermap.org/data/2.5/weather?q={}&appid=498330f0758d3b7241d4a98b4f2776d4"
     city= request.form["city"]
    # Accept contry for weather API 
     response=requests.get(weather_API.format(city))
     response=response.json();
  
     temp=float(response['main']['temp'])
     humidity=float(response['main']['humidity'])
     
    # convert kelvin temprature into degree centigried
     temp=temp-273
    #  return render_template('weather.html',weather_result="{} Humidity level={}  and temprature= {}'c  ".format(city,humidity,temp))
     return render_template('weather.html',temp=temp,hu=humidity)

     
    # float_feature2=[float(item) for item in request.form.values()]
    # feature2=[np.array(float_feature2)]
        
        #predict model
    # prediction2=random_model.predict(feature2)
   
        


# chatbot code
# basic library for chatbot 
from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents=json.loads(open('data.json',encoding='utf8').read())
# intents = open('data.json',encoding="utf8").read()
# data_file = open('data.json',encoding="utf8").read()
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res



app.static_folder = 'static'

@app.route("/chatbot")
def home():
    return render_template("chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)
    
 #flask to run    
if __name__=="__main__":
   app.run(debug=True)


