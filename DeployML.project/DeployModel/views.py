from django.http import HttpResponse
from django.shortcuts import render
from joblib import load

def home(request):
    return render(request, "home.html") # HttpResponse("<h1>This is Home.</h1>")

def result(request):

    model = load('./Model/ML_model.joblib')

    lis = []
    lis.append(request.GET['RI'])
    lis.append(request.GET['Na'])
    lis.append(request.GET['Mg'])
    lis.append(request.GET['Al'])
    lis.append(request.GET['Si'])
    lis.append(request.GET['K'])
    lis.append(request.GET['Ca'])
    lis.append(request.GET['Ba'])
    lis.append(request.GET['Fe'])
    print(lis)

    ans = 2 # model.predict([lis]) TODO: for now it's giving error: 'DecisionTreeClassifier' object has no attribute 'n_features_'

    return render(request, "result.html", {'ans':ans, 'lis':lis})

