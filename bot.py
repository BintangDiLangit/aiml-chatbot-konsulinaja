from flask import Flask, render_template, request
import os
import aiml
import nltk
nltk.download('punkt')

app = Flask(__name__)

BRAIN_FILE = "./pretrained_model/aiml_pretrained_model.dump"
k = aiml.Kernel()

if os.path.exists(BRAIN_FILE):
    print("Loading from brain file: " + BRAIN_FILE)
    k.loadBrain(BRAIN_FILE)
else:
    print("Parsing aiml files")
    k.bootstrap(learnFiles="./pretrained_model/learningFileList.aiml",
                commands="load aiml")
    print("Saving brain file: " + BRAIN_FILE)
    k.saveBrain(BRAIN_FILE)


@app.route("/")
def home():
    return render_template("home.html")
 
def listToString(s):
 
    # initialize an empty string
    str1 = ""
 
    # traverse in the string
    for ele in s:
        str1 += ele + " "
 
    # return string
    return str1

def jaro_winkler_word(word1, word2):
    # inisialisasi variabel
    len1 = len(word1)
    len2 = len(word2)
    max_len = max(len1, len2)
    match_distance = max_len // 2 - 1
    matches1 = [False] * len(word1)
    matches2 = [False] * len(word2)
    common_chars = 0
    transpositions = 0
    jaro_distance = 0.0
    
    # cari karakter yang cocok dalam kedua kata
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if not matches2[j] and word1[i] == word2[j]:
                matches1[i] = matches2[j] = True
                common_chars += 1
                break
    
    # jika tidak ada karakter yang cocok, jarak = 0
    if common_chars == 0:
        return jaro_distance
    
    # cari transpositions
    k = transpositions = 0
    for i in range(len1):
        if matches1[i]:
            while not matches2[k]:
                k += 1
            if word1[i] != word2[k]:
                transpositions += 1
            k += 1
    
    # hitung jarak Jaro
    jaro_distance = (common_chars / len1 + common_chars / len2 + (common_chars - transpositions / 2) / common_chars) / 3.0
    
    # hitung skor Winkler
    if jaro_distance > 0.7:
        prefix = 0
        for i in range(min(len1, len2)):
            if word1[i] == word2[i]:
                prefix += 1
            else:
                break
        jaro_distance += prefix * 0.1 * (1 - jaro_distance)
    
    return jaro_distance

def find_best_match_word(word, dataset):
    # inisialisasi variabel
    best_match = ''
    best_similarity = 0
    
    # cari kata dengan jarak Jaro-Winkler tertinggi
    for item in dataset:
        similarity = jaro_winkler_word(word, item)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = item
    
    return best_match, best_similarity



@app.route("/get")
def get_bot_response():
    query = request.args.get('msg')
    # Case Folding - Mengecilkan kalimat
    lowerData = query.lower()
    # Tokenizing - Misah per kata
    tokenize = nltk.word_tokenize(lowerData)

    # Jaro Winkler
    with open('katadasar.txt') as f:
        lines = f.readlines()
        # Delete new line pada array
        arr = [s.strip() for s in lines]
        print(arr)

    for i, data in enumerate(tokenize) :
        res = find_best_match_word(data, arr)
        print(res)
        if res[1] >= 0.86 :
            tokenize[i] = res[0]

    result = listToString(tokenize)
    print(result)
    response = k.respond(result)
    if response:
        return (str(response))
    else:
        return (str(":)"))


if __name__ == "__main__":
    # app.run()
    app.run(host='0.0.0.0', port='5555')
