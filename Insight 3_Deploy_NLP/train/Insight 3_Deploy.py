#!/usr/bin/env python
# coding: utf-8

# # <h1 style="font-family: Trebuchet MS; padding: 12px; font-size: 48px; text-align: center; line-height: 1.25;"><b>!!Analisis Sentimen Terhadap Film Menggunakan Algoritma Naive Bayes!!<span style="color: #000000"></span></b><br><span style="color: #6fa3a7; font-size: 24px">Insight 3</span></h1>
# <hr>

# Nama Kelompok :
# - Tiara Auliya Putri
# - Gusti Made Wijaya Kusuma
# - Stephania Getrudis Inaconta Sadipun
# - I Gusti Ketut Adi Triyoga Putra
# - Farhan Rahman

# # 1. Data Acquisition 
# Data yang digunakan adalah dataset bernama dataset_tweet_sentiment_opini_film.csv yang didalamnya berisi 200 baris dan 3 kolom. 3 kolom tersebut adalah Id, Sentiment dan Text Tweet. Sentiment disini sebrperan sebagai variabel y dan Text Tweet sebagai variabel x.

# In[1]:


#Mengimport library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Install library Sastrawi
get_ipython().system('pip -q install sastrawi')


# In[3]:


#Mengimport NLTK dimana didalamnya berisi stopword yang akan digunakan untuk text preprocessing
import nltk
nltk.download('stopwords')


# In[4]:


# Mendownload dataset
dataset = pd.read_csv('dataset_tweet_sentiment_opini_film.csv')
dataset


# Dataset diatas berisi 200 baris dengan 3 kolom yang memiliki 2 sentimen yaitu positive dan negative

# In[5]:


#menampilkan info dataset
dataset.info()


# In[6]:


#Menampilkan jumlah semua sentimen dan jumlah sentimen negative dan positive
print('Total Jumlah Sentimen :', dataset.shape[0], 'data\n')
print('terdiri dari (label):')
print('1. Sentiment Negative  :', dataset[dataset.Sentiment == "negative"].shape[0], 'data')
print('2. Sentiment Positive  :', dataset[dataset.Sentiment == "positive"].shape[0], 'data')


# Pada output terlihat bahwa jumlah sentimen ada 200 dengan pembagian yang merata antara sentimen negative dan positive yaitu 100 dan 100.

# In[7]:


#Menampilkan visualisasi pembagian jumlah sentimen untuk negative dan positive
height = dataset['Sentiment'].value_counts()
labels = ('Sentiment Negative', 'Sentiment Positive')
y_pos = np.arange(len(labels))

plt.figure(figsize=(8,5), dpi=100)
plt.ylim(0,300)
plt.title('Distribusi Kategori Sentiment', fontweight='bold')
plt.xlabel('Kategori', fontweight='bold')
plt.ylabel('Jumlah', fontweight='bold')
plt.bar(y_pos, height, color=['deepskyblue'])
plt.xticks(y_pos, labels)
plt.show()


# Pada output terlihat bahwa pembagian antar sentimennya rata.

# # 2. Text Preprocessing

# ## 2.1 Case Folding

# In[8]:


#Membuat function untuk melakukan case folding
import re
def casefolding(text):
  text = text.lower()                               # Mengubah teks menjadi lower case
  text = re.sub(r'https?://\S+|www\.\S+', '', text) # Menghapus URL
  text = re.sub(r'[-+]?[0-9]+', '', text)           # Menghapus angka
  text = re.sub(r'[^\w\s]','', text)                # Menghapus karakter tanda baca
  text = text.strip()
  return text


# In[9]:


raw_sample = dataset['Text Tweet'].iloc[10]
case_folding = casefolding(raw_sample)

print('Raw data\t: ', raw_sample)
print('\n')
print('Case folding\t: ', case_folding)


# Text processing yang pertama dilakukan adalah case folding yang bertujuan untuk membersihkan sentimen yakni dengan mengubahkan ke huruf kecil sehingga semua sentimen menjadi sama, lalu mengubah menghapus URL, menghapus angka dan menghapus karakter tanda baca. Pada ouput terlihat bahwa raw data sudah dibersihkan dengan menggunakan function casefolding.

# ## 2.2. Word Normalization
# 

# In[10]:


# Mendownload corpus kumpulan slangwords 
key_norm = pd.read_csv('https://raw.githubusercontent.com/ksnugroho/klasifikasi-spam-sms/master/data/key_norm.csv')
print(key_norm.head(10))

key_norm.shape


# In[11]:


# Membuat fucntion untuk normalisasi teks dengan nama textnormalize
def textnormalize(text):
  text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] if (key_norm['singkat'] == word).any() else word for word in text.split()])
  text = str.lower(text)
  return text


# Text preprocessing kedua yakni melakukan word normalization, dimana ini dilakukan dengan mendownload corpus slangword sehingga dapat dipakai untuk untuk mengubah kata gaul atau kata tidak baku menjadi kata baku yang nantinya digunakan pada function textnormalize. Pada output ditampilkan beberapa sample perubahan kata, misalnya kata abis akan diubah menjadi habis.

# ## 2.3. Filtering (Stopword Removal)

# In[12]:


# Mengimport corpus stopword bahasa indonesia dan bahasa inggris
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

stopwords_ind = stopwords.words('indonesian')
stopwords_eng = stopwords.words('english')


# In[13]:


# Melihat jumlah kata dalam corpus stopword bahasa indonsia
len(stopwords_ind)


# In[14]:


# Melihat jumlah kata dalam corpus stopword bahasa inggris
len(stopwords_eng)


# In[15]:


# stopword bahasa indoensia
stopwords_ind[:1000]


# In[16]:


# stopword bahasa inggris
stopwords_eng[:1000]


# In[17]:


# Membuat fungsi untuk stopword removal dengan nama remove_stopword
# Terdapat juga more_stopword yang memungkinkan developer memberikan kata-kata yang ingin dihapus yang belum terdaftar pada corpus
more_stopword = ['gwa', 'kok', 'ah', 'kebayang', 'ttg', 'ya', 'yah', 'duh', 'tll', 'sih', 'keingat', 'ogah', 'greget', 'lumyan', 'bgus', 'kaya', 'nntn', 'diharepin', 'nungguin', 'ma', 'nih', 'ajaa', 'nyedot', 'mbuh', 'dah', 'bingittsss', 'terllu', 'kentel', 'gegara']                    # Tambahkan kata lain dalam daftar stopword
stopwords_ind = stopwords_ind + more_stopword

def remove_stop_words(text):
  clean_words = []
  text = text.split()
  for word in text:
      if word not in stopwords_ind:
          clean_words.append(word)
  return " ".join(clean_words)


# In[18]:


# Menerapkan stopword removal dan text normalize pada sentimen yang sudah dilakukan case folding
raw_sample = dataset['Text Tweet'].iloc[10]
case_folding = casefolding(raw_sample)
stopword_removal = remove_stop_words(case_folding)
text_normalize = textnormalize(stopword_removal)


print('Raw data             : ', raw_sample)
print('\n')
print('Case folding         : ', case_folding)
print('\n')
print('Stopword removal     : ', stopword_removal)
print('\n')
print('Text Normalize       : ', text_normalize)
print('\n')


# Text preprocessing selanjutnya adalah stopword removal. Stopword removal digunakan untuk menghapus kata-kata yang tidak penting seperti konjugasi, kata hubung dan kata-kata tidak penting lainnya yang terdapat pada dataset. Pada output, dapat dilihat bahwa setelah diterapkan text normalize dan stowprd removal, kata-kata yang tidak baku sudah di handle dan juga beberapa kata tidak penting seperti kata 'namun' sudah dihapus

# ## 2.4. Stemming

# In[19]:


# Membuat function stemming 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(text):
  text = stemmer.stem(text)
  return text


# In[20]:


# Menerapkan stemming ke sentimen yang sebelumnya telah diterapkan case folding, stopword removal dan text normalize,
raw_sample = dataset['Text Tweet'].iloc[10]
case_folding = casefolding(raw_sample)
stopword_removal = remove_stop_words(case_folding)
text_normalize = textnormalize(stopword_removal)
text_stemming = stemming(text_normalize)

print('Raw data\t\t: ', raw_sample)
print('\n')
print('Case folding\t\t: ', case_folding)
print('\n')
print('Stopword removal\t: ', stopword_removal)
print('\n')
print('Text Normalize\t\t: ', text_normalize)
print('\n')
print('Stemming\t\t: ', text_stemming)


# Text preprocessing yang terakhir adalah stemming yaitu mengubah kata imbuhan menjadi kata dasar. Dapat dilihat pada output bahwa setelah diterapkan stemming kata yang berimbuha sudah berubah ke kata dasarnya, misalnya 'dilupakan' berubah menjadi 'lupa'. Perubahan ini diperlukan sehingg tidak terlalu banyak kata (feature) pada text, misalnya kata kerja hanya 1 tetapi jika tidak diubah ke kata dasar, maka akan ada banyak kata misalnya bekerja, pekerjaan, mempekerjakan, dan lain-lain.

# ## 2.5. Gabungan Seluruh Text Preprocessing

# In[21]:


# Membuat function untuk menggabungkan seluruh text preprocessing
def text_preprocessing_process(text):
  text = casefolding(text)
  text = textnormalize(text)
  text = remove_stop_words(text)
  text = stemming(text)
  return text


# In[22]:


# Menyimpan data teks yang sudah bersih 

get_ipython().run_line_magic('time', '')
dataset['Clean Text Sentiment Film'] = dataset['Text Tweet'].apply(text_preprocessing_process)


# In[23]:


dataset


# In[24]:


# Menyimpan data yang telah dilakukan seluruh text preprocessing agar kita kita tidak perlu menjalankan ulang proses tersebut dari awal
dataset.to_csv('clean_text_sentimen_film.csv')


# # 3. Feature Engineering

# In[25]:


# Memisahkan variabel x (kolom feature) dan variabel y (kolom target)
X = dataset['Text Tweet']
y = dataset['Sentiment']


# In[26]:


# Menampilkan isi variabel x
X


# In[27]:


# Menampilkan isi variabel y
y


# ## 3.1. Feature Extraction (TF-IDF & N-Gram)

# In[28]:


# Mengimplementasikan feature extraction menggunakan TF-IDF dengan memanfaatkan library python
from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf = TfidfVectorizer(ngram_range=(1,1))
tf_idf.fit(X)


# In[29]:


# Melihat jumlah fitur yang akan dilakukan TF-IDF
print(len(tf_idf.get_feature_names_out()))


# In[30]:


# Melihat fitur-fitur apa saja yang ada di dalam corpus
print(tf_idf.get_feature_names_out())


# In[31]:


# Melihat matriks jumlah token
# Data ini siap untuk dimasukkan dalam proses pemodelan (machine learning)

X_tf_idf = tf_idf.transform(X).toarray()
X_tf_idf


# In[32]:


# Melihat matriks jumlah token menggunakan TF-IDF
# Data ini siap untuk dimasukkan dalam proses pemodelan (machine learning)
# Selanjutnya X_tf_idf (data featurenya) dilakukan perubahan kata menjadi vektor lalu disimpan pada variabel baru bernama 'data_tf_idf'

data_tf_idf = pd.DataFrame(X_tf_idf, columns=tf_idf.get_feature_names_out())
data_tf_idf


# Ini adalah proses perubahan kata menjadi vektor dengan menggunakan TF-IDF yang ada pada library python. Kata-kata yang diubah berada pada variabel x yang mana terdapat 1172 dalam corpus TF-IDF yang akan diubah menjadi vektor. Setelah diubah, hasil selanjutnya ditampilkan dalam bentuk data frame.

# In[33]:


# Data teks yang sudah berubah menjadi vektor akan disimpan pada file pickle
with open('tf_idf_feature_sentimen.pickle', 'wb') as output:
  pickle.dump(X_tf_idf, output)


# ## 3.2. Feature Selection (Chi Square)

# In[34]:


# Mengubah nilai data tabular TF-IDF menjadi array agar dapat dijalankan pada proses seleksi fitur
X = np.array(data_tf_idf)
y = np.array(y)


# In[35]:


# Mengimplementasikan feature selection (pengurangan feature) dengan menggunakan chi-square (melihat score setiap feature)
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 

chi2_features = SelectKBest(chi2, k=1100)  # mengambil 1100 feature yang paling bagus (scorenya tinggi)
X_kbest_features = chi2_features.fit_transform(X, y) # 1100 feature pada variabel x yang terpilih akan masuk pada variabel 'X_kbest_feature'
   
print('Jumlah keseluruhan feature\t\t     :', X.shape[1]) 
print('Jumlah feature setelah mengalami pengurangan :', X_kbest_features.shape[1]) 


# Pada bagian ini, feature sudah selesai dicut atau dikurangi sehingga yang digunakan hanya 1100 feature saja, tetapi dibawahnya akan saya tampilkan bagaimana chi-square mengurangi featurenya. Chi-square mengurasngi feature dengan cara mengurutkan setiap feature berdasarkan scorenya, lalu mengambil 1100 feature teratas (1100 feature dengan score terbesar).

# In[36]:


# chi2_features.scores_ adalah nilai chi-square, semakin tinggi nilainya maka semakin baik fiturnya
data_chi2 = pd.DataFrame(chi2_features.scores_, columns=['Score'])

# Menampilkan fitur beserta nilainya
feature = tf_idf.get_feature_names_out()
data_chi2['Feature'] = feature
data_chi2


# In[37]:


# Mengurutkan fitur terbaik
data_chi2.sort_values(by='Score', ascending=False)


# In[38]:


# Menampilkan mask pada feature yang diseleksi
# False berarti feature tidak terpilih dan True berarti feature terpilih
mask = chi2_features.get_support()
mask


# 1100 feature yang sudah terpilih akan digabungkan dengan corpus pada vocab yang terbentuk dari feature extraction kita yang akan disimpan pada variabel mask, caranya dengan penentuan true false pada vocab yang terbentuk dari feature extraction, dimana false berarti feature tidak terpilih dan true berarti feature terpilih.

# In[39]:


# Menampilkan fitur-fitur terpilih berdasarkan mask atau nilai tertinggi yang sudah dikalkulasi pada chi-square
new_feature = []

for bool, f in zip(mask, feature):
  if bool:
    new_feature.append(f)
  selected_feature = new_feature

selected_feature


# Pada saat diseleksi, jika feature bernilai true, maka feature tersebut akan dipilih dan dimasukan ke dalam variabel 'new_feature' dan yang tidak terpilih akan bernilai false dan tidak akan digunakan.

# In[40]:


# Cara melihat vocab yang dihasilkan oleh TF_IDF
# tf_idf.vocabulary_ 

kbest_feature = {} # Buat dictionary kosong untuk menyimpan feature terpilih

#perulangan untuk mengecek setiap feature yang dihasilkan TF-IDF, kemudian akan dicek apakah feature tersebut ada di dalam daftar feature terseleksi, kalau ada, maka akan dimasukan ke dalam dictionary kosong
for (k,v) in tf_idf.vocabulary_.items():    
  if k in selected_feature:                 
    kbest_feature[k] = v                    


# Feature-feature yang dipilih tersebut akan masuk ke dalam dictionary kosong dengan nama kbest_feature.

# In[41]:


# Menampilkan feature-feature yang terpilih beserta score TF-IDF nya
kbest_feature


# In[42]:


# Menampilkan feature-feature yang sudah diseleksi (1100) beserta nilai vektornya (hasil dari TF-IDF) pada keseluruhan data untuk dijalankan pada proses machine learning
data_selected_feature = pd.DataFrame(X_kbest_features, columns=selected_feature)
data_selected_feature


# Dapat pada output diatas, feature yang sebelumnya berjumlah 1172 sudah berkurang menjadi 1100 dan kita bisa melihat bahwa setiap feature sudah memiliki vektornya masing-masing.

# In[43]:


with open('selected_feature.pickle', 'wb') as output:
  pickle.dump(kbest_feature, output)


# # 4. Modelling (Machine Learning)

# In[44]:


# Mengimport algoritma-algoritma yang akan digunakan untuk klasifikasi dari library python
from sklearn.naive_bayes import MultinomialNB        # naive bayes        
from sklearn.linear_model import LogisticRegression  # logistik regressi
from sklearn.svm import SVC                          # super vektor machine
from sklearn.model_selection import train_test_split    # Digunakan untuk memisahkan data uji dan data latih
from joblib import dump                                 # Digunakan untuk menyimpan model yang telah dilatih


# In[45]:


# Mmeisahkan data training dan testing dengan perbandingan 80% untuk data training, 20% untuk data uji
# Random_state digunakan untuk internal random generator

X_train, X_test, y_train, y_test = train_test_split(X_kbest_features, y, test_size=0.20, random_state=40)


# # 4.1. Implementasi 3 Algoritma
# Mengimplementasikan ketiga algoritma untuk dilihat perbandingan akurasinya

# In[46]:


# Mengimplementasikan algoritma super vector machine
algorithm = SVC()               
model1 = algorithm.fit(X_train, y_train)   

# Menyimpan model hasil traning
dump(model1, filename='model1.joblib')


# In[47]:


# Mengimplementasikan algoritma naive bayes
algorithm = MultinomialNB()               
model2 = algorithm.fit(X_train, y_train)    

# Menyimpan model hasil traning
dump(model2, filename='model2.joblib')


# In[48]:


# Mengimplementasikan algoritma losgistic regression
algorithm = LogisticRegression()               
model3 = algorithm.fit(X_train, y_train)   

# Menyimpan model hasil traning
dump(model3, filename='model3.joblib')


# In[49]:


# Melakukan prediksi pada data trainig dengan algoritma super vector machine
model_pred1 = model1.predict(X_test) 

# Tampilkan hasil prediksi label dari model
model_pred1


# In[50]:


# Melakukan prediksi pada data trainig dengan algoritma naive bayes
model_pred2 = model2.predict(X_test) 

# Tampilkan hasil prediksi label dari model
model_pred2


# In[51]:


# Melakukan prediksi pada data trainig dengan algoritma logistic regression
model_pred3 = model3.predict(X_test) #naive bayes

# Tampilkan hasil prediksi label dari model
model_pred3


# In[52]:


# Tampilkan label sebenarnya pada data testing 
y_test


# # 5. Model Evaluation

# In[53]:


# Menampilkan jumlah prediksi yang benar dan prediksi yang salah dari data testing beserta akurasinya

print("Super Vector Machine")
prediksi_benar1 = (model_pred1 == y_test).sum()
prediksi_salah1 = (model_pred1 != y_test).sum()
print('Jumlah prediksi benar\t:', prediksi_benar1)
print('Jumlah prediksi salah\t:', prediksi_salah1)
accuracy1 = prediksi_benar1 / (prediksi_benar1 + prediksi_salah1)*100
print('Akurasi pengujian\t:', accuracy1, '%')
print('\n')

print("Naive Bayes")
prediksi_benar2 = (model_pred2 == y_test).sum()
prediksi_salah2 = (model_pred2 != y_test).sum()
print('Jumlah prediksi benar\t:', prediksi_benar2)
print('Jumlah prediksi salah\t:', prediksi_salah2)
accuracy2 = prediksi_benar2 / (prediksi_benar2 + prediksi_salah2)*100
print('Akurasi pengujian\t:', accuracy2, '%')
print('\n')

print("Logistic Regression")
prediksi_benar3 = (model_pred3 == y_test).sum()
prediksi_salah3 = (model_pred3 != y_test).sum()
print('Jumlah prediksi benar\t:', prediksi_benar3)
print('Jumlah prediksi salah\t:', prediksi_salah3)
accuracy3 = prediksi_benar3 / (prediksi_benar3 + prediksi_salah3)*100
print('Akurasi pengujian\t:', accuracy3, '%')
print('\n')


# In[54]:


from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_test, model_pred1)
print('Confusion matrix:\n', cm1)
print('\n')


cm2 = confusion_matrix(y_test, model_pred2)
print('Confusion matrix:\n', cm2)
print('\n')


cm3 = confusion_matrix(y_test, model_pred3)
print('Confusion matrix:\n', cm3)
print('\n')


# In[55]:


from sklearn.metrics import classification_report

print('Classification report:\n', classification_report(y_test, model_pred1))



print('Classification report:\n', classification_report(y_test, model_pred2))



print('Classification report:\n', classification_report(y_test, model_pred3))


# In[56]:


# Menampilkan akurasi setiap model / algoritma pada cross validation

# super vector machine
from sklearn.model_selection import ShuffleSplit    
from sklearn.model_selection import cross_val_score 

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=50)

cv_accuracy1 = (cross_val_score(model1, X_kbest_features, y, cv=cv, scoring='accuracy'))
avg_accuracy1 = np.mean(cv_accuracy1)

print('Akurasi setiap split:', cv_accuracy1)
print('Rata-rata akurasi pada cross validation:', avg_accuracy1, '\n', '\n')


# In[57]:


# naive bayes
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=50)

cv_accuracy2 = (cross_val_score(model2, X_kbest_features, y, cv=cv, scoring='accuracy'))
avg_accuracy2 = np.mean(cv_accuracy2)

print('Akurasi setiap split:', cv_accuracy2)
print('Rata-rata akurasi pada cross validation:', avg_accuracy2, '\n', '\n')


# In[58]:


# logistic regression

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=50)

cv_accuracy3 = (cross_val_score(model3, X_kbest_features, y, cv=cv, scoring='accuracy'))
avg_accuracy3 = np.mean(cv_accuracy3)

print('Akurasi setiap split:', cv_accuracy3)
print('Rata-rata akurasi pada cross validation:', avg_accuracy3, '\n','\n')


# # 5.1. Model Comparison

# In[59]:


# Membuat tabel perbandingan
compare = pd.DataFrame({'Model': ['Super Vector Machine', 'Naive Bayes', 'Logistic Regression'], 
                        'Accuracy': [accuracy1, accuracy2, accuracy3], 
                        'Accuracy Cross Validation' : [avg_accuracy1, avg_accuracy2, avg_accuracy3]})

compare.sort_values(by='Accuracy', ascending=False).style.hide_index()


# Setelah dilakukan perbandingan, terlihat pada tabel bahwa akurasi terbesar dari klasifikasi sentimen terhadap film adalah model yang menggunakan naive bayes dengan akurasi sebesar 90%, maka dari itu deployment akan menggunakan naive bayes.

# # 6. Deployment
# Mencoba mengimplementasikan klasifikasi sentimen dengan algoritma naive bayes.

# In[60]:


from joblib import load

# load model naive bayes
model = load('model2.joblib')

# load vocabulary dari TF_idf yang sebelumnya sudah dimasukan ke dalam pickle file
vocab = pickle.load(open('selected_feature.pickle', 'rb'))


# In[ ]:


# Uji coba model
input_text = input("Masukan Text : ") 

pre_input_text = text_preprocessing_process(input_text)   # lakukan text pre processing pada text input

tf_idf_vec = TfidfVectorizer(vocabulary=set(vocab))       # definisikan TF_IDF

result = model.predict(tf_idf_vec.fit_transform([pre_input_text]))  # Lakukan prediksi

print('\nHasil Text Preprocessing :', pre_input_text)

print('\nHasil prediksi \t\t : ', input_text, 'adalah\n', result)


# In[ ]:




