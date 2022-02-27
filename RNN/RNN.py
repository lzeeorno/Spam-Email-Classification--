#import data, test set do not have label
import pandas as pd
import numpy as np
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
print(train.head())

#data cleaning
##empty data
print(np.sum(np.array(train.isnull()==True), axis=0))
print(np.sum(np.array(test.isnull()==True), axis=0))

##fill null data
train = train.fillna(" ")
test = test.fillna(" ")
print(np.sum(np.array(train.isnull()==True), axis=0))
print(np.sum(np.array(test.isnull()==True), axis=0))

#set label, 1 is spam, otherwise is not
print(train['spam'].unique())

#merge email content and subject into the feature
X_train = train['subject'] + ' ' + train['email']
y_train = train['spam']
X_test = test['subject'] + ' ' + test['email']

#transfer txt into tokens ids
from keras.preprocessing.text import Tokenizer
max_words = 300
tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
# 只给频率最高的300个词分配 id，其他的忽略
tokenizer.fit_on_texts(list(X_train)+list(X_test)) # tokenizer 训练
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)

#make sequence get same length
# 样本 tokens 的长度不一样，pad
maxlen = 100
from keras.preprocessing import sequence
X_train_tokens_pad = sequence.pad_sequences(X_train_tokens, maxlen=maxlen,padding='post')
X_test_tokens_pad = sequence.pad_sequences(X_test_tokens, maxlen=maxlen,padding='post')

#build model
embeddings_dim = 30 # 词嵌入向量维度
from keras.models import Model, Sequential
from keras.layers import Embedding, LSTM, GRU, SimpleRNN, Dense
model = Sequential()
model.add(Embedding(input_dim=max_words, # Size of the vocabulary
                    output_dim=embeddings_dim, # 词嵌入的维度
                    input_length=maxlen))
model.add(SimpleRNN(units=64)) # 可以改为 GRU,SimpleRNN ， LSTM
model.add(Dense(units=1, activation='sigmoid'))
model.summary()


#training
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']) # 配置模型
history = model.fit(X_train_tokens_pad, y_train,
                    batch_size=128, epochs=10, validation_split=0.2)
model.save("email_cat_lstm.h5") # 保存训练好的模型


#plot training graph
from matplotlib import pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()



#test
pred_prob = model.predict(X_test_tokens_pad).squeeze()
pred_class = np.asarray(pred_prob > 0.5).astype(np.int32)
id = test['id']
output = pd.DataFrame({'id':id, 'Class': pred_class})
output.to_csv("submission_gru.csv",  index=False)
