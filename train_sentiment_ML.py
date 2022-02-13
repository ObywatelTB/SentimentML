import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Input, Dropout
from tensorflow.keras import layers
from sklearn.utils import shuffle
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

import os
import time
from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print as rprint
import string
import re
from tqdm import tqdm   #progress bar
import gc #garbage collector
from itertools import compress  #do indeksów branych boolami

MODEL_NAME = 'Sentiment_ML'
DATASET_ENCODING = "ISO-8859-1"
BATCH_SIZE = 32    #int podzielny przez 8
EPOCH = 15 

#===getting data
def get_kaggle_tweets(dataset_path, start=0, amount = None):
    df = pd.read_csv(dataset_path,encoding=DATASET_ENCODING)
    if amount:
        df = df.iloc[start:amount]
    df= df.iloc[:,[0,-1]]
    df.columns = ['sentiment','tweet']
    df.sentiment = df.sentiment.map({0:0,4:1})
    print(f'Got {len(df)} tweets from kaggle set.')
    return df


def get_kaggle_encoded_tweets(dirs, dirname='kaggle1638301819'):
    #w przeciwnym razie header staje sie giga stringiem (pierwszym wektorem tweeta):
    dirpath = os.path.join(dirs['dir0'], 'models', 'tweets_encoded', dirname)
    dfs_arr = []
    filenames = os.listdir(dirpath)[:5]
    with tqdm(total= len(filenames) ) as progress_bar:
        for fn in filenames:
            filepath = os.path.join(dirpath, fn)
            dfs_arr.append( pd.read_csv(filepath, header=None,usecols=[1,2],names=['sentiment','tweet']) )
            progress_bar.update(1)
    return pd.concat(dfs_arr)


def string_to_nparray(a_string):
        a_list = [float(x) for x in a_string.replace('[','').replace(']','').split(',')]
        return np.asarray(a_list, dtype=np.float)


def split_datasets(df, train_size = 1024*70, test_size = 2048): #train size podzielny przez 8
    rprint('[italic red]Getting data... [/italic red]')
    #  rozmiar wszystkich zwróconych tweetów to MAX_TWEETS + TEST_SIZE
    #   max ilość tweetów dla enkodera: dla 0.33 GPU działa 4000.  dla 0.5 GPU działa 5000
    #zamiana stringa na liste:
    df['tweet'] = df['tweet'].apply(string_to_nparray)
     
    train_size = int(int(len(df)/8)*6)
    max_tweets = int(train_size / 8 * 10) 
    
    sent0 = df.query("sentiment==0")
    sent1 = df.query("sentiment==1")
    assert max_tweets+test_size <= len(df)
    
    tres1 = int(max_tweets/2)   #dzielimy przez 2, bo mamy dwie grupy sentiment
    tres2 = int(max_tweets/2) + int(test_size/2)
    new_df = pd.concat([sent0.iloc[:tres1], sent1.iloc[:tres1]])
    df_test = pd.concat([ sent0.iloc[tres1:tres2], 
                          sent1.iloc[tres1:tres2] ])
    new_df = shuffle(new_df).reset_index(drop=True)
    assert len(new_df) == max_tweets
    assert len(new_df.query("sentiment==0")) == tres1
    assert len(new_df.query("sentiment==1")) == tres1

    rprint(f'The number of analysed tweets:[italic red] {max_tweets+test_size}[/italic red].')
    df_train, df_val = train_test_split(new_df, test_size=0.2)
    return df_train, df_val, df_test


def prepare_data_set0(df):
    X = df.tweet.to_numpy()
    Y = df.sentiment.to_numpy()
    # numeric_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    # ds_final = numeric_dataset.shuffle(1000).batch(BATCH_SIZE) #numeric_batches

    AUTOTUNE = tf.data.AUTOTUNE
    # ds_final = ds_final.cache().prefetch(buffer_size=AUTOTUNE)
    return X,Y #ds_final


def switch_to_numpy_tuple(df):
    X = df.tweet.to_numpy()
    Y = df.sentiment.to_numpy()
    X = [np.asarray(x).astype('float32') for x in X]
    Y = [x.astype('float32') for x in Y]
    X = np.array(X)
    Y = np.array(Y)
    return (X,Y)


#===model,vectorization
def tweet_model(): 
    rprint('[italic red] Defining the model... [/italic red]')
    model = Sequential()
    model.add(Input(shape=(512,),dtype='float32')) #z przykladu  512
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    return model


def batch_generator(df, batch_size = BATCH_SIZE):#1024):
    #usage:   a=test_generator();   next(a); next(a)
    while True:
        for b in range(0, len(df), batch_size):
            new_df = df.iloc[b:b+batch_size]
            yield new_df # embeded_tweets, targets


def vectorize_df(df, embed):
    array = embed(df['tweet'].values.tolist()).numpy()
    df['tweet'] = [list(x) for x in array]  #UWAGA. PRZERABIA MACIERZ NA LISTE (zmniejsza wymiary z 2 do 1)
    return df #array[0].shape -> (512,)


def train(dirs, model, ds, val_ds):# train_ds, val_ds, steps_per_epoch=100):#embeded_tweets, targets):
    rprint('[italic red] Training the model... [/italic red]')
    num_epochs = EPOCH #10

    checkpoint_path = os.path.join(dirs['machine_learning'], MODEL_NAME, 'models', 'cp-.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    history = model.fit(x = ds[0], y = ds[1],#train_ds,    #pozwalające na robienie batchy
                        epochs=num_epochs,
                        validation_data = val_ds,
                        # validation_split=0.1,#validation_data = val_ds,
                        # steps_per_epoch=steps_per_epoch,#10000,
                        callbacks=[cp_callback])  
    return history, model


def visualize(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy') 
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


#===run eet
def predict(dirs):
    #Load the model
    model  = tweet_model()
    model.compile(loss='binary_crossentropy', 
            optimizer='adam',
            metrics=['acc'])
    checkpoint_path = os.path.join(dirs['models_checkpoints'], MODEL_NAME, 'cp2.ckpt')
    model.load_weights(checkpoint_path)

    #ours df
    filenames = ['0.csv'] #['0.csv','2.csv', '5.csv','7.csv']
    for const in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]: #fn in filenames
        filepath = dirs['our_rated_tweets_encoded']
        filepath = os.path.join(os.path.dirname(filepath), '0.csv') #fn
        df_ours = pd.read_csv(filepath, header=None,usecols=[1,2],names=['tweet','sentiment'])
        df_ours['tweet'] = df_ours['tweet'].apply(string_to_nparray)
        ours_tuple = switch_to_numpy_tuple(df_ours)

        # loss, acc = model.evaluate(ours_ds[0], ours_ds[1], verbose=2)
        # print(f"Restored model, tested on {fn} dataset. Accuracy: {100 * acc:5.2f}%", loss)
        def cut(x):
            if x < const or x > 1-const:
                return 1
            else:
                return 0
        pre = model.predict(ours_tuple[0])
        pre=([x[0] for x  in pre])
        # fil=[x>const for x in pre]
        fil = list(map(cut,pre))
        predictions = list(compress(pre,fil))
        predictions = list(map(round, predictions))
        check = list(compress(ours_tuple[1],fil))
        print(f'Len {len(predictions)}. Precyzja dla tres={const}:', accuracy_score(predictions, check)*100,'%')
        
    print('fin')


def vectorize_and_save_ds(dirs, which_df = 'kaggleNOT'):
    rprint('[italic red] Loading the encoder... [/italic red]')
    embed = hub.load(dirs['universal_sentence_encoder'])

    rprint('[italic red] Getting kaggle data... [/italic red]')
    if not which_df=='kaggle': #nasz zbiór ocenionych tweetów
        filename = dirs['our_rated_tweets_encoded']
        filename = os.path.join(os.path.dirname(filename),'7.csv')
        df = pd.read_csv(dirs['our_rated_tweets'],skiprows=[0,1],names=['tweet','sentiment'] )
        df = df[abs(df['sentiment'])>7]
        df['sentiment'] = df['sentiment'].apply(lambda x: round(x/20+0.5)) #(-10,10) -> (0,1)
        vec_df = vectorize_df(df, embed)
        vec_df.to_csv(filename, mode='a', header=False)
        return

    #1.6mln tweetów z kaggle'a
    df = get_kaggle_tweets(dirs['kaggle_dataset_tweets'])#, start=4, amount = 1024)#1024*500)
    dirpath = os.path.join(dirs['dir0'], 'models', 'tweets_encoded',f'kaggle{int(time.time())}')
    os.mkdir(dirpath)

    # df = df.iloc[::-1]
    df = shuffle(df).reset_index(drop=True)
    df= df.iloc[307200:]
    tweets_per_file = 1024*50 #około 100k program się wysypuje, więc daję ok 50k (wysypuje się chyba przy embed)
    parts_amount = int(len(df)/tweets_per_file)
    part_gen = batch_generator(df, batch_size = tweets_per_file)
    del df
    gc.collect()
    for inx in range(6,parts_amount):         #iteracja po pakietach przypadających na plik
        filename = os.path.join(dirpath, f'enc{inx}.csv')
        df_part = next(part_gen)
        max_batch = 256 # 1024 # przy ~4k embed definitywnie pada
        batch_amount = int(len(df_part)/max_batch)
        batch_gen = batch_generator(df_part, batch_size = max_batch)
        del df_part
        gc.collect()
        with tqdm(total= batch_amount ) as progress_bar:
            for inx in range(batch_amount): #iteracja po małych 256 t. pakietach, które umie przetrawić embed
                df_batch = next(batch_gen)
                encoded_batch = vectorize_df(df_batch, embed)
                del df_batch
                encoded_batch.to_csv(filename, mode='a', header=False) #append
                del encoded_batch
                gc.collect()
                progress_bar.update(1)


def create_model(dirs):
    #=model architecture
    logging.set_verbosity(logging.ERROR) # Reduce logging output (coś zmienia?)
    model  = tweet_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    #=getting data
    df = get_kaggle_encoded_tweets(dirs)
    df_ours = pd.read_csv(dirs['our_rated_tweets_encoded'],header=None,usecols=[1,2],names=['tweet','sentiment'])
    df_ours['tweet'] = df_ours['tweet'].apply(string_to_nparray)

    # df = pd.concat([df,df,df,df,df])    #dla 500k dzialalo
    df_train, df_val, df_test_kaggle = split_datasets(df)
    all_ds = [df_train, df_val, df_test_kaggle,  df_ours]
    train_ds, val_ds, test_ds_kaggle,  ours_ds = [switch_to_numpy_tuple(ads) for ads in all_ds]

    #=training
    history, model_fit = train(dirs, model, train_ds, val_ds)# embeded_tweets, targets)
    visualize(history)

    #=testing
    tests = {'kaggle':test_ds_kaggle, 'nasze tweety':ours_ds}# test = test_ds
    for key in tests:
        test = tests[key]
        pre = model_fit.predict(test[0])
        predictions = list(map(round,[x[0] for x  in pre]))
        print(f'Precyzja, {key}:', accuracy_score(predictions, test[1])*100,'%')

    # print('accu:',accuracy_score)
    print('fin')

