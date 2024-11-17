import tensorflow as tf
import numpy as np

def Colaborative_Filtering_model(train, test):
    num_users = len(np.unique(train[:,0]))
    num_products =  len(np.unique(train[:,1]))
    print(num_users)

    user_input = tf.keras.layers.Input(shape=(1,))
    product_input = tf.keras.layers.Input(shape=(1,))



    user_embedding = tf.keras.layers.Embedding(num_users, 10)(user_input)
    dense_user_1 = tf.keras.layers.Dense(128, activation='relu')(user_embedding)
    dense_user_2 = tf.keras.layers.Dense(16, activation='relu')(dense_user_1)



    product_embedding = tf.keras.layers.Embedding(num_products, 10)(product_input)
    dense_product_1 = tf.keras.layers.Dense(128, activation='relu')(product_embedding)
    dense_product_2 = tf.keras.layers.Dense(16, activation='relu')(dense_product_1)


    merged_embeddings = tf.keras.layers.Concatenate()([dense_user_2, dense_product_2])
    flatten_embeddings = tf.keras.layers.Flatten()(merged_embeddings)

    output = tf.keras.layers.Dense(1, activation='linear')(flatten_embeddings)

    model = tf.keras.models.Model(inputs=[user_input, product_input], outputs=output)
    
    return model



def Content_Based_model():
    num_outputs = 32
    tf.random.set_seed(1)

    user_NN = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_outputs, activation='linear'),
    ])

    product_NN = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_outputs, activation='linear'),
    ])

    input_user = tf.keras.layers.Input(shape=(1,))
    vu = user_NN(input_user)
    vu = tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vu)

    input_item = tf.keras.layers.Input(shape=(1,))
    vm = product_NN(input_item)
    vm = tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vm)

    output = tf.keras.layers.Dot(axes=1)([vu, vm])

    model = tf.keras.models.Model(inputs=[input_user, input_item], outputs=output)

    return model
