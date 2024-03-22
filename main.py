import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import recall_score, f1_score, accuracy_score, roc_auc_score

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Load dataset
df = pd.read_csv("fraud.csv", index_col=0)
"""
                                      user_id  amount device_type  is_fraud  age    income     debt  credit_score
0        32fe1d6f-2c2e-476f-981b-b254a7bca753 4701.31      Mobile         1   29 139977.00 10069.32           437
1        d684726b-775b-451c-8e2a-7a5bfbdc840c 1904.93      Tablet         1   23 137934.91  3623.21           303
2        778f102c-7932-48b3-80e3-1375a152b791  381.68     Desktop         0   44  47439.27 44627.27           663
3        30458cfd-f9b7-4efb-afce-cc4470d5baf1 2262.44      Tablet         1   58  44551.91 16533.34           479
4        21076278-c8fa-4446-b002-ced7e8b870da   15.91     Desktop         0   32  88902.17  3719.38           695
                                       ...     ...         ...       ...  ...       ...      ...           ...
1920275  d1efd401-7cf3-461d-9936-18fdca061f17  233.76     Desktop         0   24  86110.04 25129.20           474
1920276  3c3be93c-cc06-40df-ac08-c0c663db97ee  589.03     Desktop         0   22  65111.03 44359.28           471
1920277  f5e7b397-4cb3-4716-a6b2-ba480477b8ea  782.52     Desktop         0   33  30342.53 47478.12           636
1920278  5e5b10c9-a915-4448-952d-8d2fac51ff7f 2429.14      Tablet         1   53  90436.71  3405.45           469
1920279  db3248c8-9432-4eff-a43e-478fc8a13533  235.08     Desktop         0   21  96415.34 33382.39           592

"""

# Drop 'user_id' column as it's not useful for modeling
df.drop(["user_id"], axis=1, inplace=True)

# Check for missing values
print("Missing values:\n",df.isnull().sum())
"""
amount          0
device_type     0
is_fraud        0
age             0
income          0
debt            0
credit_score    0
"""

# Boxplot visualization for outliers.
num_cols = [col for col in df.columns if col not in ["is_fraud"]]
for col in num_cols:
    sns.boxplot(x=df[col])
    plt.show(block=True)
# No outliers.

# Scale numerical features using StandardScaler
scale_cols = [col for col in df.columns if col not in ["device_type", "is_fraud"]]
ss = StandardScaler()
df[scale_cols] = ss.fit_transform(df[scale_cols])

# Encode categorical feature 'device_type' using LabelEncoder
le = LabelEncoder()
df["device_type"] = le.fit_transform(df["device_type"])
"""
         amount  device_type  is_fraud   age  income  debt  credit_score
0          2.01            1         1 -0.98    1.46 -1.03         -0.87
1          0.14            2         1 -1.37    1.41 -1.48         -1.71
2         -0.88            0         0 -0.00   -1.00  1.36          0.55
3          0.37            2         1  0.91   -1.08 -0.59         -0.60
4         -1.13            0         0 -0.78    0.10 -1.47          0.76
         ...          ...       ...   ...     ...   ...           ...
1920275   -0.98            0         0 -1.31    0.03  0.01         -0.63
1920276   -0.74            0         0 -1.44   -0.53  1.34         -0.65
1920277   -0.62            0         0 -0.72   -1.46  1.56          0.38
1920278    0.49            2         1  0.59    0.14 -1.49         -0.66
1920279   -0.98            0         0 -1.50    0.30  0.58          0.11
"""

# Modeling
X = df.drop("is_fraud", axis=1).values
y = df["is_fraud"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Define a neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Convert target variable to categorical
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Train the model
model.fit(X_train, y_train_categorical, epochs=2, batch_size=32, validation_split=0.2)

# Predict probabilities and classes
y_pred_proba = model.predict(X_test)[:, 1]
y_pred_classes = y_pred_proba.round()

# Calculate evaluation metrics
recall = recall_score(y_test, y_pred_classes, average='macro')
f1 = f1_score(y_test, y_pred_classes, average='macro')
accuracy = accuracy_score(y_test, y_pred_classes)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print evaluation metrics
print(f"""
Recall Score: {recall}
F1 Score: {f1}
Accuracy Score: {accuracy}
ROC-AUC Score: {roc_auc}
""")

"""
Recall Score: 0.9982765651295072
F1 Score: 0.9983030732345686
Accuracy Score: 0.9983058026260059
ROC-AUC Score: 0.9999883558011051
"""