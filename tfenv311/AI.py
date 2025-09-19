
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning messages
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


# Utility to clean and convert both datasets, then merge them into a unified DataFrame
def merge_and_clean_datasets(training_path, dataset_path, merged_path='merged_dataset.csv'):
    # 1. Load Training.csv (one-hot format)
    df_train = pd.read_csv(training_path)
    drop_cols = [col for col in df_train.columns if col.startswith('Unnamed') or df_train[col].sum() == 0]
    df_train = df_train.drop(columns=drop_cols)
    df_train = df_train.fillna(0)
    # Standardize column names (symptoms)
    df_train.columns = [col.strip().lower().replace(' ', '_') if col != 'prognosis' else 'prognosis' for col in df_train.columns]
    # 2. Load dataset.csv (symptom list format)
    df_new = pd.read_csv(dataset_path)
    # Gather all unique symptoms from Symptom columns
    symptom_set = set()
    for col in df_new.columns:
        if col.lower().startswith('symptom'):
            symptom_set.update(df_new[col].dropna().astype(str).str.strip().str.lower().replace('nan',''))
    symptom_set.discard('')
    # Also add all symptoms from Training.csv
    symptom_set.update([col for col in df_train.columns if col != 'prognosis'])
    symptom_list = sorted(symptom_set)
    # Build one-hot encoded DataFrame for dataset.csv
    records = []
    for _, row in df_new.iterrows():
        rec = {sym: 0 for sym in symptom_list}
        for col in df_new.columns:
            if col.lower().startswith('symptom') and pd.notna(row[col]):
                sym = str(row[col]).strip().lower()
                if sym:
                    rec[sym] = 1
        rec['prognosis'] = row['Disease'].strip()
        records.append(rec)
    df_new_clean = pd.DataFrame(records)
    # 3. Align columns for both DataFrames
    for sym in symptom_list:
        if sym not in df_train.columns:
            df_train[sym] = 0
        if sym not in df_new_clean.columns:
            df_new_clean[sym] = 0
    # Ensure same column order
    ordered_cols = symptom_list + ['prognosis']
    df_train = df_train[ordered_cols]
    df_new_clean = df_new_clean[ordered_cols]
    # 4. Concatenate and remove duplicates
    df_merged = pd.concat([df_train, df_new_clean], ignore_index=True)
    df_merged = df_merged.drop_duplicates()
    # 5. Save merged dataset
    df_merged.to_csv(merged_path, index=False)
    print(f"Merged and cleaned dataset saved to {merged_path}.")
    return df_merged




# 1. Merge and clean both datasets, always use merged data
df = merge_and_clean_datasets(
    'C:/Users/bubhi/OneDrive/Desktop/AI/tfenv311/Training.csv',
    'C:/Users/bubhi/OneDrive/Desktop/AI/tfenv311/dataset.csv',
    merged_path='C:/Users/bubhi/OneDrive/Desktop/AI/tfenv311/merged_dataset.csv'
)
print("Loaded and merged both datasets!")


# 2. Prepare features and labels
symptom_cols = [col for col in df.columns if col != 'prognosis']
X = df[symptom_cols].values
y = pd.factorize(df['prognosis'])[0]
label_names = pd.factorize(df['prognosis'])[1]

# 3. Oversample minority classes
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

# 4. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)


# 4. Define a function to build the model (for cross-validation and tuning)
def build_model(optimizer='adam'):
    from tensorflow.keras import regularizers
    model = models.Sequential([
        layers.Input(shape=(len(symptom_cols),)),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        layers.Dropout(0.4),
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        layers.Dropout(0.3),
        layers.Dense(len(label_names), activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# 5. Model training/loading logic

model_path = 'disease_model.h5'
# Check if model exists and if its input shape matches the merged data
retrain = True
if os.path.exists(model_path):
    try:
        best_model = load_model(model_path)
        # Check input shape compatibility
        if best_model.input_shape[-1] == X_train.shape[1]:
            print("Loaded trained model from file.")
            best_params = (16, 'adam')
            retrain = False
        else:
            print("Model input shape does not match data. Retraining...")
            os.remove(model_path)
    except Exception as e:
        print(f"Error loading model: {e}. Retraining...")
        os.remove(model_path)

if retrain:
    from sklearn.model_selection import StratifiedKFold
    from tensorflow.keras.callbacks import EarlyStopping
    import numpy as np
    best_val_acc = 0
    best_params = None
    best_model = None
    batch_sizes = [16, 32]
    optimizers = ['adam', 'rmsprop']
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for batch_size in batch_sizes:
        for optimizer in optimizers:
            val_accuracies = []
            for train_idx, val_idx in kfold.split(X_train, y_train):
                model = build_model(optimizer=optimizer)
                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(
                    X_train[train_idx], y_train[train_idx],
                    epochs=100,
                    batch_size=batch_size,
                    validation_data=(X_train[val_idx], y_train[val_idx]),
                    verbose=0,
                    callbacks=[early_stop]
                )
                val_loss, val_acc = model.evaluate(X_train[val_idx], y_train[val_idx], verbose=0)
                val_accuracies.append(val_acc)
            avg_val_acc = np.mean(val_accuracies)
            print(f"Batch size: {batch_size}, Optimizer: {optimizer}, Avg. val accuracy: {avg_val_acc:.2f}")
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                best_params = (batch_size, optimizer)
                best_model = build_model(optimizer=optimizer)
                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                best_model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=batch_size,
                    validation_split=0.1,
                    verbose=0,
                    callbacks=[early_stop]
                )
    # Save the trained model
    best_model.save(model_path)
    print(f"Model trained and saved to {model_path}.")

# 6. Evaluate the best model
loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nBest params: batch_size={best_params[0]}, optimizer={best_params[1]}")
print(f"Test accuracy: {accuracy:.2f}")

# 7. Prompt user for symptoms

# User-friendly symptom input prompt
print("\nWelcome to the AI Disease Predictor!")
print("You can enter your symptoms and the AI will predict the most likely disease.")
print("\nHere are some example symptoms you can use:")
example_symptoms = ', '.join(symptom_cols[:10]) + ', ...'
print(example_symptoms)
print("\nType your symptoms separated by commas (e.g. fever, headache, nausea)")

while True:
    user_input = input("Enter your symptoms: ").strip().lower()
    user_symptoms = [s.strip() for s in user_input.split(',') if s.strip()]
    # Validate at least one valid symptom
    valid = any(symptom.lower() in [col.lower() for col in symptom_cols] for symptom in user_symptoms)
    if not user_symptoms or not valid:
        print("\nPlease enter at least one valid symptom from the list. Try again.")
        print("Example symptoms:", example_symptoms)
    else:
        break

# 8. Create input vector for user
user_vector = np.zeros((1, len(symptom_cols)))
for idx, symptom in enumerate(symptom_cols):
    if symptom.lower() in user_symptoms:
        user_vector[0, idx] = 1

# 9. Predict disease and show top 3 most likely
pred = best_model.predict(user_vector)
top_indices = np.argsort(pred[0])[::-1][:3]
print("\nTop 3 predicted diseases:")
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank}. {label_names[idx]} (probability: {pred[0][idx]:.2f})")

# Print all probabilities for transparency
print("\nAll disease probabilities:")
for idx, name in enumerate(label_names):
    print(f"{name}: {pred[0][idx]:.4f}")