from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, learning_curve
import h5py
import pandas as pd
import matplotlib.pyplot as plt

###################################Visualize#########################################

def plot_data(data, prefix, nrows=2, ncols=2, figsize=(12, 12)):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=100)
    for i, ax in enumerate(axs.flat):
        if i < len(data.columns) - 1:
            col = data.columns[i]
            ax.scatter(data.index, data[col], s=1)
            ax.set_title(prefix + col)
            ax.set_xlabel('Time')
            ax.set_ylabel('Acceleration')
        else:
            fig.delaxes(ax)
    fig.tight_layout()
    plt.show()

Jump = pd.read_csv("dataJumping.csv", index_col=0, parse_dates=True)
Walk = pd.read_csv("dataWalking.csv", index_col=0, parse_dates=True)
Combine = pd.read_csv("dataCombined.csv", index_col=0, parse_dates=True)

plot_data(Jump, "Jumping ", nrows=2, ncols=2, figsize=(10, 10))
plot_data(Walk, "Walking ", nrows=2, ncols=2, figsize=(10, 10))
plot_data(Combine, "Combined ", nrows=2, ncols=2, figsize=(10, 10))

jump = pd.read_csv('dataJumping.csv')
walk = pd.read_csv('dataWalking.csv')

window_size = 500

###################################Preprocessing#########################################

# Calculate the moving average for the jumping data
jump_x_acc_filtered = jump['Acceleration x (m/s^2)'].rolling(window=window_size, center=True).mean()
jump_y_acc_filtered = jump['Acceleration y (m/s^2)'].rolling(window=window_size, center=True).mean()
jump_z_acc_filtered = jump['Acceleration z (m/s^2)'].rolling(window=window_size, center=True).mean()
jump_abs_acc_filtered = jump['Absolute acceleration (m/s^2)'].rolling(window=window_size, center=True).mean()

# Calculate the moving average for the walking data
walk_x_acc_filtered = walk['Acceleration x (m/s^2)'].rolling(window=window_size, center=True).mean()
walk_y_acc_filtered = walk['Acceleration y (m/s^2)'].rolling(window=window_size, center=True).mean()
walk_z_acc_filtered = walk['Acceleration z (m/s^2)'].rolling(window=window_size, center=True).mean()
walk_abs_acc_filtered = walk['Absolute acceleration (m/s^2)'].rolling(window=window_size, center=True).mean()

# Create new dataframes with the filtered acceleration data and time for each activity
filtered_jump_data = pd.DataFrame({
    'time': jump['Time (s)'],
    'x_acc': jump_x_acc_filtered,
    'y_acc': jump_y_acc_filtered,
    'z_acc': jump_z_acc_filtered,
    'abs_acc': jump_abs_acc_filtered,
    'activity': jump['Activity']
})
filtered_jump_data.dropna(inplace=True)
filtered_jump_data.to_csv('dataJumpingFiltered.csv', index=False)

filtered_walk_data = pd.DataFrame({
    'time': walk['Time (s)'],
    'x_acc': walk_x_acc_filtered,
    'y_acc': walk_y_acc_filtered,
    'z_acc': walk_z_acc_filtered,
    'abs_acc': walk_abs_acc_filtered,
    'activity': walk['Activity']
})
filtered_walk_data.dropna(inplace=True)
filtered_walk_data.to_csv('dataWalkingFiltered.csv', index=False)

# Plot for walk data
fig, bx = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), dpi=100)
bx[0].scatter(filtered_walk_data['time'], filtered_walk_data['x_acc'])
bx[0].set_ylabel('Acceleration (m/s^2)')
bx[0].set_title('Walking Filtered X Acceleration')
bx[1].scatter(filtered_walk_data['time'], filtered_walk_data['y_acc'])
bx[1].set_ylabel('Acceleration (m/s^2)')
bx[1].set_title('Walking Filtered Y Acceleration')
bx[2].scatter(filtered_walk_data['time'], filtered_walk_data['z_acc'])
bx[2].set_xlabel('Time (s)')
bx[2].set_ylabel('Acceleration (m/s^2)')
bx[2].set_title('Walking Filtered Z Acceleration')
plt.tight_layout()
plt.show()

# Plot for jump data
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), dpi=100)
ax[0].scatter(filtered_jump_data['time'], filtered_jump_data['x_acc'])
ax[0].set_ylabel('Acceleration (m/s^2)')
ax[0].set_title('Jumping Filtered X Acceleration')
ax[1].scatter(filtered_jump_data['time'], filtered_jump_data['y_acc'])
ax[1].set_ylabel('Acceleration (m/s^2)')
ax[1].set_title('Jumping Filtered Y Acceleration')
ax[2].scatter(filtered_jump_data['time'], filtered_jump_data['z_acc'])
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Acceleration (m/s^2)')
ax[2].set_title('Jumping Filtered Z Acceleration')
plt.tight_layout()
plt.show()

###################################Features#########################################

# Feature Extraction for jumping
featureJ = pd.DataFrame(columns=['mean', 'median', 'std', 'max', 'min', 'kurt', 'skew', 'range', 'var', 'energy'])
j = filtered_jump_data.iloc[:, 4]
mean = j.rolling(window=window_size).mean()
median = j.rolling(window=window_size).median()
std = j.rolling(window=window_size).std()
max_val = j.rolling(window=window_size).max()
min_val = j.rolling(window=window_size).min()
kurt = j.rolling(window=window_size).kurt()
skew = j.rolling(window=window_size).skew()
range_val = j.rolling(window=window_size).apply(lambda x: x.max() - x.min())
var = j.rolling(window=window_size).var()
energy = j.rolling(window=window_size).apply(lambda x: sum(x ** 2))
featureJ = pd.DataFrame()
featureJ['mean'] = mean
featureJ['median'] = median
featureJ['std'] = std
featureJ['max'] = max_val
featureJ['min'] = min_val
featureJ['kurt'] = kurt
featureJ['skew'] = skew
featureJ['range'] = range_val
featureJ['var'] = var
featureJ['energy'] = energy
featureJ.dropna(inplace=True)
featureJ = featureJ.assign(Key=1)
featureJ.to_csv('featureJ.csv', index=False)


# Feature Extraction for walking
featureW = pd.DataFrame(columns=['mean', 'median', 'std', 'max', 'min', 'kurt', 'skew', 'range', 'var', 'energy'])
w = filtered_walk_data.iloc[:, 4]
mean = w.rolling(window=window_size).mean()
median = w.rolling(window=window_size).median()
max_val = w.rolling(window=window_size).max()
min_val = w.rolling(window=window_size).min()
kurt = w.rolling(window=window_size).kurt()
skew = w.rolling(window=window_size).skew()
range_val = w.rolling(window=window_size).apply(lambda x: x.max() - x.min())
std = w.rolling(window=window_size).std()
var = w.rolling(window=window_size).var()
energy = w.rolling(window=window_size).apply(lambda x: sum(x ** 2))
featureW = pd.DataFrame()
featureW['mean'] = mean
featureW['median'] = median
featureW['std'] = std
featureW['max'] = max_val
featureW['min'] = min_val
featureW['kurt'] = kurt
featureW['skew'] = skew
featureW['range'] = range_val
featureW['var'] = var
featureW['energy'] = energy
featureW.dropna(inplace=True)
featureW = featureW.assign(Key=0)
featureW.to_csv('featureW.csv', index=False)
featureW.to_csv('featureW.csv', index=False)

dataNormalized = pd.concat([featureJ, featureW])
columns=['mean', 'median', 'std', 'max', 'min', 'kurt', 'skew', 'range', 'var', 'energy']
dataNormalized[columns] = MinMaxScaler().fit_transform(dataNormalized[columns])
dataNormalized.to_csv('dataNormalized.csv', index=False)

###################################Classfier#########################################

X = dataNormalized.drop("Key", axis=1)
y = dataNormalized["Key"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
scaler = StandardScaler()
logreg = LogisticRegression(max_iter=10000)
clf = make_pipeline(scaler, logreg)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# read the two CSV files
df1 = pd.read_csv('featureW.csv')
df2 = pd.read_csv('featureJ.csv')

# concatenate the two dataframes vertically
combined_df = pd.concat([df1, df2], axis=0)

# write the combined dataframe to a new CSV file
combined_df.to_csv('Features.csv', index=False)

dfw = pd.read_csv('Features.csv')

###################################Data Storing#########################################

with h5py.File('dataset.hdf5', 'w') as hdf:
    # Create a group to store the train and test data
    dataset = hdf.create_group('dataset')

    # Create a subgroup for the train data
    train_group = dataset.create_group('Train')
    train_group.create_dataset('TrainData', data=X_train)
    train_group.create_dataset('TrainKey', data=y_train)

    # Create a subgroup for the test data
    test_group = dataset.create_group('Test')
    test_group.create_dataset('TestData', data=X_test)
    test_group.create_dataset('TestKey', data=y_test)