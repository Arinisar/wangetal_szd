import keras
from keras import Model
import numpy as np
import os
import  json

test_file = os.path.abspath('./data/test.npz')
best_model = os.path.abspath('./model/best.h5')
labels_file = os.path.abspath('./labels.json')
results_file = os.path.abspath('./results.txt')
model_plot = os.path.abspath('./model/best.png')

test_data = np.load(test_file)
x_test, y_test = test_data['x'], test_data['y']

with open(labels_file) as json_file:
    classes = json.load(json_file)
    labels = classes[str(1)]
#   tags = classes['tags']

model = keras.models.load_model(best_model)
keras.utils.plot_model(model,model_plot,
                       show_shapes=True,
                       expand_nested=True)

predictions = model.predict(x_test)
# for x in x_test:
#     prediction = model(x, training=False, verbose=1)
#     print(prediction)
results= []
correct_preds = 0
for i in range(len(predictions)):
    top_three_pred = np.argpartition(predictions[i], -3)[-3:]
    top_three_pred = np.flip(top_three_pred[np.argsort(predictions[i][top_three_pred])])
    top_three_labels = []
    top_three_values = []
    top_three_labels_values = []
    for lno in top_three_pred:
        top_three_labels.append(labels[lno])
        top_three_values.append(predictions[i][lno])
        top_three_labels_values.append([top_three_labels[-1],top_three_values[-1]])
    correct_result = labels[y_test[i]]
    if correct_result == top_three_labels[0]:
        correct_preds += 1
    results.append([top_three_labels_values, correct_result])

corr_percentage = correct_preds / len(predictions)
print(corr_percentage)

with open(results_file, 'w') as output:
    for l in range(len(results)):
        line = ''
        for n in range(len(results[l][0])):
            line = line + results[l][0][n][0] + ': ' + str(results[l][0][n][1]) + ' '
        line = line +  'Corr: ' + results[l][1] + '\n'
        output.write(line)





