from model import AdaBoostModel
from sklearn.model_selection import train_test_split

model = AdaBoostModel()

features, label = model.read_data()

# It will take 2 hours to plot the learning curves
# model.plot_learning_curves(features, label)

# It will take 1 hour to plot the model complexity graph
# model.plot_model_complexity(features, label)

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

model.train_model(x_train, y_train)

model.test_model(x_test, y_test)

model.plot_confusion_matrix(x_test, y_test)