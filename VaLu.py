import numpy as np
import matplotlib.pyplot as plt
import pickle
def VL(model,X_test,X_validation,y_test,y_validation):
    print("Accuracy:", model.score(X_test, y_test))
    validation_predictions = [model.predict(x) for x in X_validation]
    validation_accuracy = np.mean(np.array(validation_predictions) == y_validation)
    print("Validation Accuracy:", validation_accuracy)
    predictions = [model.predict(x) for x in X_test]
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', label='True Labels')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap='Set1', marker='x', label='Predictions')
    plt.legend()
    plt.title("SVM Classification")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
# 保存模型
def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

# 加载模型
def load_model(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model